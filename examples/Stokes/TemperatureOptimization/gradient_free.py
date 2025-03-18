"""examples/Stokes/TemperatureOptimization/gradient_free.py

Example of gradient-free optimization of a microstructure in fluid flow.
Optimization parameters are the tile parameters.

"""

import numpy as np
import scipy.optimize as scopt
import splinepy as sp
from export_helpers import AdditionalBlocks, export

import pygadjoints

EPS = 1e-8
BOX_LENGTH = 0.14
BOX_HEIGHT = 0.04
SHOW_MICROSTRUCTURE = False
FILENAME = "microstructure_working.xml"
N_THREADS = 1
sp.settings.NTHREADS = N_THREADS

# Material parameters
DENSITY = 736
VISCOSITY = 6000
HEAT_CAPACITY = 2900
THERMAL_DIFFUSIVITY = 1.1997e-7

# Simulation parameters
TILING = [6, 3]
N_REFINEMENTS = 0
DEGREE_ELEVATIONS = 0
INLET_BOUNDARY_ID = 2
OUTLET_BOUNDARY_ID = 3
INLET_PEAK_VELOCITY = 1.4336534897721067
CLOSING_FACE = "x"
OBJECTIVE_FUNCTION = [2]
OBJECTIVE_FUNCTION_WEIGHTS = [1]


# TODO: generator.parameter_sensitivity_function
class MicrostructureKernel:
    """Class to generate to microstructure geometry"""

    def __init__(
        self,
        initial_macro_spline,
        microtile,
        tiling,
        parameter_spline_degrees=None,
        parameter_spline_cps_dimensions=None,
        initial_parameter_value=None,
        initial_parameter_spline=None,
        closing_face=None,
        additional_parameters={},
        boundary_identifier_dict=None,
    ):
        """
        Initialize the geometry driver. Parameter may be given either as spline
        object or via degrees, cps dimensions and initial value.

        Parameters
        --------------
        initial_macro_spline: spline
            Initial macro spline. Might deform later
        microtile: splinepy.microstructure.tiles
            The used microtile in the microstructure
        tiling: list<int>
            The number of tiles in each physical direction
        parameter_spline_degrees: list<int> (optional)
            Spline degrees for parameter spline
        parameter_spline_cps_dimensions: list<int> (optional)
            Number of control points in each direction for parameter spline
        initial_parameter_value: float/list<float>/np.ndarray
            Initial value(s) for parameter spline. If tile takes multiple parameters,
            multiple values must be given.
        intial_parameter_spline: spline (optional)
            Initial spline for tile parametrization. Might change during
            optimization.
        closing_face: str (optional)
            Direction in which the whole microstructure will be closed
        additional_parameters: dict (optional)
            Additional parameters (e.g. contact_length) for microstructure generation
        boundary_identifier_dict: dict<callable, int>
            Dictionary of boundary identifier functions and corresponding boundary IDs
        """
        self.macro_spline_initial = initial_macro_spline.copy()
        self.microtile = microtile
        self.n_tile_parameters = microtile._n_info_per_eval_point
        self.parameters_shape = [
            len(microtile._evaluation_points),
            self.n_tile_parameters,
        ]
        self.tiling = tiling

        # Parameter spline
        if initial_parameter_spline is not None:
            assert (
                parameter_spline_degrees is None
                and parameter_spline_cps_dimensions is None
                and initial_parameter_value is None
            ), (
                "If parameter spline is given, other values do not have to be"
                + "implemented"
            )
            # Assert that parameter spline has the right number of parameters
            assert (
                initial_parameter_spline.cps.shape[1] == self.n_tile_parameters
            )
            self.parameter_spline = initial_parameter_spline
        elif initial_parameter_value is not None:
            assert (
                parameter_spline_degrees is not None
                and parameter_spline_cps_dimensions is not None
            ), "Ensure that other parameter spline information is provided"
            knot_vectors = [
                np.linspace(0, 1, n_cps - degree + 1)
                for n_cps, degree in zip(
                    parameter_spline_cps_dimensions, parameter_spline_degrees
                )
            ]
            knot_vectors = [
                np.hstack(([0] * degree, knot_vector, [1] * degree))
                for knot_vector, degree in zip(
                    knot_vectors, parameter_spline_degrees
                )
            ]
            self.parameter_spline = sp.BSpline(
                degrees=parameter_spline_degrees,
                control_points=initial_parameter_value
                * np.ones(
                    (
                        np.prod(parameter_spline_cps_dimensions),
                        self.n_tile_parameters,
                    )
                ),
                knot_vectors=knot_vectors,
            )
        else:
            raise NotImplementedError(
                "Microstructure without any parameters not implemented!"
            )
        self.parameter_spline_initial = self.parameter_spline.copy()

        self.generator = sp.microstructure.Microstructure(
            deformation_function=initial_macro_spline.copy(),
            tiling=tiling,
            microtile=microtile,
            parametrization_function=self.parametrization_function,
        )

        self.closing_face = closing_face
        self.additional_parameters = additional_parameters
        assert isinstance(
            boundary_identifier_dict, dict
        ), "identifer_dict must be a dictionary"
        self.boundary_identifier_dict = boundary_identifier_dict

        # Make interfaces reusable
        self.interfaces = None
        self.multipatch = None

    # TODO: if tile has multiple parameters, evaluate will just output one value
    # and repeat it
    def parametrization_function(self, points):
        return self.parameter_spline.evaluate(points)

    # TODO: check what this does and if it is correct
    def parameter_sensitivity_function(self, points):
        n_points = points.shape[0]
        basis_function_matrix = np.zeros(
            (n_points, self.parameter_spline.cps.shape[0])
        )
        basis_functions, support = self.parameter_spline.basis_and_support(
            points
        )
        np.put_along_axis(
            basis_function_matrix, support, basis_functions, axis=1
        )
        return np.tile(
            basis_function_matrix.reshape(n_points, 1, -1), [1, 2, 1]
        )

    def generate_microstructure(self, macro_sensitivities=None):
        self.multipatch = self.generator.create(
            closing_face=self.closing_face,
            macro_sensitivites=macro_sensitivities,
            **self.additional_parameters,
        )

        # Reuse existing interfaces
        if self.interfaces is None:
            self.multipatch.determine_interfaces()
            self.interfaces = self.multipatch.interfaces
        else:
            self.multipatch.interfaces = self.interfaces

        # Assign boundaries from identifier functions
        for (
            identifier_function,
            boundary_id,
        ) in self.boundary_identifier_dict.items():
            self.multipatch.boundary_from_function(
                identifier_function, boundary_id=boundary_id
            )

    def get_multipatch(self):
        return self.multipatch

    def update_parameter_spline(self, new_spline_parameters):
        self.parameter_spline.cps[:] = new_spline_parameters.reshape(
            (-1, self.n_tile_parameters)
        )

    def update_macro_spline(self, new_values, cp_indices, cp_directions):
        self.generator.deformation_function.cps[
            cp_indices, cp_directions
        ] = new_values

    def show_current_geometry(self):
        self.generator.show(
            closing_face=self.closing_face, **self.additional_parameters
        )

    def show_initial_geometry(self):
        def initial_parametrization_function(points):
            return self.parameter_spline_initial.evaluate(points)

        initial_generator = sp.microstructure.Microstructure(
            deformation_function=self.macro_spline_initial,
            tiling=self.tiling,
            microtile=self.microtile,
            parametrization_function=initial_parametrization_function,
        )

        initial_generator.show(
            closing_face=self.closing_face, **self.additional_parameters
        )

    def show_boundaries(self):
        assert (
            self.multipatch is not None
        ), "Multipatch must first be initialized"
        n_bds = len(self.multipatch.boundaries)
        sp.show(
            *[
                [f"Boundary {i}", self.multipatch.boundary_multipatch(i)]
                for i in range(1, n_bds + 1)
            ],
            use_saved=True,
            control_points=False,
        )


class SimulationKernel:
    def __init__(
        self,
        pde_problem,
        n_threads,
        material_constants,
        filename,
        gismo_export_options,
        objective_function_types,
        h_refinements=0,
        degree_elevations=0,
        print_summary=False,
        is_nonlinear_pde=False,
    ):
        """Initializees the simulation kernel with a given PDE problem

        Parameters (TODO)
        -----------
        pde_problem: pygadjoints PdeProblem
            PdeProblem module from pygadjoints
        n_threads: int
            Number of threads used for the simulation
        material_constants: dict
            Material constants for the PDE
        is_nonlinear_pde: bool (default: False)
            Set to True if PDE is nonlinear and nonlinear solver should be used
        """
        self.pde = pde_problem()
        self.pde.set_number_of_threads(nthreads=n_threads)
        self.pde.set_material_constants(**material_constants)

        if isinstance(objective_function_types, int):
            self.objective_function_types = [objective_function_types]
        else:
            self.objective_function_types = objective_function_types
        for objective_function_type in self.objective_function_types:
            self.pde.add_objective_function(objective_function_type)

        self.filename = filename
        self.h_refinements = h_refinements
        self.degree_elevations = degree_elevations
        self.print_summary = print_summary
        self.gismo_export_options = gismo_export_options
        self.is_nonlinear_pde = is_nonlinear_pde
        if is_nonlinear_pde:
            raise NotImplementedError("Nonlinear PDEs not implemented")

    def initialize(self):
        self.pde.init(
            fname=self.filename,
            refinements=self.h_refinements,
            degree_elevations=self.degree_elevations,
            print_summary=self.print_summary,
        )

    def prepare_simulation(self, multipatch_geometry):
        export(
            fname=self.filename,
            multipatch=multipatch_geometry,
            indent=True,
            additional_blocks=self.gismo_export_options,
            as_base64=False,
        )

    def update_geometry(self):
        self.pde.update_geometry(fname=self.filename, topology_changes=False)

    def forward_simulation(self):
        # Fluid simulations
        self.pde.assemble_fluid_problem()
        self.pde.solve_fluid_linear_system()
        # Heat simulation
        self.pde.assemble_heat_problem()
        self.pde.solve_heat_linear_system()

    def evaluate_objective_functions(self):
        return self.pde.compute_objective_function_values()

    def save_geometry(self, filename):
        self.pde.export_paraview(filename=filename, sample_rate=32**2)


class OptimizationKernel:
    def __init__(
        self,
        geometry_kernel,
        simulation_kernel,
        optimization_method,
        scaling_factors_objective_function=None,
        optimization_macro_indices=None,
        write_logfiles=False,
    ):
        """Initialize the optimization kernel

        Parameters
        -----------
        geometry_kernel: MicrostructureKernel
            Module for microstructure geometry generation
        simulation_kernel: SimulationKernel
            Module for simulation
        scaling_factor_objective_function: float
            Factor to scale the objective function with
        optimization_macro_indices: dict<int: list<int>/int>
            Indices of macro spline's control points which are allowed to change
            during the shape optimization. The dict entries correspond to in which
            directions the control points are allowed to be changed
        write_logfiles: bool
            If True, write logfiles
        """
        self.geometry_kernel = geometry_kernel
        self.simulation_kernel = simulation_kernel
        n_objective_functions = len(simulation_kernel.objective_function_types)
        if scaling_factors_objective_function is None:
            self.scaling_factors_objective_function = [
                1.0
            ] * n_objective_functions
        else:
            assert (
                len(scaling_factors_objective_function)
                == n_objective_functions
            ), "Each objective function must have a separate weight"
            self.scaling_factors_objective_function = (
                scaling_factors_objective_function
            )
        self.morph_macro_spline = optimization_macro_indices is not None
        self.write_logfiles = write_logfiles

        # Prepare initial optimization parameters
        # TODO: macro sensitivities: currently does not take x- and y-coordinates of
        # macro cps into consideration
        initial_parameter_spline_values = np.array(
            self.geometry_kernel.parameter_spline.cps
        ).ravel()
        self.n_design_vars_para = len(initial_parameter_spline_values)
        if self.morph_macro_spline:
            initial_macro_cps = self.geometry_kernel.macro_spline_initial.cps
            optimization_macro_cp_indices = []
            optimization_macro_cp_directions = []
            initial_macro_cp_values = []
            # Get which macro control points are allowed to be moved during the
            # optimization, into which direction and what their initial value is
            for cp_id, morph_directions in optimization_macro_indices.items():
                if isinstance(morph_directions, list):
                    for md in morph_directions:
                        optimization_macro_cp_indices.append(cp_id)
                        optimization_macro_cp_directions.append(md)
                        initial_macro_cp_values.append(
                            initial_macro_cps[cp_id, md]
                        )
                else:
                    optimization_macro_cp_indices.append(cp_id)
                    optimization_macro_cp_directions.append(morph_directions)
                    initial_macro_cp_values.append(
                        initial_macro_cps[cp_id, morph_directions]
                    )
            len(optimization_macro_cp_indices)
            self.optimization_macro_cp_indices = np.array(
                optimization_macro_cp_indices
            )
            self.optimization_macro_cp_directions = np.array(
                optimization_macro_cp_directions
            )
            optimization_parameters_initial = np.hstack(
                (
                    initial_parameter_spline_values,
                    np.array(initial_macro_cp_values),
                )
            )
        else:
            optimization_parameters_initial = initial_parameter_spline_values
        self.optimization_parameters_initial = optimization_parameters_initial
        self.n_optimization_parameters = len(optimization_parameters_initial)

        self.iteration = 0
        self.last_optimization_parameters = None

        # Prepare scipy.optimize dict
        self.scipy_optimze_dict = {
            "fun": self.evaluate_iteration,
            "x0": self.optimization_parameters_initial,
            "method": optimization_method,
            "options": {"disp": True},
            "tol": 1e-4,
        }
        self.optimizer = None

    def split_optimization_parameters(self, optimization_parameters):
        spline_parameters = optimization_parameters[: self.n_design_vars_para]
        macro_cps_parameters = optimization_parameters[
            self.n_design_vars_para :
        ]
        return spline_parameters, macro_cps_parameters

    def update_parameters(
        self, current_optimization_parameters, increase_count=True
    ):
        """

        Returns
        ---------
        have_parameters_changed: bool
            If True, parameters have been updated.
        """
        # Check if anything changed since last call
        if self.last_optimization_parameters is not None and np.allclose(
            self.last_optimization_parameters, current_optimization_parameters
        ):
            # Return and indicate that parameters have not changed
            return False

        if increase_count:
            self.iteration += 1

        # Update microstructure
        (
            current_spline_parameters,
            macro_cps_parameters,
        ) = self.split_optimization_parameters(
            optimization_parameters=current_optimization_parameters
        )
        self.geometry_kernel.update_parameter_spline(current_spline_parameters)
        if self.morph_macro_spline:
            self.geometry_kernel.update_macro_spline(
                new_values=macro_cps_parameters,
                cp_indices=self.optimization_macro_cp_indices,
                cp_directions=self.optimization_macro_cp_directions,
            )
        # for gradient free macro sensitivities are not needed
        self.geometry_kernel.generate_microstructure(macro_sensitivities=None)

        # Prepare geometry for simulation
        microstructure = self.geometry_kernel.get_multipatch()
        self.simulation_kernel.prepare_simulation(microstructure)
        if self.last_optimization_parameters is None:
            self.simulation_kernel.initialize()
        else:
            self.simulation_kernel.update_geometry()

        # PDE.read_control_point_sensitivities(filename)
        # PDE.get_control_point_sensitivities()

        # Set new parameters as old ones
        self.last_optimization_parameters = (
            current_optimization_parameters.copy()
        )

        # TODO: check if this is necessary
        # self.current_objective_function_value = None
        # self.ctps_sensitivity = None

        # Indicate that parameters have been changed
        return True

    def evaluate_iteration(self, current_optimization_parameters):
        # Update optimization parameters
        have_parameters_changed = self.update_parameters(
            current_optimization_parameters=current_optimization_parameters
        )

        # Return current objective function value if there optimization parameters
        # have not changed
        if not have_parameters_changed:
            return self.current_objective_function_value

        # Perform forward simulation
        self.simulation_kernel.forward_simulation()

        # Update objective function value
        self.current_objective_values = (
            self.simulation_kernel.evaluate_objective_functions()
        )
        print("----------", self.current_objective_values)
        self.current_objective_function_value = sum(
            [
                scaling * objective_value
                for scaling, objective_value in zip(
                    self.scaling_factors_objective_function,
                    self.current_objective_values,
                )
            ]
        )

        # For first iteration save initial geometry
        if self.iteration == 1:
            self.simulation_kernel.save_geometry(
                filename="ParaviewOutput/multipatch_initial"
            )

        # Write to logfile
        self.write_logfile(
            filename="log_parameters_macro.csv",
            values=current_optimization_parameters,
        )

        return self.current_objective_function_value

    def evaluate_jacobian(self, current_optimization_parameters):
        raise NotImplementedError(
            "The Jacobian evaluation is not yet implemented!"
        )

    def optimize(self, jacobian_provided=False, bounds=None):
        if jacobian_provided:
            self.scipy_optimze_dict["jac"] = self.evaluate_jacobian
        if bounds is not None:
            print(len(bounds), self.n_optimization_parameters)
            assert (
                len(bounds) == self.n_optimization_parameters
            ), "Bounds must have the same length as the optimization parameters"
        else:
            raise NotImplementedError("No bounds is not implemented")
        self.scipy_optimze_dict["bounds"] = bounds
        # TODO: constraints, bounds
        self.optimizer = scopt.minimize(**self.scipy_optimze_dict)

    def finalize(self):
        assert self.optimizer is not None, "Optimization has not started yet!"

        print("Best parameters: ")
        print(self.optimizer.x)
        print(self.optimizer)

        # raise NotImplementedError("Finalization not yet implemented")
        # TODO: with best parameters: forward simulation, objective function
        self.update_parameters(self.optimizer.x, increase_count=False)
        self.simulation_kernel.forward_simulation()
        self.simulation_kernel.save_geometry("ParaviewOutput/multipatch_final")

    def plot_objective_function(self):
        raise NotImplementedError(
            "Plotting the evolution of the objective function not implemented"
        )

    def write_logfile(self, filename, values, include_objective_value=True):
        with open(filename, "a") as f:
            values_to_write = [self.iteration]
            if include_objective_value:
                values_to_write.append(self.current_objective_function_value)
                values_to_write += self.current_objective_values
            values_to_write += list(values)
            newline = ", ".join([str(value) for value in values_to_write])
            f.write(newline + "\n")


if __name__ == "__main__":
    # Define microstructure deformation function
    initial_macro_spline = sp.Bezier(
        degrees=[2, 2],
        control_points=sp.utils.data.cartesian_product(
            [np.linspace(0, BOX_LENGTH, 3), np.linspace(0, BOX_HEIGHT, 3)]
        ),
    )

    # Define identifier functions for microstructure boundaries
    def identifier_inlet(points):
        return points[:, 0] < EPS

    def identifier_outlet(points):
        return points[:, 0] > BOX_LENGTH - EPS

    boundary_identifier_dict = {
        identifier_inlet: INLET_BOUNDARY_ID,
        identifier_outlet: OUTLET_BOUNDARY_ID,
    }

    # Describe initial parameter spline such that every tile is parametrized
    # knot_vectors = [np.linspace(0, 1, val + 1) for val in TILING]
    # knot_vectors = [np.hstack((kv[0], kv, kv[-1])) for kv in knot_vectors]
    # knot_vectors = [
    #     [0.0, 0.0, 1/3, 2/3, 1.0, 1.0],
    #     [0.0, 0.0, 1/3, 2/3, 1.0, 1.0]
    # ]
    knot_vectors = [[0.0, 0.0, 1.0, 1.0]] * 2

    default_parameter_value = 0.2
    control_points = default_parameter_value * np.ones(
        (np.prod([len(kv) - 2 for kv in knot_vectors]), 1)
    )

    initial_parameter_spline = sp.BSpline(
        degrees=[1, 1],
        knot_vectors=knot_vectors,
        control_points=control_points,
    )

    geometry_kernel = MicrostructureKernel(
        initial_macro_spline=initial_macro_spline,
        microtile=sp.microstructure.tiles.SMX2DInverse(),
        tiling=TILING,
        initial_parameter_spline=initial_parameter_spline,
        closing_face=CLOSING_FACE,
        boundary_identifier_dict=boundary_identifier_dict,
    )

    # Simulation parameters
    # Prepare for xml-file export
    additional_blocks = AdditionalBlocks()
    # Velocity and pressure boundary conditions
    additional_blocks.add_boundary_conditions(
        block_id=1,
        dim=2,
        function_list=[
            ("0.0", "0.0"),
            (f"{INLET_PEAK_VELOCITY} * y * ({BOX_HEIGHT}-y)", "0"),
            "0.0",
        ],
        bc_list=[
            (f"BID{INLET_BOUNDARY_ID}", "Dirichlet", 1, 0),  # Inlet
            ("BID1", "Dirichlet", 0, 0),  # Walls
            ("BID3", "Dirichlet", 2, 1),  # Pressure BCs
        ],
        multipatch_id=0,
        comment=" Velocity and pressure boundary conditions: parabolic inflow field ",
    )

    # Temperature boundary conditions
    additional_blocks.add_boundary_conditions(
        block_id=66,
        dim=2,
        function_list=[
            "0",
            "-150000000.0*y^4 + 12000000.0*y^3 - 426645.0*y^2 + 7465.8*y + 196.7645",
            "-683300000.0*y^4 + 54664000.0*y^3 - 1367620.0*y^2 + 10973.6*y + 199.592",
            # Concentration profile
            "17080000.0*y^4 - 1366400.0*y^3 + 34184.0*y^2 - 274.24*y + 0.9596",
        ],
        bc_list=[
            (f"BID{INLET_BOUNDARY_ID}", "Dirichlet", 1, 0),
            (f"BID{OUTLET_BOUNDARY_ID}", "Neumann", 0, 0),
            ("BID1", "Neumann", 0, 0),
        ],
        multipatch_id=0,
        comment=" Temperature boundary condition ",
    )

    # Body force
    additional_blocks.add_function(
        dim=2,
        block_id=100,
        function_string=("0.0", "0.0"),
        comment=" Body forces ",
    )

    # Get default assembly options
    additional_blocks.add_assembly_options(
        block_id=10, comment=" Assembler options "
    )

    fluid_material_constants = {
        "viscosity": VISCOSITY,
        "density": DENSITY,
        "heat_capacity": HEAT_CAPACITY,
        "thermal_diffusivity": THERMAL_DIFFUSIVITY,
    }

    gismo_export_options = additional_blocks.to_list()

    simulation_kernel = SimulationKernel(
        pde_problem=pygadjoints.StokesTemperatureProblem,
        n_threads=N_THREADS,
        material_constants=fluid_material_constants,
        filename=FILENAME,
        gismo_export_options=gismo_export_options,
        h_refinements=N_REFINEMENTS,
        degree_elevations=DEGREE_ELEVATIONS,
        print_summary=True,
        objective_function_types=OBJECTIVE_FUNCTION,
    )

    optimization_macro_indices = {
        1: 0,
        # 3: 1,
        # 4: [0,1],
        # 5: 1,
        # 7: 0
    }

    optimizer = OptimizationKernel(
        geometry_kernel=geometry_kernel,
        simulation_kernel=simulation_kernel,
        optimization_method="COBYQA",
        scaling_factors_objective_function=OBJECTIVE_FUNCTION_WEIGHTS,
        optimization_macro_indices=optimization_macro_indices,
    )

    bounds = [(0.04, 0.49) for _ in range(optimizer.n_design_vars_para)]

    # Set bounds for macro cp movement
    amp_factor = 0.05
    bounds += [
        (BOX_LENGTH * amp_factor, BOX_LENGTH * (1 - amp_factor)),
        # (BOX_HEIGHT*amp_factor, BOX_HEIGHT*(1-amp_factor)),
        # (BOX_LENGTH*amp_factor, BOX_LENGTH*(1-amp_factor)),
        # (BOX_HEIGHT*amp_factor, BOX_HEIGHT*(1-amp_factor)),
        # (BOX_HEIGHT*amp_factor, BOX_HEIGHT*(1-amp_factor)),
        # (BOX_LENGTH*amp_factor, BOX_LENGTH*(1-amp_factor))
    ]

    optimizer.optimize(bounds=bounds)
    optimizer.finalize()
