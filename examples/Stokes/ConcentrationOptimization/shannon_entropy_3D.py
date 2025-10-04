"""This code can be used to optimize the Shannon entropy of 3D static mixers.
The Shannon entropy calculation is deferred to postmafs. Therefore, at every step
a Paraview has to be generated so that postmafs can read it and analyze the streamlines
"""

import sys

import numpy as np
import postmafs as pm
import scipy.optimize as scopt
import splinepy as sp
from geometry_kernel import SMXKernel
from splinepy.utils.data import cartesian_product as _cartesian_product

import pygadjoints

sys.path.insert(0, "../TemperatureOptimization")
from export_helpers import AdditionalBlocks, export

EPS = 1e-8
BOX_LENGTH = 0.124
BOX_HEIGHT = 0.04
SHOW_MICROSTRUCTURE = False
FILENAME = "static_mixer_working.xml"
N_THREADS = 1
sp.settings.NTHREADS = N_THREADS

# Material parameters
DENSITY = 736
VISCOSITY = 6000
HEAT_CAPACITY = 2900
THERMAL_DIFFUSIVITY = 1.1997e-7

# Simulation parameters
TILING = [2, 2, 4]
TWISTING_LAYERS = [1, 2]
# How much percent of tile length should be reserved for linking tiles
LINKAGE_THICKNESS = 0.2
# How much percent of box length should be reserved for forerun (the same length will
# be applied for the afterrun)
FORERUN_AFTERRUN_THICKNESS = 0.2
# Determines the percentage of the whole forerun length to be dedicated to the linkage
FORERUN_AFTERRUN_LINKAGE_LENGTH = 0.1
N_REFINEMENTS = 0
DEGREE_ELEVATIONS = 0
INLET_BOUNDARY_ID = 2
OUTLET_BOUNDARY_ID = 3
INLET_PEAK_VELOCITY = 66342.5611413043
OBJECTIVE_FUNCTION = [1]  # 1: pressure loss, 2: deviation from
# mean temperature
OBJECTIVE_FUNCTION_WEIGHTS = [1, 1216]


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
        h_refinements_temperature=0,
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
        self.h_refinements_temperature = h_refinements_temperature
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
            refinements_temperature=self.h_refinements_temperature,
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
        # # Heat simulation
        # self.pde.assemble_heat_problem()
        # self.pde.solve_heat_linear_system()

    def evaluate_objective_functions(self):
        return self.pde.compute_objective_function_values()

    def evaluate_streamline_shannon_entropy(
        self,
        velocity_file,
        nx_streams=101,
        ny_streams=101,
        nbins_x=5,
        nbins_y=5,
    ):
        field_name = "SolutionField"
        integration_direction = "forward"
        streamline_file = "streamlines"
        streamline_interpolated_file = streamline_file + "_interpolated"
        streamline_direction = "z"

        # Read the velocity field
        stream_class = pm.streamlines.Streamlines(velocity_file)

        # Compute streamlines
        stream_class.add_plane_as_streamline_source(
            nx=nx_streams, ny=ny_streams
        )
        stream_class.compute_streamlines(
            field_name=field_name,
            integration_direction=integration_direction,
            max_length=BOX_LENGTH * 2,
            progress_bar=True,
        )

        # Save the streamlines
        stream_class.save_streamlines(streamline_file)

        # Analyze the streamlines
        ana = pm.analysis.Analysis(
            streamline_file + ".npz", streamline_direction
        )
        ana.set_grid_dimensions(nx=nbins_x, ny=nbins_y)
        ana.determine_successful_streamlines(
            cutoff_distance=BOX_LENGTH * (1 + 0.8 * FORERUN_AFTERRUN_THICKNESS)
        )
        ana.compute_start_and_end_grid_indices()

        z_sampling = BOX_LENGTH * np.array(
            [-FORERUN_AFTERRUN_THICKNESS, 1 + 0.8 * FORERUN_AFTERRUN_THICKNESS]
        )
        ana.interpolate_streamlines_and_determine_grid_index(
            z_sampling, savefile=streamline_interpolated_file
        )

        interpolated = pm.analysis.InterpolatedValues(
            streamline_interpolated_file + ".npz"
        )
        interpolated.compute_shannon_entropy_evolution(normalize_values=True)
        entropy_value = interpolated._shannon_entropy_list[-1]

        return entropy_value

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
        include_streamline_objective=False,
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
        if include_streamline_objective:
            n_objective_functions += 1
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
        self.include_streamline_objective = include_streamline_objective

        # Prepare initial optimization parameters
        # TODO: macro sensitivities: currently does not take x- and y-coordinates of
        # macro cps into consideration
        initial_parameter_spline_values = np.array(
            self.geometry_kernel._parameter_spline.cps
        ).ravel()
        self.n_design_vars_para = len(initial_parameter_spline_values)
        if self.morph_macro_spline:
            initial_macro_cps = self.geometry_kernel._macro_spline_initial.cps
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
        # Include streamline-based objective if required
        if self.include_streamline_objective:
            intermediate_paraview_file = "ParaviewOutput/intermediate_results"
            self.simulation_kernel.save_geometry(intermediate_paraview_file)
            shannon_entropy = (
                self.simulation_kernel.evaluate_streamline_shannon_entropy(
                    f"{intermediate_paraview_file}_velocity.pvd"
                )
            )
            self.current_objective_values += [shannon_entropy]

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
    # Create initial parameter spline with controls in the corners and one layer (of 4
    # points) in the middle into the z-direction
    macro_spline_initial = sp.BSpline(
        degrees=[1, 1, 1],
        knot_vectors=[[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 0.5, 1, 1]],
        control_points=_cartesian_product(
            [
                np.array([0, BOX_HEIGHT]),
                np.array([0, BOX_HEIGHT]),
                np.array([0, BOX_LENGTH / 2, BOX_LENGTH]),
            ]
        ),
    )

    # Define identifier functions for microstructure boundaries
    def identifier_inlet(points):
        return points[:, 2] + FORERUN_AFTERRUN_THICKNESS * BOX_LENGTH < EPS

    def identifier_outlet(points):
        return (
            points[:, 2] > (1 + FORERUN_AFTERRUN_THICKNESS) * BOX_LENGTH - EPS
        )

    boundary_identifier_dict = {
        identifier_inlet: INLET_BOUNDARY_ID,
        identifier_outlet: OUTLET_BOUNDARY_ID,
    }

    # Prepare parameter spline
    parameter_spline_initial = sp.BSpline(
        degrees=[1, 1, 1],
        knot_vectors=macro_spline_initial.kvs,
        control_points=0.05 * np.ones((12, 1)),
    )

    geometry_kernel = SMXKernel(
        box_dimensions=[BOX_HEIGHT, BOX_HEIGHT, BOX_LENGTH],
        tiling=TILING,
        twisting_layers=TWISTING_LAYERS,
        linkage_thickness=LINKAGE_THICKNESS,
        forerun_thickness=FORERUN_AFTERRUN_THICKNESS,
        forerun_linkage_length=FORERUN_AFTERRUN_LINKAGE_LENGTH,
        parameter_spline_initial=parameter_spline_initial,
        macro_spline_initial=macro_spline_initial,
        boundary_identifier_dict=boundary_identifier_dict,
    )

    geometry_kernel.generate_microstructure()
    # geometry_kernel.show_microstructure()

    # Simulation parameters
    # Prepare for xml-file export
    additional_blocks = AdditionalBlocks()
    # Velocity and pressure boundary conditions
    additional_blocks.add_boundary_conditions(
        block_id=1,
        dim=3,
        function_list=[
            ("0.0", "0.0", "0.0"),
            (
                "0.0",
                "0.0",
                f"{INLET_PEAK_VELOCITY} * x * ({BOX_HEIGHT}-x) y * ({BOX_HEIGHT}-y)",
            ),
            "0.0",
        ],
        bc_list=[
            (f"BID{INLET_BOUNDARY_ID}", "Dirichlet", 1, 0),  # Inlet
            ("BID1", "Dirichlet", 0, 0),  # Walls
            (f"BID{OUTLET_BOUNDARY_ID}", "Dirichlet", 2, 1),  # Pressure BCs
        ],
        multipatch_id=0,
        comment=" Velocity and pressure boundary conditions: bi-parabolic inflow field ",
    )

    # Temperature boundary conditions (right now not needed)
    half_length = BOX_HEIGHT / 2
    additional_blocks.add_boundary_conditions(
        block_id=66,
        dim=3,
        function_list=[
            "0.0",
            f"if(x >= {half_length}, 0, if (y >= {half_length}, 0, 1))",
        ],
        bc_list=[
            ("BID1", "Neumann", 0, 0),
            (f"BID{OUTLET_BOUNDARY_ID}", "Neumann", 0, 0),
            (f"BID{INLET_BOUNDARY_ID}", "Dirichlet", 1, 0),
        ],
        multipatch_id=0,
        comment=" Temperature boundary conditions ",
    )

    # Body force
    additional_blocks.add_function(
        dim=3,
        block_id=100,
        function_string=("0.0", "0.0", "0.0"),
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

    optimization_macro_indices = {4: 2, 5: 2, 6: 2, 7: 2}

    optimizer = OptimizationKernel(
        geometry_kernel=geometry_kernel,
        simulation_kernel=simulation_kernel,
        optimization_method="COBYQA",
        scaling_factors_objective_function=OBJECTIVE_FUNCTION_WEIGHTS,
        optimization_macro_indices=optimization_macro_indices,
        include_streamline_objective=True,
    )

    bounds = [(0.04, 0.45) for _ in range(optimizer.n_design_vars_para)]

    # Set bounds for macro cp movement
    amp_factor = 0.15
    bounds += [
        (BOX_LENGTH * amp_factor, BOX_LENGTH * (1 - amp_factor)),
        (BOX_LENGTH * amp_factor, BOX_LENGTH * (1 - amp_factor)),
        (BOX_LENGTH * amp_factor, BOX_LENGTH * (1 - amp_factor)),
        (BOX_LENGTH * amp_factor, BOX_LENGTH * (1 - amp_factor)),
    ]

    optimizer.optimize(bounds=bounds)
    optimizer.finalize()
