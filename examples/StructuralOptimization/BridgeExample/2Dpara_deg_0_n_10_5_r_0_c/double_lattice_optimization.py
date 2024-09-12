import numpy as np
import scipy
import splinepy as sp

import pygadjoints

###
# SIMULATION PARAMETERS
###
ULTRA_VERBOSE = True
N_THREAD = 1

###
# MATERIAL PARAMETERS (PETG)
###
ACTIVATE_SOURCE_FUNCTION = False
Youngs_modulus = 1.5e3
poisson_ratio = 0.4
lame_lambda_ = (
    Youngs_modulus
    * poisson_ratio
    / ((1 - 2 * poisson_ratio) * (1 + poisson_ratio))
)
lame_mu_ = Youngs_modulus / (2 * (1 + poisson_ratio))
number_of_tiles_with_load = 1
density_ = 1

source_function_ = [0.0, 0.0]
neumann_force_ = [0, -100]
dirichlet_value = [0.0, 0.0]
dim = 2

print(f"Youngs Modulus    : {Youngs_modulus}")
print(f"Poisson's ratio   : {poisson_ratio}")
print(f"First Lame para   : {lame_lambda_}")
print(f"Second Lame param : {lame_mu_}")

# Define function parameters
GISMO_OPTIONS = [
    {
        # F - function (source)
        "tag": "Function",
        "attributes": {
            "type": "FunctionExpr",
            "id": f"{1}",
            "dim": f"{dim}",
        },
        "children": [
            {
                "tag": "c",
                "attributes": {"index": "0"},
                "text": f"{source_function_[0]}",
            },
            {
                "tag": "c",
                "attributes": {"index": "1"},
                "text": f"{source_function_[1]}",
            },
        ],
    },
    {
        # Boundary Conditions first patch (arc)
        "tag": "boundaryConditions",
        "attributes": {"multipatch": "0", "id": "2"},
        "children": [
            {
                # Dirichlet for bottom surface (y=0)
                "tag": "Function",
                "attributes": {
                    "type": "FunctionExpr",
                    "dim": f"{dim}",
                    "index": "0",
                },
                "text": "0",
            },
            {
                # Neumann boundary
                "tag": "Function",
                "attributes": {
                    "type": "FunctionExpr",
                    "dim": f"{dim}",
                    "index": "1",
                    "c": "2",
                },
                "children": [
                    {
                        "tag": "c",
                        "attributes": {"index": "0"},
                        "text": f"{neumann_force_[0]}",
                    },
                    {
                        "tag": "c",
                        "attributes": {"index": "1"},
                        "text": f"{neumann_force_[1]}",
                    },
                ],
            },
            {
                # symmetry boundary (x=0)
                "tag": "Function",
                "attributes": {
                    "type": "FunctionExpr",
                    "dim": f"{dim}",
                    "index": "2",
                },
                "text": "0",
            },
        ],
    },
]


# Bottom surface
GISMO_OPTIONS[1]["children"].append(
    {
        "tag": "bc",
        "attributes": {
            "type": "Dirichlet",
            "function": str(0),
            "unknown": str(0),
            "component": str(1),
            "name": f"BID{2}",
        },
    }
)

# Symmetry condition
GISMO_OPTIONS[1]["children"].append(
    {
        "tag": "bc",
        "attributes": {
            "type": "Dirichlet",
            "function": str(2),
            "unknown": str(0),
            "component": str(0),
            "name": f"BID{3}",
        },
    }
)

# Force condition
GISMO_OPTIONS[1]["children"].append(
    {
        "tag": "bc",
        "attributes": {
            "type": "Neumann",
            "function": str(1),
            "unknown": str(0),
            "name": f"BID{4}",
        },
    }
)

if not ACTIVATE_SOURCE_FUNCTION:
    GISMO_OPTIONS.pop(0)


class Optimizer:
    def __init__(
        self,
        macro_spline,
        microtile,
        para_spline,
        identifier_function_neumann,
        n_threads=12,
        tiling=[24, 12],
        scaling_factor_objective_function=100,
        n_refinements=1,
        write_logfiles=False,
        max_volume=1.5,
        objective_function_type=1,
        macro_ctps=None,
        parameter_default_value=0.1,
        parameter_scaling=1,
        volume_scaling=1,
        macro_spline_sym=None,
        microtile_sym=None,
    ):
        self.parameter_default_value = parameter_default_value
        self.n_refinements = n_refinements
        self.microtile = microtile
        self.interfaces = None
        self.macro_spline = macro_spline.bspline
        self.macro_spline_original = self.macro_spline.copy()
        self.para_spline = para_spline.bspline
        self.identifier_function_neumann = identifier_function_neumann
        self.tiling = tiling
        self.scaling_factor_objective_function = (
            scaling_factor_objective_function
        )
        self.linear_solver = pygadjoints.LinearElasticityProblem()
        self.linear_solver.set_number_of_threads(n_threads)
        self.linear_solver.set_objective_function(objective_function_type)
        self.linear_solver.set_material_constants(lame_lambda_, lame_mu_)
        self.last_parameters = None
        self.iteration = 0
        self.write_logfiles = write_logfiles
        self.max_volume = max_volume
        self.macro_ctps = macro_ctps
        self.parameter_scaling = parameter_scaling
        self.volume_scaling = volume_scaling
        self.macro_spline_sym = macro_spline_sym
        self.microtile_sym = microtile_sym

    def prepare_microstructure(self):
        def parametrization_function(x):
            """
            Parametrization Function (determines thickness)
            """
            return self.para_spline.evaluate(x)

        def parameter_sensitivity_function(x):
            basis_function_matrix = sp.utils.data.make_matrix(
                *self.para_spline.basis_and_support(x),
                self.para_spline.cps.shape[0],
                as_array=True,
            ).reshape(x.shape[0], 1, self.para_spline.cps.shape[0])

            basis_function_matrix = np.repeat(
                np.tile(basis_function_matrix, [1, 2, 1]), repeats=2, axis=2
            )
            basis_function_matrix[:, 0, 1::2] = 0
            basis_function_matrix[:, 1, 0::2] = 0
            return basis_function_matrix

        # Initialize microstructure generator and assign values
        generator = sp.microstructure.Microstructure()
        generator.deformation_function = self.macro_spline
        generator.tiling = self.tiling
        generator.microtile = self.microtile
        generator.parametrization_function = parametrization_function
        generator.parameter_sensitivity_function = (
            parameter_sensitivity_function
        )

        generator_sym = sp.microstructure.Microstructure()
        generator_sym.deformation_function = self.macro_spline_sym
        generator_sym.tiling = [3, self.tiling[1]]
        generator_sym.microtile = self.microtile_sym

        # Creator for identifier functions
        def identifier_function(deformation_function, boundary_spline):
            def identifier_function(x):
                distance_2_boundary = boundary_spline.proximities(
                    queries=x,
                    initial_guess_sample_resolutions=[21]
                    * boundary_spline.para_dim,
                    tolerance=1e-9,
                    return_verbose=True,
                )[3]
                return distance_2_boundary.flatten() < 1e-8

            return identifier_function

        multipatch_opt = generator.create(
            contact_length=0.5, macro_sensitivities=len(self.macro_ctps) > 0
        )
        multipatch_sym = generator_sym.create()

        multipatch = sp.Multipatch(
            multipatch_opt.patches + multipatch_sym.patches
        )

        for i_field, _ in enumerate(multipatch_opt.fields):
            multipatch.add_fields(
                [
                    multipatch_opt.fields[i_field].patches
                    + [None] * len(multipatch_sym.patches)
                ],
                field_dim=2,
            )

        # Reuse existing interfaces
        if self.interfaces is None:
            multipatch.determine_interfaces()

            # Define Boundaries
            # Bottom boundary -> Gets ID 2
            multipatch.boundary_from_function(
                identifier_function(
                    generator.deformation_function,
                    self.macro_spline.extract.boundaries([0])[0],
                )
            )
            # Symmetry boundary -> Gets ID 3
            multipatch.boundary_from_function(
                identifier_function(
                    generator_sym.deformation_function,
                    self.macro_spline_sym.extract.boundaries([1])[0],
                )
            )
            # Neumann boundary -> Gets ID 4
            multipatch.boundary_from_function(
                identifier_function(
                    generator_sym.deformation_function,
                    self.macro_spline_sym.extract.boundaries([3])[0],
                )
            )

            self.interfaces = multipatch.interfaces
        else:
            multipatch.interfaces = self.interfaces

        # boundaries = []
        # for i in range(1, 5):
        #     boundaries.append(multipatch.boundary_multipatch(i))
        #     boundaries[-1].show_options["c"] = i
        # sp.show(boundaries, control_points=False, knots=False)

        sp.io.gismo.export(
            self.get_filename(),
            multipatch=multipatch,
            options=GISMO_OPTIONS,
            export_fields=True,
            as_base64=True,
            field_mask=(
                np.arange(0, np.prod(self.para_spline.cps.shape)).tolist()
                + (
                    np.array(self.macro_ctps)
                    + np.prod(self.para_spline.cps.shape)
                ).tolist()
            ),
        )

    def ensure_parameters(self, parameters, increase_count=True):
        # Check if anything changed since last call
        if self.last_parameters is not None and np.allclose(
            self.last_parameters, parameters
        ):
            return

        # Apply Parameter Scaling
        inverse_scaling = 1 / self.parameter_scaling

        if increase_count:
            self.iteration += 1

        # Something differs (or first iteration)
        self.para_spline.cps[:] = (
            parameters[: np.prod(self.para_spline.cps.shape)].reshape(-1, 2)
            * inverse_scaling
        )
        self.macro_spline.cps.ravel()[self.macro_ctps] = (
            parameters[np.prod(self.para_spline.cps.shape) :]
            + self.macro_spline_original.cps.ravel()[self.macro_ctps]
        )
        self.prepare_microstructure()
        if self.last_parameters is None:
            # First iteration
            self.linear_solver.init(
                self.get_filename(), self.n_refinements, 0, True
            )
        else:
            self.linear_solver.update_geometry(
                self.get_filename(), topology_changes=False
            )

        self.linear_solver.read_control_point_sensitivities(
            self.get_filename() + ".fields.xml"
        )
        self.control_point_sensitivities = (
            self.linear_solver.get_control_point_sensitivities()
        )
        self.last_parameters = parameters.copy()

        # Notify iteration evaluator
        self.current_objective_function_value = None
        self.ctps_sensitivity = None

    def evaluate_iteration(self, parameters):
        self.ensure_parameters(parameters)
        if self.current_objective_function_value is not None:
            return self.current_objective_function_value

        # There is no current solution all checks have been performed
        self.linear_solver.assemble()
        self.linear_solver.solve_linear_system()
        self.current_objective_function_value = (
            self.linear_solver.objective_function()
            * self.scaling_factor_objective_function
        )

        #
        if self.iteration == 1:
            self.linear_solver.export_multipatch_object("multipatch_initial")
            self.linear_solver.export_paraview("initial", False, 3**2, True)

        # Write into logfile
        with open("log_file_iterations.csv", "a") as file1:
            file1.write(
                ", ".join(
                    str(a)
                    for a in (
                        [self.iteration]
                        + [self.current_objective_function_value]
                        + parameters.tolist()
                    )
                )
                + "\n"
            )

        return self.current_objective_function_value

    def evaluate_jacobian(self, parameters):
        # Make sure that current file is valid
        _ = self.evaluate_iteration(parameters)

        # Determine Lagrange multipliers
        self.linear_solver.solve_adjoint_system()
        ctps_sensitivities = (
            self.linear_solver.objective_function_deris_wrt_ctps()
        )
        parameter_sensitivities = (
            (ctps_sensitivities @ self.control_point_sensitivities)
            * self.scaling_factor_objective_function
            / self.parameter_scaling
        )

        # Write into logfile
        with open("log_file_sensitivities.csv", "a") as file1:
            file1.write(
                ", ".join(
                    str(a)
                    for a in (
                        [self.iteration]
                        + parameter_sensitivities.tolist()
                        + parameters.tolist()
                    )
                )
                + "\n"
            )
        return parameter_sensitivities

    def volume(self, parameters):
        self.ensure_parameters(parameters)
        volume = self.linear_solver.volume()

        # Write into logfile
        with open("log_file_volume.csv", "a") as file1:
            file1.write(
                ", ".join(
                    str(a)
                    for a in (
                        [self.iteration] + [volume] + parameters.tolist()
                    )
                )
                + "\n"
            )

        return (self.max_volume - volume) * self.volume_scaling

    def volume_deriv(self, parameters):
        self.ensure_parameters(parameters)
        volume_sensitivities_ctps = self.linear_solver.volume_deris_wrt_ctps()
        volume_sensitivities = -(
            volume_sensitivities_ctps
            @ self.control_point_sensitivities
            / self.parameter_scaling
        )
        assert not np.any(np.isnan(self.control_point_sensitivities))

        # Write into logfile
        with open("log_file_volume_sensitivities.csv", "a") as file1:
            file1.write(
                ", ".join(
                    str(a)
                    for a in (
                        [self.iteration]
                        + (-volume_sensitivities).tolist()
                        + parameters.tolist()
                    )
                )
                + "\n"
            )
        return volume_sensitivities * self.volume_scaling

    def constraint(self):
        return {"type": "ineq", "fun": self.volume, "jac": self.volume_deriv}

    def finalize(self, parameters):
        self.ensure_parameters(parameters, increase_count=False)
        self.linear_solver.assemble()
        self.linear_solver.solve_linear_system()
        self.linear_solver.export_multipatch_object("multipatch_optimized")
        self.linear_solver.export_paraview("optimized", False, 3**2, True)

    def optimize(self):
        # Initialize the optimization
        n_design_vars_para = np.prod(self.para_spline.cps.size)
        n_design_vars_macro = len(self.macro_ctps)
        initial_guess = np.empty(n_design_vars_macro + n_design_vars_para)
        initial_guess[:n_design_vars_para] = (
            np.ones(n_design_vars_para)
            * self.parameter_default_value
            * self.parameter_scaling
        )
        initial_guess[n_design_vars_para:] = 0

        optim = scipy.optimize.minimize(
            self.evaluate_iteration,
            initial_guess.ravel(),
            method="SLSQP",
            jac=self.evaluate_jacobian,
            bounds=(
                [
                    (
                        0.0111 * self.parameter_scaling,
                        0.207106 * self.parameter_scaling,
                    )
                    for _ in range(n_design_vars_para)
                ]
                + [(-50, 50) for _ in range(n_design_vars_macro)]
            ),
            constraints=self.constraint(),
            options={"disp": True},
            tol=1e-4,
        )
        # Finalize
        self.finalize(optim.x)
        print("Best Parameters : ")
        print(optim.x)
        print(optim)

    def get_filename(self):
        return (
            "lattice_structure_"
            + str(self.tiling[0])
            + "x"
            + str(self.tiling[1])
            + ".xml"
        )


def main():
    # Set the number of available threads (will be passed to splinepy and
    # pygdjoints)

    # Optimization parameters
    # For volume density 0.5
    objective_function = 1
    macro_ctps = [2, 3, 8, 9]
    scaling_factor_objective_function = 1 / 46898.43832478186
    scaling_factor_parameters = 5
    scaling_factor_volume = 1 / 1166.6666653306665
    n_refinemenets = 0
    volume_density = 0.55

    # Geometry definition
    tiling = [10, 5]
    parameter_spline_degrees = [0, 0]
    parameter_spline_cps_dimensions = tiling
    # For volume density 0.5
    parameter_default_value = 0.27248556103895805 / scaling_factor_parameters

    sp.settings.NTHREADS = 1
    write_logfiles = True

    # Create parameters spline
    parameter_spline = sp.BSpline(
        degrees=parameter_spline_degrees,
        knot_vectors=[
            (
                [0] * parameter_spline_degrees[i]
                + np.linspace(
                    0,
                    1,
                    parameter_spline_cps_dimensions[i]
                    - parameter_spline_degrees[i]
                    + 1,
                ).tolist()
                + [1] * parameter_spline_degrees[i]
            )
            for i in range(len(parameter_spline_degrees))
        ],
        control_points=np.ones((np.prod(parameter_spline_cps_dimensions), 2))
        * parameter_default_value,
    )

    # Function for neumann boundary
    def identifier_function_neumann(x):
        pass

    # design domain macro spline
    macro_spline = sp.Bezier(
        degrees=[2, 1],
        control_points=[
            [20.0, 0.0],
            [20.0, 60.0],
            [80.0, 60.0],
            [0.0, 0.0],
            [0.0, 80.0],
            [80.0, 80.0],
        ],
    )

    # fixed domain macro spline
    macro_spline_sym = sp.Bezier(
        degrees=[1, 1],
        control_points=[
            [80.0, 60.0],
            [85.0, 60.0],
            [80.0, 80.0],
            [85.0, 80.0],
        ],
    )

    # generate microtile for fixed domain for conformity to design domain
    microtile_spline_list = []
    microtile_spline_list.append(
        sp.Bezier(
            degrees=[1, 1],
            control_points=[
                [0, 0],
                [1, 0],
                [0, 0.25],
                [1, 0.25],
            ],
        )
    )
    microtile_spline_list.append(
        sp.Bezier(
            degrees=[1, 1],
            control_points=[
                [0, 0.25],
                [1, 0.25],
                [0, 0.75],
                [1, 0.75],
            ],
        )
    )
    microtile_spline_list.append(
        sp.Bezier(
            degrees=[1, 1],
            control_points=[
                [0, 0.75],
                [1, 0.75],
                [0, 1],
                [1, 1],
            ],
        )
    )

    dense_volume = macro_spline.integrate.volume()
    max_volume = dense_volume * volume_density
    print(f"Max Volume is:{max_volume} out of {dense_volume}")

    optimizer = Optimizer(
        microtile=sp.microstructure.tiles.DoubleLattice(),
        macro_spline=macro_spline,
        para_spline=parameter_spline,
        identifier_function_neumann=identifier_function_neumann,
        tiling=tiling,
        scaling_factor_objective_function=scaling_factor_objective_function,
        n_refinements=n_refinemenets,
        n_threads=1,
        write_logfiles=write_logfiles,
        max_volume=max_volume,
        objective_function_type=objective_function,
        macro_ctps=macro_ctps,
        parameter_default_value=parameter_default_value,
        parameter_scaling=scaling_factor_parameters,
        volume_scaling=scaling_factor_volume,
        macro_spline_sym=macro_spline_sym,
        microtile_sym=microtile_spline_list,
    )

    # Try some parameters
    optimizer.optimize()

    exit()


if "__main__" == __name__:
    main()
