import numpy as np
import splinepy as sp

import pygadjoints

EPS = 1e-8
BOX_LENGTH = 3
BOX_HEIGHT = 1
SHOW_MICROSTRUCTURE = False
FILENAME = "microstructure.xml"
N_THREADS = 10

# Material parameters
DENSITY = 1
VISCOSITY = 1

# Simulation parameters
N_REFINEMENTS = 3
DEGREE_ELEVATIONS = 1


if __name__ == "__main__":
    # ------------------------ GEOMETRY CONSTRUCTION ---------------------------
    # Geometry definition
    tiling = [3, 3]
    sp.settings.NTHREADS = N_THREADS
    microtile = sp.microstructure.tiles.HollowOctagon()
    contact_length = 0.4
    parameter_spline_degrees = [1, 1]
    parameter_spline_cps_dimensions = [3, 2]
    parameter_default_value = 0.3

    macro_spline = sp.helpme.create.box(BOX_LENGTH, BOX_HEIGHT)

    def identifier_inlet(points):
        return points[:, 0] < EPS

    def identifier_outlet(points):
        return points[:, 0] > BOX_LENGTH - EPS

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
        control_points=np.ones((np.prod(parameter_spline_cps_dimensions), 1))
        * parameter_default_value,
    )

    def parametrization_function(points):
        return np.tile(parameter_spline.evaluate(points), [1, 1])

    generator = sp.microstructure.Microstructure(
        deformation_function=macro_spline,
        tiling=tiling,
        microtile=microtile,
        parametrization_function=parametrization_function,
    )

    microstructure = generator.create(
        closing_face="x", contact_length=contact_length
    )

    # Determine interfaces and boundaries
    microstructure.determine_interfaces()
    microstructure.boundary_from_function(identifier_inlet, boundary_id=2)
    microstructure.boundary_from_function(identifier_outlet, boundary_id=3)

    if SHOW_MICROSTRUCTURE:
        n_bds = len(microstructure.boundaries)
        sp.show(
            *[
                [f"Boundary {i}", microstructure.boundary_multipatch(i)]
                for i in range(1, n_bds + 1)
            ],
            use_saved=True,
            control_points=False,
        )

    # ------------------------ GEOMETRY FILE EXPORT ----------------------------
    # Prepare for xml-file export
    additional_blocks = sp.io.gismo.AdditionalBlocks()

    # Velocity boundary conditions
    additional_blocks.add_boundary_conditions(
        block_id=2,
        dim=2,
        function_list=["0", (f"y * ({BOX_HEIGHT}-y)", "0")],
        bc_list=[
            ("BID2", "Dirichlet", 1),  # Inlet
            ("BID1", "Dirichlet", 0),  # Walls
        ],
        unknown_id=1,
        multipatch_id=0,
        comment=" Velocity boundary conditions: parabolic inflow field ",
    )

    # Pressure boundary conditions
    additional_blocks.add_boundary_conditions(
        block_id=3,
        dim=2,
        function_list=["0"],
        bc_list=[("BID3", "Dirichlet", 0)],
        unknown_id=0,
        multipatch_id=0,
        comment=" Pressure boundary conditions: fix outlet pressure to zero ",
    )

    # Get default assembly options
    additional_blocks.add_assembly_options(
        block_id=4, comment=" Assembler options "
    )

    n_tile_patches = 8
    southeast_patch = (tiling[1] - 1) * n_tile_patches + 0
    patch_corner = 2  # southwest: 1, southeast: 2, northwest: 3, northeast: 4

    # Dirty fix: change pressure bcs to one corner value
    gismo_export_options = additional_blocks.to_list()
    gismo_export_options[1]["children"] = [
        {
            "tag": "cv",
            "attributes": {
                "unknown": "0",
                "patch": str(southeast_patch),
                "corner": str(patch_corner),
            },
            "text": "0.0",
        }
    ]

    # Export to xml-file
    sp.io.gismo.export(
        fname=FILENAME,
        multipatch=microstructure,
        indent=True,
        additional_blocks=gismo_export_options,
    )

    # -------------------------- STOKES SIMULATION -----------------------------
    stokes = pygadjoints.StokesProblem()
    stokes.set_number_of_threads(nthreads=N_THREADS)
    stokes.set_material_constants(viscosity=VISCOSITY, density=DENSITY)
    stokes.init(
        fname=FILENAME,
        refinements=N_REFINEMENTS,
        degree_elevations=DEGREE_ELEVATIONS,
        print_summary=True,
    )

    # Forward simulation
    stokes.assemble()
    stokes.solve_linear_system()

    # Write to ParaView file
    stokes.export_paraview(
        fname="microstructure_solution",
        plot_elements=False,
        sample_rate=32**2,
        binary=True,
    )
