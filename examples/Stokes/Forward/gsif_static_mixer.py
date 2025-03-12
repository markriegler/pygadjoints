"""gsif_static_mixer.py

2D static mixer simulation using gsIncompressibleFlow

Some stats

Average inlet velocity: 0.02858657270155922 m/s
Diameter: 41 mm
Length: 140 mm
Processed temperature: 220 deg
Inlet throughput: 100 kg/h
Material: blown film grade of HDPE (Hostalen GD 9550F)
    Density: 736 kg/m^3
    Carreau parameters (A,B,C): 12.87, 0.1871, 0.655
    Carreau T_s: 237 K
"""

import numpy as np
import splinepy as sp

import pygadjoints

EPS = 1e-8
# BOX_LENGTH = 0.14
# BOX_HEIGHT = 0.041
# SHOW_MICROSTRUCTURE = False
FILENAME = "microstructure_wonky.xml"
N_THREADS = 1

# Material parameters
DENSITY = 736
VISCOSITY = 6000 # Approximated by looking at https://www.ptonline.com/blog/post/understanding-the-effect-of-polymer-viscosity-on-melt-temperature
HEAT_CAPACITY = 2900
THERMAL_DIFFUSIVITY = 1.1997e-7

# Simulation parameters
N_REFINEMENTS = 0
DEGREE_ELEVATIONS = 0


if __name__ == "__main__":
    # # ------------------------ GEOMETRY CONSTRUCTION ---------------------------
    # # Geometry definition
    # tiling = [6, 3]
    # sp.settings.NTHREADS = N_THREADS
    # microtile = sp.microstructure.tiles.SMX2DInverse()
    # parameter_spline_degrees = [1, 1]
    # parameter_spline_cps_dimensions = [3, 2]
    # parameter_default_value = 0.2

    # macro_spline = sp.helpme.create.box(BOX_LENGTH, BOX_HEIGHT)

    # def identifier_inlet(points):
    #     return points[:, 0] < EPS

    # def identifier_outlet(points):
    #     return points[:, 0] > BOX_LENGTH - EPS

    # # Create parameters spline
    # parameter_spline = sp.BSpline(
    #     degrees=parameter_spline_degrees,
    #     knot_vectors=[
    #         (
    #             [0] * parameter_spline_degrees[i]
    #             + np.linspace(
    #                 0,
    #                 1,
    #                 parameter_spline_cps_dimensions[i]
    #                 - parameter_spline_degrees[i]
    #                 + 1,
    #             ).tolist()
    #             + [1] * parameter_spline_degrees[i]
    #         )
    #         for i in range(len(parameter_spline_degrees))
    #     ],
    #     control_points=np.ones((np.prod(parameter_spline_cps_dimensions), 1))
    #     * parameter_default_value,
    # )

    # def parametrization_function(points):
    #     return np.tile(parameter_spline.evaluate(points), [1, 1])

    # generator = sp.microstructure.Microstructure(
    #     deformation_function=macro_spline,
    #     tiling=tiling,
    #     microtile=microtile,
    #     parametrization_function=parametrization_function,
    # )

    # microstructure = generator.create(
    #     closing_face="x"
    # )

    # # Determine interfaces and boundaries
    # microstructure.determine_interfaces()
    # microstructure.boundary_from_function(identifier_inlet, boundary_id=2)
    # microstructure.boundary_from_function(identifier_outlet, boundary_id=3)

    # if SHOW_MICROSTRUCTURE:
    #     n_bds = len(microstructure.boundaries)
    #     sp.show(
    #         *[
    #             [f"Boundary {i}", microstructure.boundary_multipatch(i)]
    #             for i in range(1, n_bds + 1)
    #         ],
    #         use_saved=True,
    #         control_points=False,
    #     )

    # # ------------------------ GEOMETRY FILE EXPORT ----------------------------
    # # Prepare for xml-file export
    # additional_blocks = sp.io.gismo.AdditionalBlocks()

    # # Velocity boundary conditions
    # additional_blocks.add_boundary_conditions(
    #     block_id=2,
    #     dim=2,
    #     function_list=[("0", "0"), (f"{INLET_PEAK_VELOCITY} * y * ({BOX_HEIGHT}-y)", "0")],
    #     bc_list=[
    #         ("BID2", "Dirichlet", 1),  # Inlet
    #         ("BID1", "Dirichlet", 0),  # Walls
    #     ],
    #     unknown_id=1,
    #     multipatch_id=0,
    #     comment=" Velocity boundary conditions: parabolic inflow field ",
    # )

    # # Pressure boundary conditions
    # additional_blocks.add_boundary_conditions(
    #     block_id=3,
    #     dim=2,
    #     function_list=["0"],
    #     bc_list=[("BID3", "Dirichlet", 0)],
    #     unknown_id=0,
    #     multipatch_id=0,
    #     comment=" Pressure boundary conditions: fix outlet pressure to zero ",
    # )

    # # Get default assembly options
    # additional_blocks.add_assembly_options(
    #     block_id=4, comment=" Assembler options "
    # )

    # # Export to xml-file
    # sp.io.gismo.export(
    #     fname=FILENAME,
    #     multipatch=microstructure,
    #     indent=True,
    #     additional_blocks=additional_blocks.to_list(),
    # )

    # -------------------------- STOKES SIMULATION -----------------------------
    stokes = pygadjoints.StokesTemperatureProblem()
    stokes.set_number_of_threads(nthreads=N_THREADS)
    stokes.set_material_constants(
        viscosity=VISCOSITY,
        density=DENSITY,
        heat_capacity=HEAT_CAPACITY,
        thermal_diffusivity=THERMAL_DIFFUSIVITY
    )
    stokes.init(
        fname=FILENAME,
        refinements=N_REFINEMENTS,
        degree_elevations=DEGREE_ELEVATIONS,
        print_summary=False,
    )
    
    # stokes.add_objective_function(2)
    # stokes.add_objective_function(3)

    # Forward simulation
    stokes.assemble()
    stokes.solve_linear_system()
    
    # obj_values = stokes.compute_objective_function_values()
    # print("--------")
    # print(obj_values)

    # Write to ParaView file
    stokes.export_paraview(
        filename="gsif_static_mixer_solution",
        sample_rate=int(64**2),
    )