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


import pygadjoints

EPS = 1e-8
FILENAME = "microstructure_wonky.xml"
N_THREADS = 1

# Material parameters
DENSITY = 736
VISCOSITY = 6000  # Approximated by looking at https://www.ptonline.com/blog/post/understanding-the-effect-of-polymer-viscosity-on-melt-temperature
HEAT_CAPACITY = 2900
THERMAL_DIFFUSIVITY = 1.1997e-7

# Simulation refinements (0,0 and 0,2 work best)
N_REFINEMENTS = 0
DEGREE_ELEVATIONS = 0


if __name__ == "__main__":
    # -------------------------- STOKES SIMULATION -----------------------------
    stokes = pygadjoints.StokesTemperatureProblem()
    stokes.set_number_of_threads(nthreads=N_THREADS)
    stokes.set_material_constants(
        viscosity=VISCOSITY,
        density=DENSITY,
        heat_capacity=HEAT_CAPACITY,
        thermal_diffusivity=THERMAL_DIFFUSIVITY,
    )
    stokes.init(
        fname=FILENAME,
        refinements=N_REFINEMENTS,
        degree_elevations=DEGREE_ELEVATIONS,
        print_summary=False,
    )

    stokes.add_objective_function(1)
    stokes.add_objective_function(2)

    # Forward simulation
    stokes.assemble_fluid_problem()
    stokes.solve_fluid_linear_system()
    stokes.assemble_heat_problem()
    stokes.solve_heat_linear_system()

    obj_values = stokes.compute_objective_function_values()
    print("--------")
    print(obj_values)

    # Write to ParaView file
    stokes.export_paraview(
        filename="ParaviewOutput/gsif_static_mixer_solution",
        sample_rate=int(16**2),
    )
