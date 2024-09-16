#include "pygadjoints/py_stokes.hpp"

#include <pybind11/pybind11.h>

namespace py = pybind11;

using stokes = pygadjoints::StokesProblem;
using arg = py::arg;

void add_stokes_problem(py::module_ &m) {
  py::class_<stokes> klasse(m, "StokesProblem");

  klasse.def(py::init<>())
      .def("init", &stokes::Init, arg("fname"), arg("refinements"),
           arg("degree_elevations"), arg("print_summary") = false)
      .def("set_material_constants", &stokes::SetMaterialConstants,
           arg("viscosity"), arg("density"))
      .def("export_paraview", &stokes::ExportParaview, arg("fname"),
           arg("plot_elements"), arg("sample_rate"), arg("binary"))
      .def("assemble", &stokes::Assemble)
      .def("solve_linear_system", &stokes::SolveLinearSystem)

// OpenMP specifics
#ifdef PYGADJOINTS_USE_OPENMP
      .def("set_number_of_threads", &stokes::SetNumberOfThreads,
           arg("nthreads"))
#endif
      ;
}

// StokesProblem functions
namespace pygadjoints {

void StokesProblem::ReadInputFromFile(const std::string &filename) {
  const Timer timer("ReadInputFromFile");
  // IDs in the xml input file (might change later)
  const index_t mp_id{0}, vel_bc_id{2}, p_bc_id{3}, ass_opt_id{4},
      vel_ana_id{12}, p_ana_id{13};

  // Import mesh and load relevant information
  gsFileData<> fd(filename);
  fd.getId(mp_id, mp_pde);

  // Check if file has analytical solution for velocity and pressure
  // respectively
  if (fd.hasId(vel_ana_id)) {
    fd.getId(vel_ana_id, vel_analytical_solution);
    has_velocity_solution = true;
  }
  if (fd.hasId(p_ana_id)) {
    fd.getId(p_ana_id, p_analytical_solution);
    has_pressure_solution = true;
  }

  // Read boundary conditions for velocity
  fd.getId(vel_bc_id, vel_bcs);
  vel_bcs.setGeoMap(mp_pde);
  // Read pressure boundary conditions if given
  if (fd.hasId(p_bc_id)) {
    fd.getId(p_bc_id, p_bcs);
    has_pressure_bcs = true;
    p_bcs.setGeoMap(mp_pde);
  }

  // Read assembly options if given
  if (fd.hasId(ass_opt_id)) {
    gsOptionList Aopt;
    fd.getId(ass_opt_id, Aopt);

    // Set assembly options in the expression assembler
    expr_assembler_pde.setOptions(Aopt);
  }
}

void StokesProblem::Init(const std::string &filename,
                         const int number_of_refinements,
                         const int number_degree_elevations,
                         const bool print_summary) {
  const Timer timer("Initialization");
  const index_t pressure_id{0}, velocity_id{1};

  // Process information from input file
  StokesProblem::ReadInputFromFile(filename);

  // Set the dimension
  dimensionality_ = mp_pde.geoDim();

  // Set up function basis and refinements
  velocity_basis = gsMultiBasis<>(mp_pde, true);
  pressure_basis = gsMultiBasis<>(mp_pde, true);

  // p-refinement
  // Use Taylor-Hood elements: eleveate the degree of velocity for inf-sup
  // stability
  velocity_basis.setDegree(velocity_basis.maxCwiseDegree() +
                           number_degree_elevations + 1);
  pressure_basis.setDegree(pressure_basis.maxCwiseDegree() +
                           number_degree_elevations);

  // h-refinement
  for (index_t href{0}; href < number_of_refinements; href++) {
    velocity_basis.uniformRefine();
    pressure_basis.uniformRefine();
  }

  // Elements used for numerical integration
  expr_assembler_pde.setIntegrationElements(velocity_basis);
  // Get geometry mapping
  pGeometry_expression =
      std::make_shared<geometryMap>(expr_assembler_pde.getMap(mp_pde));

  // Set the discretization spaces
  pVelocity_space = std::make_shared<space>(expr_assembler_pde.getSpace(
      velocity_basis, dimensionality_, velocity_id));
  pPressure_space = std::make_shared<space>(
      expr_assembler_pde.getSpace(pressure_basis, 1, pressure_id));

  // Partial solution vectors
  pVelocity_solution = std::make_shared<solution>(
      expr_assembler_pde.getSolution(*pVelocity_space, solVector));
  pPressure_solution = std::make_shared<solution>(
      expr_assembler_pde.getSolution(*pPressure_space, solVector));

  // Set up Dirichlet boundary conditions, with C0-continuity
  pVelocity_space->setup(vel_bcs, dirichlet::l2Projection, 0);
  if (has_pressure_bcs) {
    pPressure_space->setup(p_bcs, dirichlet::l2Projection, 0);
  }

  // Assign DoF mapper to pressure basis (since velocity is elevated in degree)
  pDof_mapper = std::make_shared<gsDofMapper>(pPressure_space->mapper());

  // Expression evaluator
  pExpr_evaluator = std::make_shared<gsExprEvaluator<>>(
      gsExprEvaluator<>(expr_assembler_pde));

  // Initialize the system
  expr_assembler_pde.initSystem();

  // Print summary of function bases
  if (print_summary) {
    std::string padding{"        : "};
    std::cout << padding << "Simulation summary:\n"
              << padding << "--------------------\n"
              << padding << "Number of DOFs :\t\t"
              << expr_assembler_pde.numDofs() << "\n"
              << padding << "Number of patches :\t\t" << mp_pde.nPatches()
              << "\n";
    std::cout << padding << "Spline degrees (min->max):\n";
    std::cout << padding << "\tVelocity: \t\t"
              << velocity_basis.minCwiseDegree() << "->"
              << velocity_basis.maxCwiseDegree() << '\n';
    std::cout << padding << "\tPressure: \t\t"
              << pressure_basis.minCwiseDegree() << "->"
              << pressure_basis.maxCwiseDegree() << '\n';
    std::cout << std::endl;
  }
}

void StokesProblem::Assemble() { const Timer timer("Assemble"); }

void StokesProblem::ExportParaview(const std::string &fname,
                                   const bool &plot_elements,
                                   const int &sample_rate,
                                   const bool &export_b64) {
  Timer timer("Exporting Paraview");
}

} // namespace pygadjoints
