#include "pygadjoints/py_stokes_temperature.hpp"

#include <pybind11/pybind11.h>

namespace py = pybind11;

using stokes = pygadjoints::StokesTemperatureProblem;
using arg = py::arg;

void add_stokes_temperature_problem(py::module_ &m) {
  py::class_<stokes> klasse(m, "StokesTemperatureProblem");

  klasse.def(py::init<>())
      .def("init", &stokes::Init, arg("fname"), arg("refinements"),
           arg("degree_elevations"), arg("use_direct_solver") = true,
           arg("use_equal_order_bases") = false, arg("print_summary") = false)
      .def("set_material_constants", &stokes::SetMaterialConstants,
           arg("viscosity"), arg("density"), arg("heat_capacity"),
           arg("thermal_diffusivity"))
      .def("export_paraview", &stokes::ExportParaview, arg("filename"),
                              arg("sample_rate"))
  //     .def("export_xml", &stokes::ExportXML, arg("fname"))
      .def("assemble", &stokes::Assemble)
      .def("solve_linear_system", &stokes::SolveLinearSystem)
  //     .def("update_geometry", &stokes::UpdateGeometry, arg("fname"),
  //          arg("topology_changes"))
  //     .def("add_objective_function", &stokes::AddObjectiveFunction,
  //          arg("objective_function"))
  //     .def("compute_objective_function_values",
  //          &stokes::ComputeObjectiveFunctionValues)
  //     .def("compute_outflow", &stokes::ComputeOutflow)
  //     .def("compute_vel_divergence", &stokes::ComputeVelDivergence)
  //     .def("h_refine", &stokes::HRefine)

// OpenMP specifics
#ifdef PYGADJOINTS_USE_OPENMP
      .def("set_number_of_threads", &stokes::SetNumberOfThreads,
           arg("nthreads"))
#endif
      ;
}

// StokesProblem functions
namespace pygadjoints {

void StokesTemperatureProblem::ReadInputFromFile(const std::string &filename) {
  const Timer timer("ReadInputFromFile");
  // IDs in the xml input file (might change later)
  const index_t mpId{0}, fluidBcId{1},
                assembly_options_id{10},
                velocityAnalyticalId{12}, pressureAnalyticalId{13},
                temperatureBcId{66}, sourceFunctionId{100};

  // Import mesh and relevant information
  gsFileData<> fd(filename);
  fd.getId(mpId, mpPde);

  dimensionality_ = mpPde.geoDim();

  // Read boundary conditions for fluid velocity (id 0) and pressure (id 1)
  fd.getId(fluidBcId, bcInfo);

  // Set source term to zero if not given, otherwise read from file
  if (fd.hasId(sourceFunctionId)) {
    gsFunctionExpr<> sourceTerm("0", dimensionality_);
    fSource = sourceTerm;
  } else {
    fd.getId(sourceFunctionId, fSource);
  }

  // Check if file has analytical solution for velocity and pressure
  // respectively
  if (fd.hasId(velocityAnalyticalId)) {
    fd.getId(velocityAnalyticalId, velocityAnalyticalSolution);
    hasPressureSolution = true;
  }
  if (fd.hasId(pressureAnalyticalId)) {
    fd.getId(pressureAnalyticalId, pressureAnalyticalSolution);
    hasVelocitySolution = true;
  }

  // Read heat problem related info
  
}

void StokesTemperatureProblem::Init(const std::string &filename,
                                    const int numberOfRefinements,
                                    const int numberDegreeElevations,
                                    const bool useDirectSolver,
                                    const bool useEqualOrderBases,
                                    const bool printSummary) {
  const Timer timer("Initialization");

  // Process information from input file
  StokesTemperatureProblem::ReadInputFromFile(filename);

  // Set up discretization bases
  gsMultiBasis<> basis(mpPde);
  // Elevate degree
  basis.setDegree(basis.maxCwiseDegree() + numberDegreeElevations);
  // h-refinement
  for (int r = 0; r < numberOfRefinements; ++r) {
    basis.uniformRefine();
  }
  // Create bases for velocity and pressure
  std::vector<gsMultiBasis<>> discreteBases{basis, basis};
  // Elevate degree of velocity if not using equal order bases -> Taylor-Hood elements
  if (!useEqualOrderBases) {
    discreteBases[0].degreeElevate(1);
  }

  // Initialize Navier-Stokes PDE object
  NSPde = std::make_shared<gsNavStokesPde<real_t>>(mpPde, bcInfo, &fSource, viscosity_);
  flowParams = std::make_shared<gsFlowSolverParams<real_t>>(*NSPde, discreteBases);
  flowParams->options().setSwitch("quiet", printSummary);
  // TODO: for now element by element assembly. Maybe in future make user decide
  flowParams->options().setString("assemb.loop", "EbE");

  solveOpt.addInt("geo", "", 0);
  
  solveOpt.addInt("plotPts", "", 10000);
  // solveOpt.addInt("animStep", "", animStep);
  solveOpt.addReal("tol", "", 1e-5);
  solveOpt.addSwitch("plot", "", true);
  solveOpt.addSwitch("plotMesh", "", false);
  solveOpt.addString("id", "", "");


  // Steady without any iterations
  if (useDirectSolver) {
    solveOpt.setString("id", "steady");
    flowParams->options().setString("lin.solver", "direct");
  } else {
    solveOpt.setString("id", "steadyIt");
    flowParams->options().setString("lin.solver", "iter");
    flowParams->options().setString("lin.solver", "iter");
    flowParams->options().setInt("lin.maxIt", 50);
    flowParams->options().setReal("lin.tol", 1e-6);
    flowParams->options().setString("lin.precType", "MSIMPLER_FdiagEqual");
  }
  
  // Initialize fluid solver
  pNSSolver = std::make_shared<gsINSSolverSteady<real_t, ColMajor>>(flowParams);

  // Prepare heat problem

}

void StokesTemperatureProblem::Assemble() {
  const Timer timer("Assemble");
  pNSSolver->initialize();
}

void StokesTemperatureProblem::SolveLinearSystem() {
  const Timer timer("SolveLinearSystem");
  pNSSolver->solveStokes();
}

void StokesTemperatureProblem::ExportParaview(const std::string& fname, const int &sampleRate) {
  const Timer timer("ExportParaview");
  
  gsField<> velocityField = pNSSolver->constructSolution(0);
  gsField<> pressureField = pNSSolver->constructSolution(1);

  gsWriteParaview<>(velocityField, fname+"_velocity", sampleRate);
  gsWriteParaview<>(pressureField, fname+"_pressure", sampleRate);
}

}// namespace pygadjoints
