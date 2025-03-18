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
      .def("assemble_fluid_problem", &stokes::AssembleFluidProblem)
      .def("solve_fluid_linear_system", &stokes::SolveFluidLinearSystem)
      .def("assemble_heat_problem", &stokes::AssembleHeatProblem)
      .def("solve_heat_linear_system", &stokes::SolveHeatLinearSystem)
      .def("update_geometry", &stokes::UpdateGeometry, arg("fname"),
           arg("topology_changes"))
      .def("add_objective_function", &stokes::AddObjectiveFunction,
           arg("objective_function"))
      .def("compute_objective_function_values",
           &stokes::ComputeObjectiveFunctionValues)
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
  const index_t mpId{0}, fluidBcId{1}, assemblyOptionsId{10},
      velocityAnalyticalId{12}, pressureAnalyticalId{13}, temperatureBcId{66},
      sourceFunctionId{100};

  // Import mesh and relevant information
  gsFileData<> fd(filename);
  fd.getId(mpId, mpPde);

  dimensionality_ = mpPde.geoDim();

  // Read assembly options
  fd.getId(assemblyOptionsId, assemblyOptions);

  // Read boundary conditions for fluid velocity (id 0) and pressure (id 1)
  fd.getId(fluidBcId, bcInfo);
  fd.getId(temperatureBcId, temperatureBcInfo);

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
  functionBasisTemperature = gsMultiBasis<>(mpPde);
  // Elevate degree
  basis.setDegree(basis.maxCwiseDegree() + numberDegreeElevations);
  functionBasisTemperature.setDegree(basis.maxCwiseDegree() +
                                     numberDegreeElevations);
  // Create bases for velocity and pressure
  std::vector<gsMultiBasis<>> discreteBases{basis, basis};
  // Elevate degree of velocity if not using equal order bases -> Taylor-Hood
  // elements
  if (!useEqualOrderBases) {
    discreteBases[0].degreeElevate(1);
    functionBasisTemperature.degreeElevate(1);
  }
  // h-refinement
  for (int r = 0; r < numberOfRefinements; ++r) {
    discreteBases[0].uniformRefine();
    discreteBases[1].uniformRefine();
    functionBasisTemperature.uniformRefine();
  }

  // Initialize Navier-Stokes PDE object
  pNSPde = std::make_shared<gsNavStokesPde<real_t>>(mpPde, bcInfo, &fSource,
                                                    viscosity_);
  pFlowParams =
      std::make_shared<gsFlowSolverParams<real_t>>(*pNSPde, discreteBases);
  pFlowParams->options().setSwitch("quiet", printSummary);
  // TODO: for now element by element assembly. Maybe in future make user decide
  pFlowParams->options().setString("assemb.loop", "EbE");

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
    pFlowParams->options().setString("lin.solver", "direct");
  } else {
    solveOpt.setString("id", "steadyIt");
    pFlowParams->options().setString("lin.solver", "iter");
    pFlowParams->options().setString("lin.solver", "iter");
    pFlowParams->options().setInt("lin.maxIt", 50);
    pFlowParams->options().setReal("lin.tol", 1e-6);
    pFlowParams->options().setString("lin.precType", "MSIMPLER_FdiagEqual");
  }

  // Initialize fluid solver
  pNSSolver =
      std::make_shared<gsINSSolverSteady<real_t, ColMajor>>(pFlowParams);

  // Prepare heat problem
  // Define diffusion term
  std::vector<std::string> diffusionTermStrings;
  for (int i = 0; i < dimensionality_; ++i) {
    for (int j = 0; j < dimensionality_; ++j) {
      // Fill diagonal with thermal diffusivity value
      if (i == j) {
        diffusionTermStrings.push_back(std::to_string(thermalDiffusivity_));
      } else {
        diffusionTermStrings.push_back("0.0");
      }
    }
  }
  gsFunctionExpr<> diffusionTerm(diffusionTermStrings, dimensionality_);
  coeffDiffusion = diffusionTerm;
  // Reaction term
  gsFunctionExpr<> reactionTerm("0.0", dimensionality_);
  coeffReaction = reactionTerm;
  // Rhs term, TODO: heat generation due to viscous dissipation
  gsFunctionExpr<> rhsTerm("0.0", dimensionality_);
  cdrRhs = rhsTerm;
}

void StokesTemperatureProblem::AssembleFluidProblem() {
  const Timer timer("AssembleFluidProblem");
  pNSSolver->initialize();
}

void StokesTemperatureProblem::SolveFluidLinearSystem() {
  const Timer timer("SolveFluidLinearSystem");
  pNSSolver->solveStokes();
}

void StokesTemperatureProblem::AssembleHeatProblem() {
  const Timer timer("AssembleHeatProblem");

  // Get velocity field, TODO: make this a variable and update it
  gsField<> velocityField = pNSSolver->constructSolution(0);
  const gsFunctionSet<> &velocityFieldSet = velocityField.fields();

  pHeatPde = std::make_shared<gsConvDiffRePde<real_t>>(
      mpPde, temperatureBcInfo, &coeffDiffusion, &velocityFieldSet,
      &coeffReaction, &cdrRhs);

  // Define assembler, TODO: define before and just assemble here
  pHeatAssembler = std::make_shared<gsCDRAssembler<real_t>>(
      *pHeatPde, functionBasisTemperature);
  pHeatAssembler->options().setInt("Stabilization", stabilizerCDR::SUPG);
  pHeatAssembler->options().setInt("DirichletValues", dirichlet::l2Projection);

  pHeatAssembler->assemble();
}

void StokesTemperatureProblem::SolveHeatLinearSystem() {
  const Timer timer("SolveHeatLinearSystem");

  heatSolver.compute(pHeatAssembler->matrix());
  heatSolutionVector = heatSolver.solve(pHeatAssembler->rhs());
}

void StokesTemperatureProblem::ExportParaview(const std::string &fname,
                                              const int &sampleRate) {
  const Timer timer("ExportParaview");

  gsField<> velocityField = pNSSolver->constructSolution(0);
  gsField<> pressureField = pNSSolver->constructSolution(1);

  gsWriteParaview<>(velocityField, fname + "_velocity", sampleRate);
  gsWriteParaview<>(pressureField, fname + "_pressure", sampleRate);

  // Heat problem
  gsField<> temperatureField =
      pHeatAssembler->constructSolution(heatSolutionVector);
  gsWriteParaview<>(temperatureField, fname + "_temperature", sampleRate);
}

void StokesTemperatureProblem::AddObjectiveFunction(
    const int objective_function_selector) {
  objective_functions_selected.push_back(objective_function_selector);
}

std::vector<real_t> StokesTemperatureProblem::ComputeObjectiveFunctionValues() {
  const Timer timer("ComputeObjectiveFunction");

  // Define in- and oulet boundary IDs
  const std::string inletID = "BID2";
  const std::string outletID = "BID3";

  real_t objective_value;
  std::vector<real_t> objective_function_values;

  gsMapData<> mdBoundary(NEED_MEASURE), mdPatch(NEED_MEASURE);

  // Prepare quadrature
  const gsINSAssembler<real_t, ColMajor> *fluidAssembler =
      pNSSolver->getAssembler();

  gsField<> velocityField = pNSSolver->constructSolution(0);
  gsField<> pressureField = pNSSolver->constructSolution(1);
  gsField<> temperatureField =
      pHeatAssembler->constructSolution(heatSolutionVector);

  for (auto &objective_function_index : objective_functions_selected) {
    objective_value = 0.0;

    // Objective 1: pressure loss
    if (objective_function_index == 1) {
      real_t inletPressure{0.0}, outletPressure(0.0);
      // Go through every boundary
      for (gsMultiPatch<>::const_biterator bit = mpPde.bBegin();
           bit != mpPde.bEnd(); ++bit) {
        // Compute inlet pressure
        if (!(bit->label() == inletID or bit->label() == outletID)) {
          continue;
        }
        const gsGeometry<> &patch = mpPde[bit->patch];
        // Get basis for pressure (patch and boundary side)
        gsBasis<> &pressureBasis = pFlowParams->getBases()[1].basis(bit->patch);
        typename gsBasis<>::uPtr pressureBoundaryBasis =
            pressureBasis.boundaryBasis(bit->side());
        // Get quadrature rules
        QuRuleBoundary =
            gsQuadrature::getPtr(*pressureBoundaryBasis, assemblyOptions);
        QuRulePatch = gsQuadrature::getPtr(pressureBasis, assemblyOptions);
        // Iterators over sides of boundary element patches
        typename gsBasis<>::domainIter boundaryElementIt =
            pressureBoundaryBasis->domain()->beginAll();
        typename gsBasis<>::domainIter boundaryElementItEnd =
            pressureBoundaryBasis->domain()->endAll();
        // Iterator over patches of boundary elements
        typename gsBasis<>::domainIter boundaryPatchIt =
            pressureBasis.domain()->beginBdr(bit->side());
        // Get boundary side basis
        typename gsGeometry<>::uPtr pBoundary = patch.boundary(bit->side());
        for (; boundaryElementIt < boundaryElementItEnd; ++boundaryElementIt) {
          // Map quadrature to corresponding patch (side)
          QuRuleBoundary->mapTo(boundaryElementIt.lowerCorner(),
                                boundaryElementIt.upperCorner(),
                                mdBoundary.points, quWeightsBoundary);
          QuRulePatch->mapTo(boundaryPatchIt.lowerCorner(),
                             boundaryPatchIt.upperCorner(), mdPatch.points,
                             quWeightsPatch);
          // Compute mapping for boundary;s side
          pBoundary->computeMap(mdBoundary);
          // Get values at boundary's patch
          basisValues = pressureField.value(mdPatch.points, bit->patch);
          // Perform numerical integration
          for (index_t k = 0; k != mdBoundary.points.cols(); ++k) {
            if (bit->label() == inletID) {
              inletPressure +=
                  quWeightsBoundary[k] * mdBoundary.measure(k) * basisValues(k);
            } else if (bit->label() == outletID) {
              outletPressure +=
                  quWeightsBoundary[k] * mdBoundary.measure(k) * basisValues(k);
            }
          }
          // Update boundary patch iterator
          ++boundaryPatchIt;
        }
        objective_value = inletPressure - outletPressure;
      }
      // Length computation
    } else if (objective_function_index == 2) {
      std::vector<real_t> quadratureWeights, basisValuesList;
      for (gsMultiPatch<>::const_biterator bit = mpPde.bBegin();
           bit != mpPde.bEnd(); ++bit) {
        // Compute inlet volume
        if (bit->label() != outletID) {
          continue;
        }
        const gsGeometry<> &patch = mpPde[bit->patch];
        // Get basis for pressure (patch and boundary side)
        const gsBasis<> &temperatureBasis =
            functionBasisTemperature.basis(bit->patch);
        typename gsBasis<>::uPtr temperatureBoundaryBasis =
            temperatureBasis.boundaryBasis(bit->side());
        // Get quadrature rules
        QuRuleBoundary =
            gsQuadrature::getPtr(*temperatureBoundaryBasis, assemblyOptions);
        QuRulePatch = gsQuadrature::getPtr(temperatureBasis, assemblyOptions);
        // Iterators over sides of boundary element patches
        typename gsBasis<>::domainIter boundaryElementIt =
            temperatureBoundaryBasis->domain()->beginAll();
        typename gsBasis<>::domainIter boundaryElementItEnd =
            temperatureBoundaryBasis->domain()->endAll();
        // Iterator over patches of boundary elements
        typename gsBasis<>::domainIter boundaryPatchIt =
            temperatureBasis.domain()->beginBdr(bit->side());
        // Get boundary side basis
        typename gsGeometry<>::uPtr pBoundary = patch.boundary(bit->side());
        for (; boundaryElementIt < boundaryElementItEnd; ++boundaryElementIt) {
          // Map quadrature to corresponding patch (side)
          QuRuleBoundary->mapTo(boundaryElementIt.lowerCorner(),
                                boundaryElementIt.upperCorner(),
                                mdBoundary.points, quWeightsBoundary);
          QuRulePatch->mapTo(boundaryPatchIt.lowerCorner(),
                             boundaryPatchIt.upperCorner(), mdPatch.points,
                             quWeightsPatch);
          // Compute mapping for boundary;s side
          pBoundary->computeMap(mdBoundary);
          // Get values at boundary's patch
          basisValues = temperatureField.value(mdPatch.points, bit->patch);
          // Collect all quadrature information
          for (index_t k = 0; k != mdBoundary.points.cols(); ++k) {
            quadratureWeights.push_back(quWeightsBoundary[k] *
                                        mdBoundary.measure(k));
            basisValuesList.push_back(basisValues(k));
          }
          // Update boundary patch iterator
          ++boundaryPatchIt;
        }
        // Compute average temperature
        real_t temperatureIntegral{0.0}, boundaryLength{0.0};
        index_t nEntries = quadratureWeights.size();
        for (index_t i = 0; i < nEntries; ++i) {
          boundaryLength += quadratureWeights[i];
          temperatureIntegral += quadratureWeights[i] * basisValuesList[i];
        }
        real_t temperatureAverage = temperatureIntegral / boundaryLength;
        // Compute L2-deviation to average temperature
        real_t temperatureDifference;
        for (index_t i = 0; i < nEntries; ++i) {
          temperatureDifference = basisValuesList[i] - temperatureAverage;
          objective_value += temperatureDifference * temperatureDifference *
                             quadratureWeights[i];
        }
        objective_value = math::sqrt(objective_value);
      }
    } else {
      throw std::runtime_error("Objective function not known!\n");
    }
    objective_function_values.push_back(objective_value);
  }

  return objective_function_values;
}

void StokesTemperatureProblem::UpdateGeometry(const std::string &fname,
                                              const bool &topology_changes) {
  const Timer timer("UpdateGeometry");
  if (topology_changes) {
    throw std::runtime_error("Not Implemented!");
  }

  // Import mesh and load relevant information
  gsMultiPatch<> mpNew;

  gsFileData<> fd(fname);
  fd.getId(0, mpNew);

  // This update does not require refinement or elevation, in theory the mp is
  // not touched, only the solution field
  size_t n_patches_new, n_patches_old;
  n_patches_new = mpNew.nPatches();
  n_patches_old = mpPde.nPatches();

  // Ignore all other information!
  if (n_patches_new != n_patches_old) {
    throw std::runtime_error(
        "This does not work - I am fucked. Expected number of "
        "patches " +
        std::to_string(n_patches_old) + ", but got " +
        std::to_string(n_patches_new));
  }
  // Manually update coefficients as to not overwrite any precomputed
  // values
  size_t n_new_coefs, n_old_coefs;

  for (size_t patch_id{}; patch_id < n_patches_new; patch_id++) {
    n_new_coefs = mpNew.patch(patch_id).coefs().size();
    n_old_coefs = mpPde.patch(patch_id).coefs().size();
    if (n_new_coefs != n_old_coefs) {
      throw std::runtime_error(
          "This does not work - I am fucked. Expected number of "
          "coefficients " +
          std::to_string(n_old_coefs) + ", but got " +
          std::to_string(n_new_coefs));
    }
    for (size_t i_coef = 0; i_coef != n_old_coefs; i_coef++) {
      mpPde.patch(patch_id).coefs().at(i_coef) =
          mpNew.patch(patch_id).coefs().at(i_coef);
    }
  }
  // pGeometry_expression->copyCoefs(mpNew);
}

} // namespace pygadjoints
