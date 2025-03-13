#include <gismo.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <chrono>

#ifdef PYGADJOINTS_USE_OPENMP
#include <omp.h>
#endif

#include <gsIncompressibleFlow/src/gsINSSolver.h>
#include <gsIncompressibleFlow/src/gsFlowUtils.h>

#include "pygadjoints/timer.hpp"

namespace pygadjoints {

using namespace gismo;

namespace py = pybind11;

// enum class ObjectiveFunction : int {
//   // Maximize ∫ ε(v) : ε(v) dΩ = 0.5⋅∫ ∇v : (∇v+(∇v)ᵀ) dΩ
//   viscous_dissipation = 1,
//   // Early deflection: maximize ∫ (l-x)⋅(v_y)² dΩ if flow is going in
//   // x-direction
//   early_deflection = 2,
//   // pressure loss: minimize ∮ p dsᵢₙ - ∮ p dsₒᵤₜ
//   pressure_loss = 3
// };

class StokesTemperatureProblem {
  private:
  #ifdef PYGADJOINTS_USE_OPENMP
    int n_omp_threads{1};
  #endif
    // Fluid parameters
    real_t density_{1};             // Unit: kg/m³
    real_t viscosity_{1};           // Dynamic viscosity in Pa⋅s
    real_t heatCapacity_{1};        // Specific heat capacity in J/(kg⋅K)
    real_t thermalDiffusivity_{1};  // Unit: m²/s
  
    // Multipatch object
    gsMultiPatch<> mpPde;
  
    // Indicator if input file has analytical solution(s)
    bool hasVelocitySolution{false}, hasPressureSolution{false};
  
    // Analytical solutions to velocity and pressure
    gsFunctionExpr<> velocityAnalyticalSolution{}, pressureAnalyticalSolution{};
  
    // Boundary conditions
    gsBoundaryConditions<> bcInfo;
    gsBoundaryConditions<> temperatureBcInfo;
  
    // Source function (for Stokes assembler)
    gsFunctionExpr<> fSource;

    // Functions for the convection-diffusion equation
    gsFunctionExpr<> coeffDiffusion, coeffReaction, cdrRhs;
  
    // Number of refinements in the current iteration
    int n_refinements{};
  
    // Number of degree elevations
    int n_degree_elevations{};
  
    // Dimension of the physical space
    int dimensionality_{};
  
    // Equal order discretization bases
    bool equalOrderBases{false};
  
    // Navier-Stokes PDE object
    std::shared_ptr<gsNavStokesPde<real_t>> pNSPde{nullptr};
  
    // Parameters for the flow solver
    std::shared_ptr<gsFlowSolverParams<real_t>> pFlowParams{nullptr};
  
    // Solver option list
    gsOptionList solveOpt;
  
    // Fluid solver
    std::shared_ptr<gsINSSolverSteady<real_t, ColMajor>> pNSSolver{nullptr};

    // Heat problem PDE
    std::shared_ptr<gsConvDiffRePde<real_t>> pHeatPde{nullptr};
  
    // Function basis for temperature
    gsMultiBasis<> functionBasisTemperature;

    // Heat problem assembler
    std::shared_ptr<gsCDRAssembler<real_t>> pHeatAssembler{nullptr};

    // Heat problem linear system solver
    gsSparseSolver<>::BiCGSTABILUT heatSolver;

    // Heat problem solution vector
    gsMatrix<> heatSolutionVector;
  
    // // Discretization spaces
    // std::shared_ptr<space> pVelocity_space{nullptr}, pPressure_space{nullptr};
  
    // // Partial solution values/expression for the variables
    // std::shared_ptr<solution> pVelocity_solution{nullptr},
    //     pPressure_solution{nullptr};
  
    // std::vector<int> objective_functions_selected{};
public:
  /// @brief Constructor
  StokesTemperatureProblem() {
#ifdef PYGADJOINTS_USE_OPENMP
    omp_set_num_threads(std::min(omp_get_max_threads(), n_omp_threads));
#endif
  };

  gsStopwatch timer;

#ifdef PYGADJOINTS_USE_OPENMP
  /**
   * @brief Set the Number Of Threads for OpenMP
   *
   * Somehow does not compile
   * @param n_threads
   */
  void SetNumberOfThreads(const int &n_threads) {
    n_omp_threads = n_threads;
    omp_set_num_threads(n_threads);
}
#endif

  /// @brief Set up material constants
  /// @param viscosity Dynamic viscosity in Pa⋅s
  /// @param density Density in kg/m³
  /// @param heatCapacity Specific heat capacity in J/(kg⋅K)
  /// @param thermalDiffusivity Thermal diffusivity in m²/s
  void SetMaterialConstants(const real_t &viscosity, const real_t &density,
                            const real_t &heatCapacity,
                            const real_t &thermalDiffusivity) {
    density_ = density;
    viscosity_ = viscosity;
    heatCapacity_ = heatCapacity;
    thermalDiffusivity_ = thermalDiffusivity;
  }

  /// @brief Read relevant information from file
  /// @param filename Input xml file
  void ReadInputFromFile(const std::string &filename);

  /// @brief Initialize geometry and function spaces
  /// @param filename Filename with geometry, function, boundary conditions and
  /// assembly definitions
  /// @param numberOfRefinements Number of h-refinements
  /// @param numberDegreeElevations Number of degree elevations
  /// @param useDirectSolver If true, use direct solver to solve equations
  /// @param useEqualOrderBases If true, use equal order bases
  /// @param printSummary If true, print a summary of the geometry and function
  /// spaces
  void Init(const std::string &filename, const int numberOfRefinements,
            const int numberDegreeElevations, const bool useDirectSolver = true,
            const bool useEqualOrderBases = false, const bool printSummary = false);

  /// @brief Assemble the system matrix and rhs of the Stokes equation
  void AssembleFluidProblem();

  /// @brief Asemble the system matrix and rhs for the heat problem
  void AssembleHeatProblem();

  /// @brief Solve linear system of Stokes' system matrix and rhs
  void SolveFluidLinearSystem();

  /// @brief Solve linear system of the heat problem's system matrix and rhs
  void SolveHeatLinearSystem();

  /// @brief Exporting the field variables to a ParaView file
  /// @param fname Output file name
  /// @param plot_elements If true, plot patch boundaries
  /// @param sample_rate Samples per element
  /// @param export_b64 If true, export values in 64-bit binary format
  void ExportParaview(const std::string &fname, const int &sampleRate);

  // void AddObjectiveFunction(const int objective_function_selector);

  // std::vector<double> ComputeObjectiveFunctionValues();

  // // Compute the outflow via surface integral of x-velocity
  // double ComputeOutflow();

  // double ComputeVelDivergence();

  // void HRefine();

  // void ExportXML(const std::string &fname);
};

} // namespace pygadjoints
