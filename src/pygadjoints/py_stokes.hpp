#include "pygadjoints/py_pde_base.hpp"

namespace pygadjoints {

using namespace gismo;

namespace py = pybind11;

enum class ObjectiveFunction : int {
  // Maximize ∫ ε(v) : ε(v) dΩ = 0.5⋅∫ ∇v : (∇v+(∇v)ᵀ) dΩ
  viscous_dissipation = 1
};

class StokesProblem : public PdeProblem {
private:
  using SolverType = gsSparseSolver<>::LU;
  // using SolverType = gsSparseSolver<>::BiCGSTABILUT;

public:
  StokesProblem() : PdeProblem(2, 2){};

  gsStopwatch timer;

  /// @brief Set up material constants
  /// @param viscosity Dynamic viscosity in Pa⋅s
  /// @param density Density in kg/m³
  void SetMaterialConstants(const real_t &viscosity, const real_t &density) {
    density_ = density;
    viscosity_ = viscosity;
  }

  /// @brief Read relevant information from file
  /// @param filename Input xml file
  void ReadInputFromFile(const std::string &filename);

  /// @brief Initialize geometry and function spaces
  /// @param filename Filename with geometry, function, boundary conditions and
  /// assembly definitions
  /// @param number_of_refinements Number of h-refinements
  /// @param number_degree_elevations Number of degree elevations
  /// @param print_summary If true, print a summary of the geometry and function
  /// spaes
  void Init(const std::string &filename, const int number_of_refinements,
            const int number_degree_elevations,
            const bool print_summary = false);

  /// @brief Assemble the system matrix and rhs of the Stokes equation
  void Assemble();

  /// @brief Solve linear system of system matrix and system rhs
  void SolveLinearSystem() { PdeProblem::SolveLinearSystem<SolverType>(); }

  /// @brief Exporting the field variables to a ParaView file
  /// @param fname Output file name
  /// @param plot_elements If true, plot patch boundaries
  /// @param sample_rate Samples per element
  /// @param export_b64 If true, export values in 64-bit binary format
  void ExportParaview(const std::string &fname, const bool &plot_elements,
                      const int &sample_rate, const bool &export_b64);

  void SetObjectiveFunction(const int objective_function_selector);

  double ComputeObjectiveFunctionValue();

  void ExportXML(const std::string &fname);

private:
  real_t density_{1};
  real_t viscosity_{1};

  // Indicator if input file has analytical solution(s)
  bool has_velocity_solution{false}, has_pressure_solution{false};

  // Analytical solutions to velocity and pressure
  gsFunctionExpr<> vel_analytical_solution{}, p_analytical_solution{};

  // Boundary conditions
  gsBoundaryConditions<> vel_bcs, p_bcs;
  bool has_pressure_bcs{false};

  // Function basis for velocity and pressure
  gsMultiBasis<> velocity_basis, pressure_basis;

  // Discretization spaces
  std::shared_ptr<space> pVelocity_space{nullptr}, pPressure_space{nullptr};

  // Partial solution values/expression for the variables
  std::shared_ptr<solution> pVelocity_solution{nullptr},
      pPressure_solution{nullptr};

  // Objective function
  ObjectiveFunction objective_function_{ObjectiveFunction::viscous_dissipation};
};

} // namespace pygadjoints
