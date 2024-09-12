#include <gismo.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <chrono>

#ifdef PYGADJOINTS_USE_OPENMP
#include <omp.h>
#endif

#include "pygadjoints/custom_expression.hpp"
#include "pygadjoints/timer.hpp"

namespace pygadjoints {

using namespace gismo;

namespace py = pybind11;

class PdeProblem {
protected:
  // Typedefs
  typedef gsExprAssembler<>::geometryMap geometryMap;
  typedef gsExprAssembler<>::variable variable;
  typedef gsExprAssembler<>::space space;
  typedef gsExprAssembler<>::solution solution;

  /// Expression assembler related to the forward problem
  gsExprAssembler<> expr_assembler_pde;

  /// Multipatch object of the forward problem
  gsMultiPatch<> mp_pde;

  /// Expression assembler related to the forward problem
  std::shared_ptr<gsExprEvaluator<>> expr_evaluator_ptr;

  /// List of expression that describes the last calculated solution variables
  std::shared_ptr<solution> solution_expression_ptr = nullptr; // map

  /// Expression that describes the last calculated solution
  std::shared_ptr<space> basis_function_ptr = nullptr; // map

  /// Expression that describes the last calculated solution
  std::shared_ptr<geometryMap> geometry_expression_ptr = nullptr;

  /// Global reference to solution vector
  gsMatrix<> solVector{};

  /// Boundary conditions pointer
  gsBoundaryConditions<> boundary_conditions; // mamp

  /// Source function
  gsFunctionExpr<> source_function{};

  /// Solution function
  gsFunctionExpr<> analytical_solution{}; // map

  /// Solution function flag
  bool has_solution{false};

  // Flag for source function
  bool has_source_id{false};

  /// Function basis
  gsMultiBasis<> function_basis{}; // map

  // Linear System Matrix
  std::shared_ptr<const gsSparseMatrix<>> system_matrix = nullptr;

  // Linear System Matrix
  std::shared_ptr<gsMatrix<>> ctps_sensitivities_matrix_ptr = nullptr;

  // Linear System RHS
  std::shared_ptr<gsMatrix<>> system_rhs = nullptr;

  // Solution of the adjoint problem
  std::shared_ptr<gsMatrix<>> lagrange_multipliers_ptr = nullptr; // map

  // DOF-Mapper
  std::shared_ptr<gsDofMapper> dof_mapper_ptr = nullptr;

  // Number of refinements in the current iteration
  int n_refinements{};

  // Number of degree elevations
  int n_degree_elevations{};

  // Number of refinements in the current iteration
  int dimensionality_{};

#ifdef PYGADJOINTS_USE_OPENMP
  int n_omp_threads{1};
#endif

public:
  // Constructor
  PdeProblem(index_t numTest, index_t numTrial)
      : expr_assembler_pde(numTest, numTrial) {
#ifdef PYGADJOINTS_USE_OPENMP
    omp_set_num_threads(std::min(omp_get_max_threads(), n_omp_threads));
#endif
  }

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

  template <typename SolverType> void SolveLinearSystem() {
    const Timer timer("SolveLinearSystem");

    ///////////////////
    // Linear Solver //
    ///////////////////
    if ((!system_matrix) || (!system_rhs)) {
      gsWarn << "System matrix and system rhs are required for solving!"
             << std::endl;
      return;
    }
    // Initialize linear solver
    SolverType solver;
    solver.compute(*system_matrix);
    solVector = solver.solve(*system_rhs);
  }

  double ComputeVolume() {
    const Timer timer("ComputeVolume");

    // Compute volume of domain
    if (!expr_evaluator_ptr) {
      GISMO_ERROR("ExprEvaluator not initialized");
    }
    return expr_evaluator_ptr->integral(meas(*geometry_expression_ptr));
  }

  py::array_t<double> GetParameterSensitivities() {
    if (!ctps_sensitivities_matrix_ptr) {
      throw std::runtime_error("CTPS Matrix has not been computed yet.");
    }

    const int _rows = ctps_sensitivities_matrix_ptr->rows();
    const int _cols = ctps_sensitivities_matrix_ptr->cols();
    py::array_t<double> matrix(_rows * _cols);
    double *matrix_ptr = static_cast<double *>(matrix.request().ptr);
    for (int i{}; i < _rows; i++) {
      for (int j{}; j < _cols; j++) {
        matrix_ptr[i * _cols + j] = (*ctps_sensitivities_matrix_ptr)(i, j);
      }
    }
    matrix.resize({_rows, _cols});
    return matrix;
  }

  void UpdateGeometry(const std::string &fname, const bool &topology_changes) {
    const Timer timer("UpdateGeometry");
    if (topology_changes) {
      throw std::runtime_error("Not Implemented!");
    }

    // Import mesh and load relevant information
    gsMultiPatch<> mp_new;

    gsFileData<> fd(fname);
    fd.getId(0, mp_new);

    // This update does not require refinement or elevation, in theory the mp is
    // not touched, only the solution field

    // Ignore all other information!
    if (mp_new.nPatches() != mp_pde.nPatches()) {
      throw std::runtime_error(
          "This does not work - I am fucked. Expected number of "
          "patches " +
          std::to_string(mp_pde.nPatches()) + ", but got " +
          std::to_string(mp_new.nPatches()));
    }
    // Manually update coefficients as to not overwrite any precomputed
    // values
    for (size_t patch_id{}; patch_id < mp_new.nPatches(); patch_id++) {
      if (mp_new.patch(patch_id).coefs().size() !=
          mp_pde.patch(patch_id).coefs().size()) {
        throw std::runtime_error(
            "This does not work - I am fucked. Expected number of "
            "coefficients " +
            std::to_string(mp_pde.patch(patch_id).coefs().size()) +
            ", but got " +
            std::to_string(mp_new.patch(patch_id).coefs().size()));
      }
      for (int i_coef = 0; i_coef != mp_pde.patch(patch_id).coefs().size();
           i_coef++) {
        mp_pde.patch(patch_id).coefs().at(i_coef) =
            mp_new.patch(patch_id).coefs().at(i_coef);
      }
    }
    // geometry_expression_ptr->copyCoefs(mp_new);
  }
};

} // namespace pygadjoints
