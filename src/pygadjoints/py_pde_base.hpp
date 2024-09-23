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
  std::shared_ptr<gsExprEvaluator<>> pExpr_evaluator;

  /// List of expression that describes the last calculated solution variables
  std::shared_ptr<solution> pSolution_expression = nullptr; // map

  /// Expression that describes the last calculated solution
  std::shared_ptr<space> pFunction_space = nullptr; // map

  /// Expression that describes the last calculated solution
  std::shared_ptr<geometryMap> pGeometry_expression = nullptr;

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
  std::shared_ptr<const gsSparseMatrix<>> pSystem_matrix = nullptr;

  // Linear System Matrix
  std::shared_ptr<gsMatrix<>> pCtps_sensitivities_matrix = nullptr;

  // Linear System RHS
  std::shared_ptr<gsMatrix<>> pSystem_rhs = nullptr;

  // Solution of the adjoint problem
  std::shared_ptr<gsMatrix<>> pLagrange_multipliers = nullptr; // map

  // DOF-Mapper
  std::shared_ptr<gsDofMapper> pDof_mapper = nullptr;

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

  void PrepareMatrixAndRhs() {
    // Assign to matrix and vector
    pSystem_matrix =
        std::make_shared<const gsSparseMatrix<>>(expr_assembler_pde.matrix());
    pSystem_rhs = std::make_shared<gsMatrix<>>(expr_assembler_pde.rhs());

    // Clear for future evaluations
    expr_assembler_pde.clearMatrix(true);
    expr_assembler_pde.clearRhs();
  }

  template <typename SolverType> void SolveLinearSystem() {
    const Timer timer("SolveLinearSystem");

    ///////////////////
    // Linear Solver //
    ///////////////////
    if ((!pSystem_matrix) || (!pSystem_rhs)) {
      gsWarn << "System matrix and system rhs are required for solving!"
             << std::endl;
      return;
    }
    // Initialize linear solver
    SolverType solver;
    solver.compute(*pSystem_matrix);
    solVector = solver.solve(*pSystem_rhs);
  }

  double ComputeVolume() {
    const Timer timer("ComputeVolume");

    // Compute volume of domain
    if (!pExpr_evaluator) {
      GISMO_ERROR("ExprEvaluator not initialized");
    }
    return pExpr_evaluator->integral(meas(*pGeometry_expression));
  }

  py::array_t<double> GetParameterSensitivities() {
    if (!pCtps_sensitivities_matrix) {
      throw std::runtime_error("CTPS Matrix has not been computed yet.");
    }

    const int _rows = pCtps_sensitivities_matrix->rows();
    const int _cols = pCtps_sensitivities_matrix->cols();
    py::array_t<double> matrix(_rows * _cols);
    double *matrix_ptr = static_cast<double *>(matrix.request().ptr);
    for (int i{}; i < _rows; i++) {
      for (int j{}; j < _cols; j++) {
        matrix_ptr[i * _cols + j] = (*pCtps_sensitivities_matrix)(i, j);
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
    size_t n_patches_new, n_patches_old;
    n_patches_new = mp_new.nPatches();
    n_patches_old = mp_pde.nPatches();

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
      n_new_coefs = mp_new.patch(patch_id).coefs().size();
      n_old_coefs = mp_pde.patch(patch_id).coefs().size();
      if (n_new_coefs != n_old_coefs) {
        throw std::runtime_error(
            "This does not work - I am fucked. Expected number of "
            "coefficients " +
            std::to_string(n_old_coefs) + ", but got " +
            std::to_string(n_new_coefs));
      }
      for (size_t i_coef = 0; i_coef != n_old_coefs; i_coef++) {
        mp_pde.patch(patch_id).coefs().at(i_coef) =
            mp_new.patch(patch_id).coefs().at(i_coef);
      }
    }
    // pGeometry_expression->copyCoefs(mp_new);
  }
};

} // namespace pygadjoints
