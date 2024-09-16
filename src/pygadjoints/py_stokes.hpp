#include "pygadjoints/py_pde_base.hpp"

namespace pygadjoints {

using namespace gismo;

namespace py = pybind11;

class StokesProblem : public PdeProblem {
private:
  using SolverType = gsSparseSolver<>::LU;
  // using SolverType = gsSparseSolver<>::BiCGSTABILUT;

public:
  StokesProblem() : PdeProblem(2, 2){};

  gsStopwatch timer;

  void SetMaterialConstants(const real_t &viscosity, const real_t &density) {
    density_ = density;
    viscosity_ = viscosity;
  }

  void ExportParaview(const std::string &fname, const bool &plot_elements,
                      const int &sample_rate, const bool &export_b64) {
    Timer timer("Exporting Paraview");
  }

  void Init(const std::string &filename, const int number_of_refinements,
            const int number_degree_elevations,
            const bool print_summary = false) {
    const Timer timer("Initialisation");

    // }
  }

  void Assemble() { const Timer timer("Assemble"); }

  void SolveLinearSystem() { PdeProblem::SolveLinearSystem<SolverType>(); }

private:
  real_t density_{1};
  real_t viscosity_{1};
};

} // namespace pygadjoints
