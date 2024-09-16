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
