#include <pybind11/pybind11.h>

// declare signatures
void add_adjoint_class(pybind11::module_ &);
void add_diffusion_problem(pybind11::module_ &);
void add_stokes_temperature_problem(pybind11::module_ &);

PYBIND11_MODULE(pygadjoints, m) {
  add_adjoint_class(m);
  add_diffusion_problem(m);
  add_stokes_temperature_problem(m);
}
