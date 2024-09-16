# pygadjoint
A python wrapper for `g+smo` specifically used for adjoint shape optimization of
specific geometries (microstructures / lattice structures / metamaterials).


## Installation

### Geometry: use `splinepy`

Installation instructions are found on the main repo.
To make the export to g+smo compatible xml-files easier, prefarably don't install
the main branch, but perform the following instructions:

```bash
git clone git@github.com:markriegler/splinepy.git -b ft-gismo-bc-export
cd splinepy
git submodule update --init --recursive
pip install -e .
```

### `pygadjoints` install

To install `pygadjoints` and specifically use the Stokes equation, type in:

```bash
git clone git@github.com:markriegler/pygadjoints.git -b ft-stokes
cd pygadjoints
git submodule update --init --recursive
CMAKE_BUILD_PARALLEL_LEVEL=<n-parallel> python3 setup.py develop
```

## Features

### Supported PDEs

- Diffusion
- Linear Elasticity
- Stokes equation

### Shape optimization

The optimization uses `scipy.optimize` as the optimization driver.