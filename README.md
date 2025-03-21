# pygadjoints
python gismo adjoint

This branch is focused on fluid simulations.

# Installation

The installation is using Ninja as a generator. This makes building the code when making a small code change faster.

This branch also takes [`gsIncompressibleFlow`](https://github.com/gismo/gsIncompressibleFlow) as an additional gismo module.

To get the code running, type in:

```bash
git submodule update --init --recursive
CMAKE_BUILD_PARALLEL_LEVEL=<n-parallel> python3 setup.py develop
```

In order to build in Debug mode, type in the following:

```bash
CMAKE_BUILD_PARALLEL_LEVEL=<n-parallel> python3 setup.py develop --debug
```

#### Quick info for developers

If you have already built the whole library, made code changes and want to rebuild, you can also type in:

```bash
python3 setup.py build_ext --inplace
```