add_subdirectory(pybind11)

set(GISMO_BUILD_EXAMPLES OFF)
set(GISMO_WITH_OPENMP ${PYGADJOINTS_USE_OPENMP})
set(GISMO_OPTIONAL "gsIncompressibleFlow")
add_subdirectory(gismo)
