import numpy as np
import splinepy as sp

EPS = 1e-8
BOX_LENGTH = 0.14
BOX_HEIGHT = 0.04

TILING = [2, 2, 4]
TWISTING_LAYERS = [2, 3]
# How much percent of tile length should be reserved for linking tiles
LINKAGE_THICKNESS = 0.1
# How much percent of box length should be reserved for forerun (the same length will
# be applied for the afterrun)
FORE_THICKNESS = 0.5


class SMXKernel:
    """Class to generate the 3D SMX mixer geometry"""

    def __init__(
        self,
        box_dimensions,
        tiling,
        twisting_layers,
        linkage_thickness,
        forerun_thickness,
        parameter_spline_initial,
        macro_spline_initial,
    ):
        self._box_dimensions = box_dimensions
        self._tiling = tiling
        self._z_tiling = tiling[-1]
        self._twisting_layers = twisting_layers
        self._linkage_thickness = linkage_thickness
        self._forerun_thickness = forerun_thickness
        self._parameter_spline_initial = parameter_spline_initial
        self._macro_spline_initial = macro_spline_initial

        self._compute_parametric_starting_points()

    def _determine_linkage_needs(self):
        """Determine where linkages are needed. Saves them as an array of integers,
        where these values are used:
            -1: linkage from twisted to non-twisted layer
            0: no linkage
            1: linkage from non-twisted to twisted
        """
        # Create an array, where 0 says non-twisted layer and 1 twisted layer
        twist_array = np.zeros(self._z_tiling)
        twist_array[self._twisting_layers] = 1
        self._linkage_array = twist_array[1:] - twist_array[:-1]

    def _compute_parametric_starting_points(self):
        """Compute the starting points of the tiles and linkages in the parametric
        domain."""
        self._determine_linkage_needs()

        # Determine the starting points in x- and y-direction
        x_grid_points = np.linspace(0, 1, self._tiling[0] + 1)
        y_grid_points = np.linspace(0, 1, self._tiling[1] + 1)
        # For the start points in z-direction also account for the linkages
        z_layer_length = np.ones(2 * self._z_tiling - 1)
        z_layer_length[1::2] = self._linkage_thickness * self._linkage_array
        # Remove the zeros where no linkage is
        z_layer_length = z_layer_length[z_layer_length != 0.0]
        # Add zero for starting point at inlet
        z_layer_length = np.insert(z_layer_length, 0, 0.0)
        z_grid_points = np.cumsum(z_layer_length)
        z_grid_points /= z_grid_points[-1]

        self._parametric_grid_points = [x_grid_points, y_grid_points, z_grid_points]

        self._parametric_start_points = sp.utils.data.cartesian_product(
            [x_grid_points[:-1], y_grid_points[:-1], z_grid_points[:-1]]
        )


if __name__ == "__main__":
    # Create initial parameter spline with controls in the corners and one layer (of 4
    # points) in the middle into the z-direction
    macro_spline_initial = sp.BSpline(
        degrees=[1, 1, 1],
        knot_vectors=[[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 0.5, 1, 1]],
        control_points=sp.utils.data.cartesian_product(
            [
                np.array([0, BOX_HEIGHT]),
                np.array([0, BOX_HEIGHT]),
                np.array([0, BOX_LENGTH / 2, BOX_LENGTH]),
            ]
        ),
    )

    parameter_spline_initial = sp.BSpline(
        degrees=[1, 1, 1],
        knot_vectors=macro_spline_initial.kvs,
        control_points=np.array(
            [0.1, 0.1, 0.1, 0.1, 0.4, 0.4, 0.4, 0.4, 0.2, 0.2, 0.2, 0.2]
        ).reshape(-1, 1),
    )

    geokernel = SMXKernel(
        box_dimensions=[BOX_HEIGHT, BOX_HEIGHT, BOX_LENGTH],
        tiling=TILING,
        twisting_layers=TWISTING_LAYERS,
        linkage_thickness=LINKAGE_THICKNESS,
        forerun_thickness=FORE_THICKNESS,
        parameter_spline_initial=parameter_spline_initial,
        macro_spline_initial=macro_spline_initial,
    )
