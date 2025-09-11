import numpy as np
import splinepy as sp
from splinepy.utils.data import cartesian_product as _cartesian_product

from sulzer_inverse import SulzerSMXInverse

EPS = 1e-8
BOX_LENGTH = 0.14
BOX_HEIGHT = 0.04

TILING = [2, 2, 4]
TWISTING_LAYERS = [2]
# How much percent of tile length should be reserved for linking tiles
LINKAGE_THICKNESS = 0.1
# How much percent of box length should be reserved for forerun (the same length will
# be applied for the afterrun)
FORERUN_AFTERRUN_THICKNESS = 0.5


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
        self._tile = SulzerSMXInverse()

        self.interfaces = None
        self.multipatch = None

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
        z_layer_length[1::2] = self._linkage_thickness * np.abs(
            self._linkage_array
        )
        # Remove the zeros where no linkage is
        z_layer_length = z_layer_length[z_layer_length != 0.0]
        # Add zero for starting point at inlet
        z_layer_length = np.insert(z_layer_length, 0, 0.0)
        z_grid_points = np.cumsum(z_layer_length)
        z_grid_points /= z_grid_points[-1]

        # Save the grid points into each points
        parametric_grid_points = [
            x_grid_points,
            y_grid_points,
            z_grid_points,
        ]

        # Determine the indices of the rows in the grid points array which correspond
        # to starting points of tiles
        tile_z_indices = np.arange(self._z_tiling) + np.hstack(
            ([0], np.cumsum(np.abs(self._linkage_array)))
        ).astype(np.int64)
        # Mark every invalid (aka end points) as -1
        tile_x_indices_marked = np.arange(self._tiling[0] + 1, dtype=np.int64)
        tile_y_indices_marked = np.arange(self._tiling[1] + 1, dtype=np.int64)
        tile_x_indices_marked[-1] = -1
        tile_y_indices_marked[-1] = -1
        tile_z_indices_marked = -1 * np.ones(
            len(parametric_grid_points[-1]), dtype=np.int64
        )
        tile_z_indices_marked[tile_z_indices] = tile_z_indices

        # Save the grid points
        self._parametric_grid_points = _cartesian_product(
            parametric_grid_points
        )

        # Determine which points in the grid points array belong to the start of tiles
        self._tile_start_grid_indices = _cartesian_product(
            [
                tile_x_indices_marked,
                tile_y_indices_marked,
                tile_z_indices_marked,
            ]
        )
        self._tile_start_grid_indices = np.where(
            ~np.any(self._tile_start_grid_indices == -1, axis=1)
        )[0]

        if not len(self._tile_start_grid_indices) == np.prod(self._tiling):
            raise ValueError(
                "Something went wrong in determining the starting points of the tiles"
            )

        # self._parametric_start_points = _cartesian_product(
        #     [x_grid_points[:-1], y_grid_points[:-1], z_grid_points[:-1]]
        # )

    def _generate_geometry(self):
        # From compute grid points choose the right indices to get the corner points
        # of each tile (still in parametric domain)
        x_npoints = self._tiling[0] + 1
        y_npoints = self._tiling[1] + 1
        layer_npoints = x_npoints * y_npoints
        # Determine stencil of indices to which points belong to a tile
        tile_point_indices_stencil = np.array(
            [
                0,
                1,
                x_npoints,
                x_npoints + 1,
                layer_npoints,
                layer_npoints + 1,
                layer_npoints + x_npoints,
                layer_npoints + x_npoints + 1,
            ]
        )

        # Determine the tiles' starting points in the physical domain
        grid_points_physical = self._macro_spline_initial.evaluate(
            self._parametric_grid_points
        )
        # start_grid_points_physical = grid_points_physical[self._tile_start_grid_indices])
        
        dummy_tile = sp.helpme.create.box(1,1,1)
        tile_evaluation_points = SulzerSMXInverse._evaluation_points
        
        for tile_start_grid_index in self._tile_start_grid_indices:
            # Compute the tile's corner points in physical space
            tile_corner_points_indices = (
                tile_point_indices_stencil + tile_start_grid_index
            )
            tile_corner_points = grid_points_physical[
                tile_corner_points_indices
            ]
            # Evaluate the parameters
            tile_corner_points_parametric = self._parametric_grid_points[tile_corner_points_indices]
            # Get the evaluation points in the parametric domain of the parameter spline
            dummy_tile.cps = tile_corner_points_parametric
            parameters_evaluation_points_parametric = dummy_tile.evaluate(tile_evaluation_points)
            # Evaluate parameter spline on tile
            tile_parameters = parameter_spline_initial.evaluate(parameters_evaluation_points_parametric)
            # Evaluate tile
            tile_patches, _ = self._tile.create_tile(tile_parameters)
            sp.Multipatch(tile_patches).show()
            

        raise ValueError()

    def generate_microstructure(self, macro_sensitivities=None):
        # Generate the mixer
        self.multipatch = self._generate_geometry()

        # Reuse existing interfaces
        if self.interfaces is None:
            self.multipatch.determine_interfaces()
            self.interfaces = self.multipatch.interfaces
        else:
            self.multipatch.interfaces = self.interfaces

        # Assign boundaries from identifier functions
        for (
            identifier_function,
            boundary_id,
        ) in self.boundary_identifier_dict.items():
            self.multipatch.boundary_from_function(
                identifier_function, boundary_id=boundary_id
            )


if __name__ == "__main__":
    # Create initial parameter spline with controls in the corners and one layer (of 4
    # points) in the middle into the z-direction
    macro_spline_initial = sp.BSpline(
        degrees=[1, 1, 1],
        knot_vectors=[[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 0.5, 1, 1]],
        control_points=_cartesian_product(
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
        forerun_thickness=FORERUN_AFTERRUN_THICKNESS,
        parameter_spline_initial=parameter_spline_initial,
        macro_spline_initial=macro_spline_initial,
    )

    geokernel.generate_microstructure()
