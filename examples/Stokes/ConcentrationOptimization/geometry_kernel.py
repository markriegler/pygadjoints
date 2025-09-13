import numpy as np
import splinepy as sp
from scipy.interpolate import LinearNDInterpolator
from splinepy.utils.data import cartesian_product as _cartesian_product
from sulzer_inverse import SulzerSMXInverse

EPS = 1e-8
BOX_LENGTH = 0.14
BOX_HEIGHT = 0.04

TILING = [2, 2, 6]
TWISTING_LAYERS = [2, 3]
# How much percent of tile length should be reserved for linking tiles
LINKAGE_THICKNESS = 0.2
# How much percent of box length should be reserved for forerun (the same length will
# be applied for the afterrun)
FORERUN_AFTERRUN_THICKNESS = 0.5
# Determines the percentage of the whole forerun length to be dedicated to the linkage
FORERUN_AFTERRUN_LINKAGE_LENGTH = 0.1
INLET_BOUNDARY_ID = 2
OUTLET_BOUNDARY_ID = 3


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
        boundary_identifier_dict=None,
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
        self.boundary_identifier_dict = boundary_identifier_dict

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
        # Determine which ids in the grid in z-direction correspond to the start of the
        # tile layers
        self._tile_layer_ids = np.arange(
            self._tiling[2], dtype=self._linkage_array.dtype
        )
        self._tile_layer_ids[1:] += np.cumsum(np.abs(self._linkage_array))

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
        self._parametric_grid_points_single = parametric_grid_points

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

        dummy_tile = sp.helpme.create.box(1, 1, 1)
        tile_evaluation_points = SulzerSMXInverse._evaluation_points

        all_patches = []

        # Auxiliary values for the linear interpolation
        e = np.array([0, 1])
        E = _cartesian_product([e, e, e])

        for tile_start_grid_index in self._tile_start_grid_indices:
            # Compute the tile's corner points in physical space
            tile_corner_points_indices = (
                tile_point_indices_stencil + tile_start_grid_index
            )
            tile_corner_points_physical = grid_points_physical[
                tile_corner_points_indices
            ]
            # Evaluate the parameters
            tile_corner_points_parametric = self._parametric_grid_points[
                tile_corner_points_indices
            ]
            # Get the evaluation points in the parametric domain of the parameter spline
            dummy_tile.cps = tile_corner_points_parametric
            parameters_evaluation_points_parametric = dummy_tile.evaluate(
                tile_evaluation_points
            )
            # Evaluate parameter spline on tile
            tile_parameters = parameter_spline_initial.evaluate(
                parameters_evaluation_points_parametric
            )
            # Evaluate tile in tile's parametric space
            tile_patches_parametric, _ = self._tile.create_tile(
                tile_parameters
            )

            tile_linear_interpolator = LinearNDInterpolator(
                E, tile_corner_points_physical
            )

            # Trilinearly interpolate to fit physical domain
            tile_patches = []
            for patch_parametric in tile_patches_parametric:
                new_patch = patch_parametric.copy()
                new_cps = tile_linear_interpolator(patch_parametric.cps.copy())
                new_patch.cps = new_cps
                tile_patches.append(new_patch)

            all_patches += tile_patches

        # Start with the fore and afterrun
        forerun_length = FORERUN_AFTERRUN_THICKNESS * BOX_LENGTH
        forerun_linkage_length = (
            forerun_length * FORERUN_AFTERRUN_LINKAGE_LENGTH
        )
        forerun_z_points = np.array(
            [-forerun_length, -forerun_length / 2, -forerun_linkage_length]
        )
        afterrun_z_points = BOX_LENGTH - np.flip(forerun_z_points)

        sp.helpme.create.box(1, 1, 1)

        # inlet_xy_points = self._macro_spline_initial.evaluate(self._parametric_grid_points[:layer_npoints])
        x_points_para = np.linspace(0, 1, self._tiling[0] + 1)
        y_points_para = np.linspace(0, 1, self._tiling[1] + 1)
        xy_parametric_grid_evaluation_points = _cartesian_product(
            [x_points_para[:-1], y_points_para[:-1], np.array([0.0, 1.0])]
        )
        xy_box_points = self._macro_spline_initial.evaluate(
            xy_parametric_grid_evaluation_points
        )
        forerun_start_xy_points, afterrun_start_xy_points = np.split(
            xy_box_points, 2, axis=0
        )
        # Compute the length in x-directions for the inlet tiles
        box_grid_points_parametric = _cartesian_product(
            [x_points_para, y_points_para[:-1], np.array([0.0, 1.0])]
        )
        all_box_grid_points = self._macro_spline_initial.evaluate(
            box_grid_points_parametric
        )
        dx_box_points = [
            np.diff(points, axis=0)[:, 0]
            for points in np.split(
                all_box_grid_points, 2 * self._tiling[1], axis=0
            )
        ]
        forerun_dx, afterrun_dx = np.split(np.hstack(dx_box_points), 2)
        # compute the length y-direction for the inlet tiles
        box_grid_points_parametric = _cartesian_product(
            [x_points_para[:-1], y_points_para, np.array([0.0, 1.0])]
        )
        all_box_grid_points = self._macro_spline_initial.evaluate(
            box_grid_points_parametric
        )
        # Reorder all rows such that the y-coordinate is the first ascending direction
        reorder_indices = []
        offset = 0
        for grid_points in np.split(all_box_grid_points, 2, axis=0):
            new_indices = np.concatenate(
                [
                    offset + np.arange(i, len(grid_points), self._tiling[0])
                    for i in range(self._tiling[0])
                ]
            )
            reorder_indices.append(new_indices)
            offset += (self._tiling[0] + 1) * self._tiling[1]
        all_box_grid_points = all_box_grid_points[np.hstack(reorder_indices)]
        dy_box_points = [
            np.diff(points, axis=0)[:, 1]
            for points in np.split(
                all_box_grid_points, 2 * self._tiling[0], axis=0
            )
        ]
        forerun_dy, afterrun_dy = np.split(np.hstack(dy_box_points), 2)
        forerun_dx_dy = np.column_stack([forerun_dx, forerun_dy])
        afterrun_dx_dy = np.column_stack([afterrun_dx, afterrun_dy])

        # Create forerun patches
        for xy_start_point, dx_dy in zip(
            forerun_start_xy_points, forerun_dx_dy
        ):
            # Make 4 quarters of beam
            quarter_start_points = _cartesian_product(
                [
                    np.array(
                        [xy_start_point[0], xy_start_point[0] + dx_dy[0] / 2]
                    ),
                    np.array(
                        [xy_start_point[1], xy_start_point[1] + dx_dy[1] / 2]
                    ),
                ]
            )
            for z_start, dz in zip(
                forerun_z_points[:2], np.diff(forerun_z_points[:3])
            ):
                quarter_beam = sp.helpme.create.box(
                    dx_dy[0] / 2, dx_dy[1] / 2, dz
                )
                for quarter_start_point in quarter_start_points:
                    new_patch = quarter_beam.copy()
                    new_patch.cps += np.hstack(
                        (quarter_start_point, np.array([z_start]))
                    )
                    all_patches.append(new_patch)

        # Create afterrun patches
        for xy_start_point, dx_dy in zip(
            afterrun_start_xy_points, afterrun_dx_dy
        ):
            # Make 4 quarters of beam
            quarter_start_points = _cartesian_product(
                [
                    np.array(
                        [xy_start_point[0], xy_start_point[0] + dx_dy[0] / 2]
                    ),
                    np.array(
                        [xy_start_point[1], xy_start_point[1] + dx_dy[1] / 2]
                    ),
                ]
            )
            for z_start, dz in zip(
                afterrun_z_points[:2], np.diff(afterrun_z_points)
            ):
                quarter_beam = sp.helpme.create.box(
                    dx_dy[0] / 2, dx_dy[1] / 2, dz
                )
                for quarter_start_point in quarter_start_points:
                    new_patch = quarter_beam.copy()
                    new_patch.cps += np.hstack(
                        (quarter_start_point, np.array([z_start]))
                    )
                    all_patches.append(new_patch)

        # Create forerun linkage patches
        linkage_evaluation_points_para = _cartesian_product(
            [x_points_para, y_points_para, np.array([0.0, 1.0])]
        )
        # For nontwisted layer dy is needed; similarly dx is needed for twisted layer
        xyz_points_linkage = self._macro_spline_initial.evaluate(
            linkage_evaluation_points_para
        )
        # x-points are the tiles' x-points plus the points in the middle
        x_points_linkage = np.split(xyz_points_linkage[:, 0], 2 * y_npoints)

        def interleave_with_means(array):
            n = len(array)
            if array.ndim == 1:
                result = np.empty(2 * n - 1, dtype=array.dtype)
                # Place original values
                result[0::2] = array
                # Compute means for in-between positions
                result[1::2] = (array[:-1] + array[1:]) / 2
            else:
                result = np.empty(
                    (2 * n - 1, array.shape[1]), dtype=array.dtype
                )
                result[0::2, :] = array
                result[1::2, :] = (array[:-1, :] + array[1:, :]) / 2

            return result

        # x points for linkage are the tile x-points + a point in the middle for
        # each tile
        for i in range(2 * y_npoints):
            x_points_linkage[i] = interleave_with_means(x_points_linkage[i])
        # In y-direction we also interleave the tiles in two
        x_points_linkage_forerun = interleave_with_means(
            np.vstack(x_points_linkage[:y_npoints])
        )
        x_points_linkage_afterrun = interleave_with_means(
            np.vstack(x_points_linkage[y_npoints:])
        )

        # Points at tile interfaces have to be doubled
        def double_interface_points(array):
            x_indices = np.arange(len(array))
            x_repeats = np.repeat(
                x_indices, np.where(x_indices % 2 == 0, 2, 1)
            )[1:-1]
            return array[x_repeats, :]

        x_points_linkage_forerun = double_interface_points(
            x_points_linkage_forerun
        )
        x_points_linkage_afterrun = double_interface_points(
            x_points_linkage_afterrun
        )

        # For the y-points also account for the tile parameters
        # For every tile we have to account for the tile's y-coordinates
        def compute_y_linkage_points(y_points, parameters):
            """

            Returns
            ------------
            new_y_points_both: list<np.ndarray>
                Returns two arrays of y_points. First one is for the "outline", which is
                for the fore- or afterrun. The second one is for the actual tiles at
                the start/end of the actual mixer.
            """
            unique_indices = np.arange(len(y_points))
            row_indices = np.split(unique_indices, x_npoints)
            # Only the middle row indices should be duplicated
            duplicating_indices = np.hstack(
                (
                    row_indices[0],
                    np.hstack(
                        [np.tile(indices, 2) for indices in row_indices[1:-1]]
                    ),
                    row_indices[-1],
                )
            )
            y_points_corners_outline = y_points[duplicating_indices]
            parameters_corners = parameters[duplicating_indices]
            # Compute the dy values at the corners
            dy_values = np.diff(
                y_points[unique_indices].reshape(y_npoints, x_npoints), axis=0
            )
            dy_corner_values = np.repeat(dy_values, 2, axis=0).ravel()
            y_shift_values = dy_corner_values * parameters_corners
            # y-shift should be negative for upper edges of tiles
            y_shift_values = y_shift_values.reshape(-1, x_npoints)
            y_shift_values[1::2, :] *= -1.0
            y_shift_values = y_shift_values.ravel()

            y_points_tile_corners = y_points_corners_outline + y_shift_values

            # Interleave in x-direction
            new_y_points_both = []
            for y_points_i in [
                y_points_corners_outline,
                y_points_tile_corners,
            ]:
                new_y_points = np.vstack(
                    [
                        interleave_with_means(points)
                        for points in np.split(y_points_i, 2 * self._tiling[1])
                    ]
                )
                new_y_points = np.vstack(
                    [
                        interleave_with_means(points)
                        for points in np.split(new_y_points, self._tiling[1])
                    ]
                )
                new_y_points_both.append(new_y_points)
            return new_y_points_both

        new_y_points_outline = []
        new_y_points_linkage = []
        for y_points_linkage, parameters_linkage in zip(
            np.split(xyz_points_linkage[:, 1], 2),
            np.split(
                self._parameter_spline_initial.evaluate(
                    linkage_evaluation_points_para
                ).ravel(),
                2,
            ),
        ):
            new_y_points_both = compute_y_linkage_points(
                y_points_linkage, parameters_linkage
            )
            new_y_points_outline.append(new_y_points_both[0])
            new_y_points_linkage.append(new_y_points_both[1])

        def grid_points_to_tile_points(array, rows_to_not_build_grid=[]):
            """Turn an array of points and return a list of the points in tiles

            Parameters
            -----------------
            array: np.ndarray
                Array of points, must be 2-dimensional
            rows_to_not_build_grid: list<int>
                At these rows there should not be a build a tile, meaning that the
                corners of the tiles will not be returned for those rows
            """
            assert array.ndim == 2, "Array should be 2-dimensional"
            n_rows, n_cols = array.shape
            mask = np.array([0, 1, n_cols, n_cols + 1])
            # Remove the last row
            row_indices = np.arange((n_rows - 1) * n_cols)
            # Remove the right column
            row_indices = row_indices[(row_indices + 1) % array.shape[1] != 0]
            # Remove the rows where not to build a grid
            row_indices = row_indices[
                ~np.isin(row_indices // n_cols, rows_to_not_build_grid)
            ]
            mask = mask + row_indices[:, None]
            # Compute the corner values of every tile
            result = np.split(
                array.ravel()[mask.ravel()],
                (n_rows - 1 - len(rows_to_not_build_grid)) * (n_cols - 1),
            )
            return result

        rows_to_not_build_grid = 3 * np.arange(1, self._tiling[1]) - 1
        x_cps_linkage_forerun = grid_points_to_tile_points(
            x_points_linkage_forerun, rows_to_not_build_grid
        )
        x_cps_linkage_afterrun = grid_points_to_tile_points(
            x_points_linkage_afterrun, rows_to_not_build_grid
        )
        y_cps_outline_forerun = grid_points_to_tile_points(
            new_y_points_outline[0], rows_to_not_build_grid
        )
        y_cps_outline_afterrun = grid_points_to_tile_points(
            new_y_points_outline[1], rows_to_not_build_grid
        )
        y_cps_linkage_forerun = grid_points_to_tile_points(
            new_y_points_linkage[0], rows_to_not_build_grid
        )
        y_cps_linkage_afterrun = grid_points_to_tile_points(
            new_y_points_linkage[1], rows_to_not_build_grid
        )

        # Append patches of fore- and afterrun
        for x_cps_layer, y_cps_layer_zmin, y_cps_layer_zmax, z_points in zip(
            [x_cps_linkage_forerun, x_cps_linkage_afterrun],
            [y_cps_outline_forerun, y_cps_linkage_afterrun],
            [y_cps_linkage_forerun, y_cps_outline_afterrun],
            [
                np.array([forerun_z_points[-1], 0.0]),
                np.array([BOX_LENGTH, afterrun_z_points[0]]),
            ],
        ):
            z_cps = np.repeat(z_points, len(x_cps_layer[0]))
            for x_cps, y_cps_zmin, y_cps_zmax in zip(
                x_cps_layer, y_cps_layer_zmin, y_cps_layer_zmax
            ):
                dummy_tile.cps = np.column_stack(
                    (
                        np.tile(x_cps, 2),
                        np.hstack((y_cps_zmin, y_cps_zmax)),
                        z_cps,
                    )
                )
                all_patches.append(dummy_tile.copy())

        all_patches += self.compute_layer_linkage_patches()

        return sp.Multipatch(all_patches)

    def compute_layer_linkage_patches(self):
        # Determine the layers where the linkage t2nt (twist to non-twist) and nt2t
        # (non-twist to twist) are
        tile_layer_ids = self._tile_layer_ids[:-1]
        twist_indices = np.nonzero(self._linkage_array)
        linkage_layer_ids = tile_layer_ids[twist_indices].astype(np.int64) + 1
        twist_ids = self._linkage_array[twist_indices]
        # nt2t_layer_ids = tile_layer_ids[self._linkage_array > 0].astype(np.int64) + 1
        # t2nt_layer_ids = tile_layer_ids[self._linkage_array < 0].astype(nt2t_layer_ids.dtype) + 1

        x_tiling = self._tiling[0]
        y_tiling = self._tiling[1]
        x_npoints = x_tiling + 1
        y_npoints = y_tiling + 1

        def interleave_with_means(array):
            n = len(array)
            if array.ndim == 1:
                result = np.empty(2 * n - 1, dtype=array.dtype)
                # Place original values
                result[0::2] = array
                # Compute means for in-between positions
                result[1::2] = (array[:-1] + array[1:]) / 2
            else:
                result = np.empty(
                    (2 * n - 1, array.shape[1]), dtype=array.dtype
                )
                result[0::2, :] = array
                result[1::2, :] = (array[:-1, :] + array[1:, :]) / 2

            return result

        def compute_x_linkage_points(x_points, parameters):
            """

            Returns
            ------------
            new_x_points_both: list<np.ndarray>
                Returns two arrays of y_points. First one is for the "outline", which is
                for the fore- or afterrun. The second one is for the actual tiles at
                the start/end of the actual mixer.
            """
            unique_indices = np.arange(len(x_points))
            column_indices = np.split(unique_indices, y_npoints)
            # Only the middle column indices should be duplicated
            duplicating_indices = np.hstack(
                [np.repeat(x_indices, 2)[1:-1] for x_indices in column_indices]
            )
            x_points_corners_outline = x_points[duplicating_indices]
            parameters_corners = parameters[duplicating_indices]
            # Compute the dx values at the corners
            dx_values = np.diff(
                x_points[unique_indices].reshape(y_npoints, x_npoints), axis=1
            )
            dx_corner_values = np.repeat(dx_values, 2, axis=0).ravel()
            x_shift_values = dx_corner_values * parameters_corners
            # x-shift should be negative for right edges of tiles
            x_shift_values[1::2] *= -1.0

            x_points_tile_corners = x_points_corners_outline + x_shift_values

            # Interleave in x-direction
            new_x_points_both = []
            for x_points_i in [
                x_points_corners_outline,
                x_points_tile_corners,
            ]:
                # Interleave in y-direction and stack rows vertically
                new_x_points = interleave_with_means(
                    x_points_i.reshape(y_npoints, -1)
                )

                # Interleave in x-direction
                all_new_x_points = np.vstack(
                    [
                        np.hstack(
                            [
                                interleave_with_means(points)
                                for points in np.split(row_points, x_tiling)
                            ]
                        )
                        for row_points in new_x_points
                    ]
                )
                new_x_points_both.append(all_new_x_points)
            return new_x_points_both

        def compute_y_linkage_points(y_points, parameters):
            """

            Returns
            ------------
            new_y_points_both: list<np.ndarray>
                Returns two arrays of y_points. First one is for the "outline", which is
                for the fore- or afterrun. The second one is for the actual tiles at
                the start/end of the actual mixer.
            """
            unique_indices = np.arange(len(y_points))
            row_indices = np.split(unique_indices, y_npoints)
            # Only the middle row indices should be duplicated
            duplicating_indices = np.hstack(
                (
                    row_indices[0],
                    np.hstack(
                        [np.tile(indices, 2) for indices in row_indices[1:-1]]
                    ),
                    row_indices[-1],
                )
            )
            y_points_corners_outline = y_points[duplicating_indices]
            parameters_corners = parameters[duplicating_indices]
            # Compute the dy values at the corners
            dy_values = np.diff(
                y_points[unique_indices].reshape(y_npoints, x_npoints), axis=0
            )
            dy_corner_values = np.repeat(dy_values, 2, axis=0).ravel()
            y_shift_values = dy_corner_values * parameters_corners
            # y-shift should be negative for upper edges of tiles
            y_shift_values = y_shift_values.reshape(-1, x_npoints)
            y_shift_values[1::2, :] *= -1.0
            y_shift_values = y_shift_values.ravel()

            y_points_tile_corners = y_points_corners_outline + y_shift_values

            # Interleave in x-direction
            new_y_points_both = []
            for y_points_i in [
                y_points_corners_outline,
                y_points_tile_corners,
            ]:
                new_y_points = np.vstack(
                    [
                        interleave_with_means(points)
                        for points in np.split(y_points_i, 2 * self._tiling[1])
                    ]
                )
                new_y_points = np.vstack(
                    [
                        interleave_with_means(points)
                        for points in np.split(new_y_points, self._tiling[1])
                    ]
                )
                new_y_points_both.append(new_y_points)
            return new_y_points_both

        def grid_points_to_tile_points(array, rows_to_not_build_grid=[]):
            """Turn an array of points and return a list of the points in tiles

            Parameters
            -----------------
            array: np.ndarray
                Array of points, must be 2-dimensional
            rows_to_not_build_grid: list<int>
                At these rows there should not be a build a tile, meaning that the
                corners of the tiles will not be returned for those rows
            """
            assert array.ndim == 2, "Array should be 2-dimensional"
            n_rows, n_cols = array.shape
            mask = np.array([0, 1, n_cols, n_cols + 1])
            # Remove the last row
            row_indices = np.arange((n_rows - 1) * n_cols)
            # Remove the right column
            row_indices = row_indices[(row_indices + 1) % array.shape[1] != 0]
            # Remove the rows where not to build a grid
            row_indices = row_indices[
                ~np.isin(row_indices // n_cols, rows_to_not_build_grid)
            ]
            mask = mask + row_indices[:, None]
            # Compute the corner values of every tile
            result = np.split(
                array.ravel()[mask.ravel()],
                (n_rows - 1 - len(rows_to_not_build_grid)) * (n_cols - 1),
            )
            return result

        def grid_points_to_tile_points_twist(array, cols_to_not_build_grid=[]):
            assert array.ndim == 2, "Array should be 2-dimensional"
            n_rows, n_cols = array.shape
            mask = np.array([0, 1, n_cols, n_cols + 1])
            # Remove the last row
            col_indices = np.arange((n_rows - 1) * n_cols)
            # Remove the right column
            col_indices = col_indices[(col_indices + 1) % array.shape[1] != 0]
            # Remove the columns where not to build a grid
            col_indices = col_indices[
                ~np.isin(col_indices % n_cols, cols_to_not_build_grid)
            ]
            mask = mask + col_indices[:, None]
            # Compute the corner values of every tile
            result = np.split(
                array.ravel()[mask.ravel()],
                (n_rows - 1) * (n_cols - 1 - len(cols_to_not_build_grid)),
            )
            return result

        # Compute the twist linkages
        rows_to_not_build_grid = 3 * np.arange(1, y_tiling) - 1
        cols_to_not_build_grid = 3 * np.arange(1, x_tiling) - 1
        all_patches = []
        dummy_tile = sp.helpme.create.box(1, 1, 1)

        for layer_id, twist_type in zip(linkage_layer_ids, twist_ids):
            front_cps_list = []
            back_cps_list = []
            # Go through the front and back layers
            for i in range(2):
                evaluation_points_para = _cartesian_product(
                    [
                        self._parametric_grid_points_single[0],
                        self._parametric_grid_points_single[1],
                        np.array(
                            [
                                self._parametric_grid_points_single[2][
                                    layer_id + i
                                ]
                            ]
                        ),
                    ]
                )
                grid_points = self._macro_spline_initial.evaluate(
                    evaluation_points_para
                )
                parameters = self._parameter_spline_initial.evaluate(
                    evaluation_points_para
                ).ravel()
                # Non-twist
                if (i == 0 and twist_type == 1) or (
                    i == 1 and twist_type == -1
                ):
                    # Parameters don't affect x-points
                    # Interleave in x-direction within rows
                    interleaved_x_points = np.vstack(
                        [
                            interleave_with_means(x_points)
                            for x_points in np.split(
                                grid_points[:, 0], y_tiling + 1
                            )
                        ]
                    )
                    # Interleave in y-direction only within tile
                    new_x_points = np.vstack(
                        [
                            interleave_with_means(
                                interleaved_x_points[i : i + 2, :]
                            )
                            for i in range(y_tiling)
                        ]
                    )
                    # Compute y points
                    _, new_y_points = compute_y_linkage_points(
                        grid_points[:, 1], parameters
                    )

                    interleaved_z_points = np.vstack(
                        [
                            interleave_with_means(z_points)
                            for z_points in np.split(
                                grid_points[:, 2], y_tiling + 1
                            )
                        ]
                    )
                    new_z_points = np.vstack(
                        [
                            interleave_with_means(
                                interleaved_z_points[i : i + 2, :]
                            )
                            for i in range(y_tiling)
                        ]
                    )

                    x_points = grid_points_to_tile_points(
                        new_x_points, rows_to_not_build_grid
                    )
                    y_points = grid_points_to_tile_points(
                        new_y_points, rows_to_not_build_grid
                    )
                    z_points = grid_points_to_tile_points(
                        new_z_points, rows_to_not_build_grid
                    )
                    front_cps_list += [
                        np.column_stack((x, y, z))
                        for x, y, z in zip(x_points, y_points, z_points)
                    ]
                # Twist
                else:
                    # Parameters don't affect y-points
                    y_grid_points = grid_points[:, 1].reshape(-1, x_npoints)
                    # Interleave tile-wise in x-direction
                    interleaved_y_points = np.vstack(
                        [
                            np.hstack(
                                [
                                    interleave_with_means(
                                        row_points[i : i + 2]
                                    )
                                    for i in range(x_tiling)
                                ]
                            )
                            for row_points in y_grid_points
                        ]
                    )
                    # Interleave in y-direction
                    new_y_points = interleave_with_means(interleaved_y_points)
                    # Compute x points - reuse compute_y_linkage_points by changing
                    # input array to function
                    _, new_x_points = compute_x_linkage_points(
                        grid_points[:, 0], parameters
                    )

                    z_grid_points = grid_points[:, 2].reshape(-1, x_npoints)
                    # Interleave tile-wise in x-direction
                    interleaved_z_points = np.vstack(
                        [
                            np.hstack(
                                [
                                    interleave_with_means(
                                        row_points[i : i + 2]
                                    )
                                    for i in range(x_tiling)
                                ]
                            )
                            for row_points in z_grid_points
                        ]
                    )
                    # Interleave in y-direction
                    new_z_points = interleave_with_means(interleaved_z_points)

                    x_points = grid_points_to_tile_points_twist(
                        new_x_points, cols_to_not_build_grid
                    )
                    y_points = grid_points_to_tile_points_twist(
                        new_y_points, cols_to_not_build_grid
                    )
                    z_points = grid_points_to_tile_points_twist(
                        new_z_points, cols_to_not_build_grid
                    )

                if i == 0:
                    front_cps_list += [
                        np.column_stack((x, y, z))
                        for x, y, z in zip(x_points, y_points, z_points)
                    ]
                elif i == 1:
                    back_cps_list += [
                        np.column_stack((x, y, z))
                        for x, y, z in zip(x_points, y_points, z_points)
                    ]

            for front_cps, back_cps in zip(front_cps_list, back_cps_list):
                dummy_tile.cps = np.vstack((front_cps, back_cps))
                all_patches.append(dummy_tile.copy())

        return all_patches

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

    def show_microstructure(self):
        self.multipatch.show(control_points=False, knots=False)


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
            [0.3, 0.1, 0.05, 0.3, 0.4, 0.4, 0.4, 0.4, 0.2, 0.2, 0.2, 0.2]
        ).reshape(-1, 1),
    )

    # Declare identifier function for the inlet and outlet
    def identifier_inlet(points):
        return points[:, 2] < EPS

    def identifier_outlet(points):
        return points[:, 2] > BOX_LENGTH - EPS

    boundary_identifier_dict = {
        identifier_inlet: INLET_BOUNDARY_ID,
        identifier_outlet: OUTLET_BOUNDARY_ID,
    }

    geokernel = SMXKernel(
        box_dimensions=[BOX_HEIGHT, BOX_HEIGHT, BOX_LENGTH],
        tiling=TILING,
        twisting_layers=TWISTING_LAYERS,
        linkage_thickness=LINKAGE_THICKNESS,
        forerun_thickness=FORERUN_AFTERRUN_THICKNESS,
        parameter_spline_initial=parameter_spline_initial,
        macro_spline_initial=macro_spline_initial,
        boundary_identifier_dict=boundary_identifier_dict,
    )

    geokernel.generate_microstructure()

    geokernel.show_microstructure()
