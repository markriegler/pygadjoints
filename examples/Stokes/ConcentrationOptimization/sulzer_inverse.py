import numpy as _np
from splinepy.bezier import Bezier as _Bezier
from splinepy.microstructure.tiles.tile_base import TileBase as _TileBase


class SulzerSMXInverse(_TileBase):
    """Inverse of element used for Sulzer SMX static mixers. Using their nomenclature,
    the design parameters are: Nx=2 (number of cross bars), Np=1 (number of parallel
    cross bars)

    Ideas for parameters:
        - Thickness of one crossbar
        - Curvature of one crossbar:
            - just one curve
            - maybe multiple curves (wave-like)
        - If just one curve parameter, may set _n_info_per_eval_point to 2
    """

    _dim = 3
    _para_dim = 3
    _evaluation_points = _np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
        ]
    )
    _n_info_per_eval_point = 1
    _parameter_bounds = [[0.0, 0.5]] * 8
    _parameters_shape = (8, 1)
    _sensitivities_implemented = False
    _inverse_implemented = True

    def create_tile(
        self,
        parameters=None,
        parameter_sensitivities=None,
        closure=None,
        **kwargs,  # noqa ARG002
    ):
        """Create a tile based on the parameters which describe the thickness of one
        crossbar.

        Parameters
        -----------
        parameters: np.ndarray
            The design parameters of the microtile. The parameters are as follows:
                1. Thickness of front crossbar
                2. Thickness of back crossbar
        parameter_sensitivities: np.ndarray
            The sensitivities of the parameters
        closure: str(optional)
            Parametric dimension which needs to be closed. Currently not implemented.

        Returns
        ----------
        microtile_list: list(splines)
        """

        if (
            parameters is not None
            and parameters.shape != self._parameters_shape
        ):
            raise ValueError(
                "Parameters are not in the correct shape. Needs "
                + str(self._parameters_shape)
            )

        # Set parameters to default values if nothing is given
        if parameters is None:
            self._logd("Setting cross bar thickness to default 0.2")
            parameters = 0.2 * _np.ones(self._parameters_shape)

        # TODO: Check if parameters are in allowed range

        self.check_params(parameters)

        if parameter_sensitivities is None:
            n_derivatives = 0
            derivatives = None
        else:
            self.check_param_derivatives(parameter_sensitivities)

            n_derivatives = parameter_sensitivities.shape[2]
            derivatives = []
            raise NotImplementedError(
                "Parameter sensitivities not implemented yet!"
            )

        if closure is not None:
            raise NotImplementedError("Closure not implemented!")

        def create_triangle_yz_cps(v_zero, v_one, th_ne, th_se, th_sw, th_nw):
            # Get crossing points
            (
                crossing_s_x,
                crossing_s_y,
                crossing_w_x,
                crossing_w_y,
                crossing_e_x,
                crossing_e_y,
                crossing_n_x,
                crossing_n_y,
            ) = self._get_crossing_points(th_ne, th_se, th_sw, th_nw)
            # Connecting points at the interface
            connecting_interface_point_s_x = (
                th_sw + (v_one - th_sw - th_se) / 2
            )
            connecting_interface_point_w_y = (
                th_sw + (v_one - th_sw - th_nw) / 2
            )
            connecting_interface_point_e_y = (
                th_se + (v_one - th_se - th_ne) / 2
            )
            connecting_interface_point_n_x = (
                th_nw + (v_one - th_nw - th_ne) / 2
            )

            # Triangle middle points
            triangle_s_middle = [
                (2 * connecting_interface_point_s_x + crossing_s_x) / 3,
                crossing_s_y / 3,
            ]
            triangle_w_middle = [
                crossing_w_x / 3,
                (2 * connecting_interface_point_w_y + crossing_w_y) / 3,
            ]
            triangle_e_middle = [
                (2 * v_one + crossing_e_x) / 3,
                (2 * connecting_interface_point_e_y + crossing_e_y) / 3,
            ]
            triangle_n_middle = [
                (2 * connecting_interface_point_n_x + crossing_n_x) / 3,
                (2 * v_one + crossing_n_y) / 3,
            ]

            # South triangle
            triangle_s_sw = _np.array(
                [
                    [th_sw, v_zero],
                    [connecting_interface_point_s_x, v_zero],
                    [(th_sw + crossing_s_x) / 2, crossing_s_y / 2],
                    triangle_s_middle,
                ]
            )

            triangle_s_se = _np.array(
                [
                    [connecting_interface_point_s_x, v_zero],
                    [v_one - th_se, v_zero],
                    triangle_s_middle,
                    [(crossing_s_x + v_one - th_se) / 2, crossing_s_y / 2],
                ]
            )

            triangle_s_n = _np.array(
                [
                    [(th_sw + crossing_s_x) / 2, crossing_s_y / 2],
                    triangle_s_middle,
                    [crossing_s_x, crossing_s_y],
                    [(crossing_s_x + v_one - th_se) / 2, crossing_s_y / 2],
                ]
            )

            # West triangle
            triangle_w_s = _np.array(
                [
                    [v_zero, th_sw],
                    [crossing_w_x / 2, (th_sw + crossing_w_y) / 2],
                    [v_zero, connecting_interface_point_w_y],
                    triangle_w_middle,
                ]
            )

            triangle_w_n = _np.array(
                [
                    [v_zero, connecting_interface_point_w_y],
                    triangle_w_middle,
                    [v_zero, v_one - th_nw],
                    [crossing_w_x / 2, (v_one - th_nw + crossing_w_y) / 2],
                ]
            )

            triangle_w_e = _np.array(
                [
                    [crossing_w_x / 2, (th_sw + crossing_w_y) / 2],
                    [crossing_w_x, crossing_w_y],
                    triangle_w_middle,
                    [crossing_w_x / 2, (v_one - th_nw + crossing_w_y) / 2],
                ]
            )

            # East triangle
            triangle_e_s = _np.array(
                [
                    [(v_one + crossing_e_x) / 2, (th_se + crossing_e_y) / 2],
                    [v_one, th_se],
                    triangle_e_middle,
                    [v_one, connecting_interface_point_e_y],
                ]
            )

            triangle_e_n = _np.array(
                [
                    triangle_e_middle,
                    [v_one, connecting_interface_point_e_y],
                    [
                        (v_one + crossing_e_x) / 2,
                        (crossing_e_y + v_one - th_ne) / 2,
                    ],
                    [v_one, v_one - th_ne],
                ]
            )

            triangle_e_w = _np.array(
                [
                    [crossing_e_x, crossing_e_y],
                    [(crossing_e_x + v_one) / 2, (crossing_e_y + th_se) / 2],
                    [
                        (v_one + crossing_e_x) / 2,
                        (crossing_e_y + v_one - th_ne) / 2,
                    ],
                    triangle_e_middle,
                ]
            )

            # North triangle
            triangle_n_s = _np.array(
                [
                    [crossing_n_x, crossing_n_y],
                    [
                        (crossing_n_x + v_one - th_ne) / 2,
                        (crossing_n_y + v_one) / 2,
                    ],
                    [(crossing_n_x + th_nw) / 2, (crossing_n_y + v_one) / 2],
                    triangle_n_middle,
                ]
            )

            triangle_n_nw = _np.array(
                [
                    [(crossing_n_x + th_nw) / 2, (crossing_n_y + v_one) / 2],
                    triangle_n_middle,
                    [th_nw, v_one],
                    [connecting_interface_point_n_x, v_one],
                ]
            )

            triangle_n_ne = _np.array(
                [
                    [
                        (crossing_n_x + v_one - th_ne) / 2,
                        (crossing_n_y + v_one) / 2,
                    ],
                    [v_one - th_ne, v_one],
                    triangle_n_middle,
                    [connecting_interface_point_n_x, v_one],
                ]
            )

            return (
                crossing_s_x,
                crossing_s_y,
                crossing_w_x,
                crossing_w_y,
                crossing_e_x,
                crossing_e_y,
                crossing_n_x,
                crossing_n_y,
                connecting_interface_point_s_x,
                connecting_interface_point_w_y,
                connecting_interface_point_e_y,
                connecting_interface_point_n_x,
                triangle_s_middle,
                triangle_w_middle,
                triangle_e_middle,
                triangle_n_middle,
                triangle_s_sw,
                triangle_s_se,
                triangle_s_n,
                triangle_w_s,
                triangle_w_n,
                triangle_w_e,
                triangle_e_s,
                triangle_e_n,
                triangle_e_w,
                triangle_n_s,
                triangle_n_nw,
                triangle_n_ne,
            )

        def create_bar_yz_cps(
            v_zero,
            v_one,
            th_ne,
            th_se,
            th_sw,
            th_nw,
            crossing_s_x,
            crossing_s_y,
            crossing_w_x,
            crossing_w_y,
            crossing_e_x,
            crossing_e_y,
            crossing_n_x,
            crossing_n_y,
        ):
            sw_bar_bottom = _np.array(
                [
                    [th_sw, v_zero],
                    [(th_sw + crossing_s_x) / 2, crossing_s_y / 2],
                    [v_zero, th_sw],
                    [crossing_w_x / 2, (th_sw + crossing_w_y) / 2],
                ]
            )

            sw_bar_top = _np.array(
                [
                    [(th_sw + crossing_s_x) / 2, crossing_s_y / 2],
                    [crossing_s_x, crossing_s_y],
                    [crossing_w_x / 2, (th_sw + crossing_w_y) / 2],
                    [crossing_w_x, crossing_w_y],
                ]
            )

            se_bar_bottom = _np.array(
                [
                    [v_one - th_se, v_zero],
                    [v_one, th_se],
                    [(v_one - th_se + crossing_s_x) / 2, crossing_s_y / 2],
                    [(crossing_e_x + v_one) / 2, (th_se + crossing_e_y) / 2],
                ]
            )

            se_bar_top = _np.array(
                [
                    [(v_one - th_se + crossing_s_x) / 2, crossing_s_y / 2],
                    [(crossing_e_x + v_one) / 2, (th_se + crossing_e_y) / 2],
                    [crossing_s_x, crossing_s_y],
                    [crossing_e_x, crossing_e_y],
                ]
            )

            nw_bar_bottom = _np.array(
                [
                    [crossing_w_x, crossing_w_y],
                    [crossing_n_x, crossing_n_y],
                    [crossing_w_x / 2, (crossing_w_y + v_one - th_nw) / 2],
                    [
                        (th_nw + crossing_n_x) / 2,
                        (v_one + crossing_n_y) / 2,
                    ],
                ]
            )

            nw_bar_top = _np.array(
                [
                    [crossing_w_x / 2, (crossing_w_y + v_one - th_nw) / 2],
                    [
                        (th_nw + crossing_n_x) / 2,
                        (v_one + crossing_n_y) / 2,
                    ],
                    [v_zero, v_one - th_nw],
                    [th_nw, v_one],
                ]
            )

            ne_bar_bottom = _np.array(
                [
                    [crossing_n_x, crossing_n_y],
                    [crossing_e_x, crossing_e_y],
                    [
                        (crossing_n_x + v_one - th_ne) / 2,
                        (crossing_n_y + v_one) / 2,
                    ],
                    [
                        (crossing_e_x + v_one) / 2,
                        (crossing_e_y + v_one - th_ne) / 2,
                    ],
                ]
            )

            ne_bar_top = _np.array(
                [
                    [
                        (crossing_n_x + v_one - th_ne) / 2,
                        (crossing_n_y + v_one) / 2,
                    ],
                    [
                        (crossing_e_x + v_one) / 2,
                        (crossing_e_y + v_one - th_ne) / 2,
                    ],
                    [v_one - th_ne, v_one],
                    [v_one, v_one - th_ne],
                ]
            )

            return (
                sw_bar_bottom,
                sw_bar_top,
                se_bar_bottom,
                se_bar_top,
                nw_bar_top,
                nw_bar_bottom,
                ne_bar_bottom,
                ne_bar_top,
            )

        e = _np.ones((4, 1))
        splines = []
        for i_derivative in range(n_derivatives + 1):
            if i_derivative == 0:
                (
                    th_swf,
                    th_sef,
                    th_nwf,
                    th_nef,
                    th_swb,
                    th_seb,
                    th_nwb,
                    th_neb,
                ) = parameters.flatten()

                v_zero = 0.0
                v_one_half = 0.5
                v_one = 1.0

            else:
                raise NotImplementedError("Derivatives not yet implemented")

            spline_list = []

            # Depth/x control points
            x_cps_front = _np.vstack((v_zero * e, v_one_half * e))
            x_cps_back = _np.vstack((v_one_half * e, v_one * e))

            # Create triangle pieces and auxiliary values
            triangle_pieces_front = create_triangle_yz_cps(
                v_zero, v_one, th_nef, th_sef, th_swf, th_nwf
            )
            triangle_pieces_back = create_triangle_yz_cps(
                v_zero, v_one, th_neb, th_seb, th_swb, th_nwb
            )

            # Go through the triangle pieces and create patches front to back
            for triangle_yz_cps_front, triangle_yz_cps_back in zip(
                triangle_pieces_front[16:], triangle_pieces_back[16:]
            ):
                # Cps are defined in xy-plane, but physical plane is zy-plane
                triangle_yz_cps_front = _np.fliplr(triangle_yz_cps_front)
                triangle_yz_cps_back = _np.fliplr(triangle_yz_cps_back)
                # Yz-control points at the middle yz-plane of the tile
                middle_yz_cps = (
                    triangle_yz_cps_front + triangle_yz_cps_back
                ) * 0.5
                # Go through front and back and create patches
                for x_cps, yz_cps in zip(
                    [x_cps_front, x_cps_back],
                    [
                        _np.vstack((triangle_yz_cps_front, middle_yz_cps)),
                        _np.vstack((middle_yz_cps, triangle_yz_cps_back)),
                    ],
                ):
                    control_points = _np.hstack((x_cps, yz_cps))
                    spline_list.append(
                        _Bezier(
                            degrees=[1, 1, 1], control_points=control_points
                        )
                    )

            # Add patches for the bars
            # Get bar pieces
            bar_pieces_front = create_bar_yz_cps(
                v_zero,
                v_one,
                th_nef,
                th_sef,
                th_swf,
                th_nwf,
                *triangle_pieces_front[:8],
            )
            bar_pieces_back = create_bar_yz_cps(
                v_zero,
                v_one,
                th_neb,
                th_seb,
                th_swb,
                th_nwb,
                *triangle_pieces_back[:8],
            )

            # Define whether to put patches at front or at back. The list corresponds
            # to the return values in create_bar_yz_cps
            is_bar_at_front_list = [
                False,
                False,
                True,
                True,
                True,
                True,
                False,
                False,
            ]

            for bar_yz_front, bar_yz_back, is_bar_at_front in zip(
                bar_pieces_front, bar_pieces_back, is_bar_at_front_list
            ):
                # Control points are defined in xy-plane, but physical points should
                # lie in zy-plane
                bar_yz_front = _np.fliplr(bar_yz_front)
                bar_yz_back = _np.fliplr(bar_yz_back)
                middle_yz_cps = (bar_yz_front + bar_yz_back) * 0.5
                if is_bar_at_front:
                    x_cps = x_cps_front
                    yz_cps = _np.vstack((bar_yz_front, middle_yz_cps))
                else:
                    x_cps = x_cps_back
                    yz_cps = _np.vstack((middle_yz_cps, bar_yz_back))
                control_points = _np.hstack((x_cps, yz_cps))
                spline_list.append(
                    _Bezier(degrees=[1, 1, 1], control_points=control_points)
                )

            if i_derivative == 0:
                splines = spline_list.copy()
            else:
                derivatives.append(spline_list)

        return (splines, derivatives)

    def _get_crossing_points(self, th_ne, th_se, th_sw, th_nw):
        """ "Helper function to compute the crossing points of the bars given the
        thickness parameters

        Parameters
        -------------
        th_ne: float
            Thickness of the bar in the NE-corner
        th_se: float
            Thickness of the bar in the SE-corner
        th_sw: float
            Thickness of the bar in the SW-corner
        th_nw: float
            Thickness of the bar in the NW-corner

        Returns
        --------------
        crossing_s_x, crossing_s_y: float
            Coordinates of bar crossing in the south
        crossing_w_x, crossing_w_y: float
            Coordinates of bar crossing in the west
        crossing_e_x, crossing_e_y: float
            Coordinates of bar crossing in the east
        crossing_n_x, crossing_n_y: float
            Coordinates of bar crossing in the north
        """
        crossing_s_x = (
            -th_ne * th_se * th_sw
            + th_ne * th_sw
            + th_nw * th_se * th_sw
            - th_nw * th_se
            - th_nw * th_sw
            + th_nw
            + th_se
            - 1
        ) / (
            -th_ne * th_se + th_ne - th_nw * th_sw + th_nw + th_se + th_sw - 2
        )
        crossing_s_y = (
            -th_ne * th_nw * th_se
            - th_ne * th_nw * th_sw
            + th_ne * th_nw
            + th_ne * th_se
            + th_ne * th_sw
            - th_ne
            + th_nw * th_se
            + th_nw * th_sw
            - th_nw
            - th_se
            - th_sw
            + 1
        ) / (th_ne * th_se - th_ne + th_nw * th_sw - th_nw - th_se - th_sw + 2)
        crossing_w_x = (
            th_ne * th_nw * th_se
            - th_ne * th_nw
            + th_ne * th_se * th_sw
            - th_ne * th_se
            - th_ne * th_sw
            + th_ne
            - th_nw * th_se
            + th_nw
            - th_se * th_sw
            + th_se
            + th_sw
            - 1
        ) / (
            -th_ne * th_nw + th_ne + th_nw - th_se * th_sw + th_se + th_sw - 2
        )
        crossing_w_y = (
            th_ne * th_nw * th_sw
            - th_ne * th_sw
            - th_nw * th_se * th_sw
            + th_nw * th_se
            - th_nw
            + th_se * th_sw
            - th_se
            + 1
        ) / (th_ne * th_nw - th_ne - th_nw + th_se * th_sw - th_se - th_sw + 2)
        crossing_e_x = (
            -th_ne * th_nw * th_sw
            + th_ne * th_sw
            - th_nw * th_se * th_sw
            + th_nw * th_se
            + th_nw * th_sw
            - 1
        ) / (
            -th_ne * th_nw + th_ne + th_nw - th_se * th_sw + th_se + th_sw - 2
        )
        crossing_e_y = (
            th_ne * th_nw * th_se
            - th_ne * th_se * th_sw
            + th_ne * th_sw
            - th_ne
            - th_nw * th_se
            + th_se * th_sw
            - th_sw
            + 1
        ) / (th_ne * th_nw - th_ne - th_nw + th_se * th_sw - th_se - th_sw + 2)
        crossing_n_x = (
            -th_ne * th_nw * th_se
            + th_ne * th_nw * th_sw
            - th_ne * th_sw
            + th_ne
            + th_nw * th_se
            - th_nw * th_sw
            + th_sw
            - 1
        ) / (
            -th_ne * th_se + th_ne - th_nw * th_sw + th_nw + th_se + th_sw - 2
        )
        crossing_n_y = (
            th_ne * th_se * th_sw
            - th_ne * th_sw
            + th_nw * th_se * th_sw
            - th_nw * th_se
            - th_se * th_sw
            + 1
        ) / (th_ne * th_se - th_ne + th_nw * th_sw - th_nw - th_se - th_sw + 2)

        return (
            crossing_s_x,
            crossing_s_y,
            crossing_w_x,
            crossing_w_y,
            crossing_e_x,
            crossing_e_y,
            crossing_n_x,
            crossing_n_y,
        )

    def compute_linkage_patches(self, parameters):
        """Compute the control points for linkage patch (from non-twist to twist)"""
        (
            th_swf,
            th_sef,
            th_nwf,
            th_nef,
            th_swb,
            th_seb,
            th_nwb,
            th_neb,
        ) = parameters.flatten()

        v_zero = 0.0
        v_one_half = 0.5
        v_one = 1.0

        spline_list = []

        sf_cps = _np.array(
            [
                [v_zero, th_swf, v_zero],
                [v_one_half, (th_swf + th_sef) / 2, v_zero],
                [v_zero, th_swf + (v_one - th_swf - th_nwf) / 2, v_zero],
                [v_one_half, v_one_half, v_zero],
                [th_sef, v_zero, v_one],
                [th_sef + (v_one - th_sef - th_seb) / 2, v_zero, v_one],
                [(th_sef + th_nef) / 2, v_one_half, v_one],
                [v_one_half, v_one_half, v_one],
            ]
        )

        nf_cps = _np.array(
            [
                sf_cps[2, :],
                sf_cps[3, :],
                [v_zero, v_one - th_nwf, v_zero],
                [v_one_half, v_one - (th_nwf + th_nwb) / 2, v_zero],
                sf_cps[6, :],
                sf_cps[7, :],
                [th_nef, v_one, v_one],
                [th_nef + (v_one - th_nef - th_neb) / 2, v_one, v_one],
            ]
        )

        sb_cps = _np.array(
            [
                sf_cps[1, :],
                [v_one, th_swb, v_zero],
                sf_cps[3, :],
                [v_one, th_swb + (v_one - th_swb - th_nwb) / 2, v_zero],
                sf_cps[5, :],
                [v_one - th_seb, v_zero, v_one],
                sf_cps[7, :],
                [v_one - (th_seb + th_neb) / 2, v_one_half, v_one],
            ]
        )

        nb_cps = _np.array(
            [
                sb_cps[2, :],
                sb_cps[3, :],
                nf_cps[3, :],
                [v_one, v_one - th_neb, v_zero],
                sb_cps[6, :],
                sb_cps[7, :],
                nf_cps[7, :],
                [v_one - th_neb, v_one, v_one],
            ]
        )

        for cps in [sf_cps, nf_cps, sb_cps, nb_cps]:
            spline_list.append(_Bezier(degrees=[1, 1, 1], control_points=cps))

        return spline_list

    def compute_connection_points(self, parameters):
        """Compute the necessary control points of the patches of a connection piece.

        Parameters
        -----------
        parameters: np.ndarray
            Parameter array

        Returns
        ----------
        cps_min: list<np.ndarray>
            List of control points which belong to a patch of the connection at the
            inlet
        cps_max: list<np.ndarray>
            List of control points which belong to a patch of the connection at the
            outlet
        """
        (
            th_swf,
            th_sef,
            th_nwf,
            th_nef,
            th_swb,
            th_seb,
            th_nwb,
            th_neb,
        ) = parameters.flatten()

        v_zero = 0.0
        v_one_half = 0.5
        v_one = 1.0

        sf_cps_min = _np.array(
            [
                [v_zero, v_zero, v_zero],
                [v_one_half, v_zero, v_zero],
                [v_zero, v_one_half, v_zero],
                [v_one_half, v_one_half, v_zero],
                [v_zero, th_swf, v_one],
                [v_one_half, (th_swf + th_sef) / 2, v_one],
                [v_zero, th_swf + (v_one - th_swf - th_nwf) / 2, v_one],
                [v_one_half, v_one_half, v_one],
            ]
        )

        nf_cps_min = _np.array(
            [
                sf_cps_min[2, :],
                sf_cps_min[3, :],
                [v_zero, v_one, v_zero],
                [v_one_half, v_one, v_zero],
                sf_cps_min[6, :],
                sf_cps_min[7, :],
                [v_zero, v_one - th_nwf, v_one],
                [v_one_half, v_one - (th_nwf + th_nwb) / 2, v_one],
            ]
        )

        sb_cps_min = _np.array(
            [
                sf_cps_min[1, :],
                [v_one, v_zero, v_zero],
                sf_cps_min[3, :],
                [v_one, v_one_half, v_zero],
                sf_cps_min[5, :],
                [v_one, th_swb, v_one],
                sf_cps_min[7, :],
                [v_one, th_swb + (v_one - th_swb - th_nwb) / 2, v_one],
            ]
        )

        nb_cps_min = _np.array(
            [
                sb_cps_min[2, :],
                sb_cps_min[3, :],
                nf_cps_min[3, :],
                [v_one, v_one, v_zero],
                sb_cps_min[6, :],
                sb_cps_min[7, :],
                nf_cps_min[7, :],
                [v_one, v_one - th_neb, v_one],
            ]
        )

        sf_cps_max = _np.array(
            [
                [v_zero, th_sef, v_zero],
                [v_one_half, (th_sef + th_seb) / 2, v_zero],
                [v_zero, th_sef + (v_one - th_sef - th_nef) / 2, v_zero],
                [v_one_half, v_one_half, v_zero],
                [v_zero, v_zero, v_one],
                [v_one_half, v_zero, v_one],
                [v_zero, v_one_half, v_one],
                [v_one_half, v_one_half, v_one],
            ]
        )

        nf_cps_max = _np.array(
            [
                sf_cps_max[2, :],
                sf_cps_max[3, :],
                [v_zero, v_one - th_nef, v_zero],
                [v_one_half, v_one - (th_nef + th_neb) / 2, v_zero],
                sf_cps_max[6, :],
                sf_cps_max[7, :],
                [v_zero, v_one, v_one],
                [v_one_half, v_one, v_one],
            ]
        )

        sb_cps_max = _np.array(
            [
                sf_cps_max[1, :],
                [v_one, th_seb, v_zero],
                sf_cps_max[3, :],
                [v_one, th_seb + (v_one - th_seb - th_neb) / 2, v_zero],
                sf_cps_max[5, :],
                [v_one, v_zero, v_one],
                sf_cps_max[7, :],
                [v_one, v_one_half, v_one],
            ]
        )

        nb_cps_max = _np.array(
            [
                sb_cps_max[2, :],
                sb_cps_max[3, :],
                nf_cps_max[3, :],
                [v_one, v_one - th_neb, v_zero],
                sb_cps_max[6, :],
                sb_cps_max[7, :],
                nf_cps_max[7, :],
                [v_one, v_one, v_one],
            ]
        )

        patches_min = []
        patches_max = []
        for cps_min in [sf_cps_min, nf_cps_min, sb_cps_min, nb_cps_min]:
            patches_min.append(
                _Bezier(degrees=[1, 1, 1], control_points=cps_min)
            )
        for cps_max in [sf_cps_max, nf_cps_max, sb_cps_max, nb_cps_max]:
            patches_max.append(
                _Bezier(degrees=[1, 1, 1], control_points=cps_max)
            )

        return patches_min, patches_max

    def create_inverse_tile(
        self,
        parameters=None,
        parameter_sensitivities=None,
        closure=None,
        **kwargs,  # noqa ARG002
    ):
        """Create a tile which is the inverse of the original tile"""

        if (
            parameters is not None
            and parameters.shape != self._parameters_shape
        ):
            raise ValueError(
                "Parameters are not in the correct shape. Needs "
                + str(self._parameters_shape)
            )

        # Set parameters to default values if nothing is given
        if parameters is None:
            self._logd("Setting cross bar thickness to default 0.2")
            parameters = 0.2 * _np.ones(self._parameters_shape)

        # TODO: Check if parameters are in allowed range

        self.check_params(parameters)

        if parameter_sensitivities is None:
            n_derivatives = 0
            derivatives = None
        else:
            self.check_param_derivatives(parameter_sensitivities)

            n_derivatives = parameter_sensitivities.shape[2]
            derivatives = []
            raise NotImplementedError(
                "Parameter sensitivities not implemented yet!"
            )

        if closure is not None:
            raise NotImplementedError("Closure not implemented!")

        def determine_yz_cps(v_zero, v_one, th_ne, th_se, th_sw, th_nw):
            # Get crossing points
            (
                crossing_s_x,
                crossing_s_y,
                crossing_w_x,
                crossing_w_y,
                crossing_e_x,
                crossing_e_y,
                crossing_n_x,
                crossing_n_y,
            ) = self._get_crossing_points(th_ne, th_se, th_sw, th_nw)
            # Connecting points at the interface
            (th_sw + (v_one - th_sw - th_se) / 2)
            (th_sw + (v_one - th_sw - th_nw) / 2)
            (th_se + (v_one - th_se - th_ne) / 2)
            (th_nw + (v_one - th_nw - th_ne) / 2)

            # SW triangle
            # Define control points
            swt_c_sw = _np.array([v_zero, v_zero])
            swt_e_s = _np.array([th_sw / 2, v_zero])
            swt_c_se = _np.array([th_sw, v_zero])
            swt_e_ne = _np.array([th_sw / 2, th_sw / 2])
            swt_c_n = _np.array([v_zero, th_sw])
            swt_e_w = _np.array([v_zero, th_sw / 2])
            swt_middle = _np.array([th_sw / 3, th_sw / 3])

            triangle_sw_sw = _np.array(
                [swt_c_sw, swt_e_s, swt_e_w, swt_middle]
            )
            triangle_sw_se = _np.array(
                [swt_e_s, swt_c_se, swt_middle, swt_e_ne]
            )
            triangle_sw_n = _np.array([swt_e_w, swt_middle, swt_c_n, swt_e_ne])

            # SW bar
            swb_sw = _np.copy(swt_c_se)
            swb_se = _np.array([crossing_s_x, crossing_s_y])
            swb_nw = _np.copy(swt_c_n)
            swb_ne = _np.array([crossing_w_x, crossing_w_y])
            swb_s = (swb_sw + swb_se) / 2
            swb_e = (swb_se + swb_ne) / 2
            swb_n = (swb_nw + swb_ne) / 2
            swb_w = (swb_sw + swb_nw) / 2
            swb_middle = (swb_e + swb_w) / 2

            bar_sw_sw = _np.array([swb_sw, swb_s, swb_w, swb_middle])
            bar_sw_se = _np.array([swb_s, swb_se, swb_middle, swb_e])
            bar_sw_nw = _np.array([swb_w, swb_middle, swb_nw, swb_n])
            bar_sw_ne = _np.array([swb_middle, swb_e, swb_n, swb_ne])

            # NW triangle
            nwt_c_s = _np.array([v_zero, v_one - th_nw])
            nwt_e_se = _np.array([th_nw / 2, v_one - th_nw / 2])
            nwt_c_e = _np.array([th_nw, v_one])
            nwt_e_n = _np.array([th_nw / 2, v_one])
            nwt_c_nw = _np.array([v_zero, v_one])
            nwt_e_w = _np.array([v_zero, v_one - th_nw / 2])
            nwt_middle = _np.array([th_nw / 3, v_one - th_nw / 3])

            triangle_nw_s = _np.array([nwt_c_s, nwt_e_se, nwt_e_w, nwt_middle])
            triangle_nw_e = _np.array([nwt_e_se, nwt_c_e, nwt_middle, nwt_e_n])
            triangle_nw_nw = _np.array(
                [nwt_e_w, nwt_middle, nwt_c_nw, nwt_e_n]
            )

            # NW bar
            nwb_sw = _np.copy(nwt_c_s)
            nwb_se = _np.array([crossing_w_x, crossing_w_y])
            nwb_nw = _np.copy(nwt_c_e)
            nwb_ne = _np.array([crossing_n_x, crossing_n_y])
            nwb_s = (nwb_sw + nwb_se) / 2
            nwb_e = (nwb_se + nwb_ne) / 2
            nwb_n = (nwb_nw + nwb_ne) / 2
            nwb_w = (nwb_sw + nwb_nw) / 2
            nwb_middle = (nwb_e + nwb_w) / 2

            bar_nw_sw = _np.array([nwb_sw, nwb_s, nwb_w, nwb_middle])
            bar_nw_se = _np.array([nwb_s, nwb_se, nwb_middle, nwb_e])
            bar_nw_nw = _np.array([nwb_w, nwb_middle, nwb_nw, nwb_n])
            bar_nw_ne = _np.array([nwb_middle, nwb_e, nwb_n, nwb_ne])

            # SE triangle
            set_c_sw = _np.array([v_one - th_se, v_zero])
            set_e_s = _np.array([v_one - th_se / 2, v_zero])
            set_c_se = _np.array([v_one, v_zero])
            set_e_e = _np.array([v_one, th_se / 2])
            set_c_n = _np.array([v_one, th_se])
            set_e_nw = _np.array([v_one - th_se / 2, th_se / 2])
            set_middle = _np.array([v_one - th_se / 3, th_se / 3])

            triangle_se_sw = _np.array(
                [set_c_sw, set_e_s, set_e_nw, set_middle]
            )
            triangle_se_se = _np.array(
                [set_e_s, set_c_se, set_middle, set_e_e]
            )
            triangle_se_n = _np.array([set_middle, set_e_e, set_e_nw, set_c_n])

            # SE bar
            seb_sw = _np.array([crossing_s_x, crossing_s_y])
            seb_se = _np.copy(set_c_sw)
            seb_nw = _np.array([crossing_e_x, crossing_e_y])
            seb_ne = _np.copy(set_c_n)
            seb_s = (seb_sw + seb_se) / 2
            seb_e = (seb_se + seb_ne) / 2
            seb_n = (seb_nw + seb_ne) / 2
            seb_w = (seb_sw + seb_nw) / 2
            seb_middle = (seb_e + seb_w) / 2

            bar_se_sw = _np.array([seb_sw, seb_s, seb_w, seb_middle])
            bar_se_se = _np.array([seb_s, seb_se, seb_middle, seb_e])
            bar_se_nw = _np.array([seb_w, seb_middle, seb_nw, seb_n])
            bar_se_ne = _np.array([seb_middle, seb_e, seb_n, seb_ne])

            # NE triangle
            net_c_w = _np.array([v_one - th_ne, v_one])
            net_e_sw = _np.array([v_one - th_ne / 2, v_one - th_ne / 2])
            net_c_s = _np.array([v_one, v_one - th_ne])
            net_e_e = _np.array([v_one, v_one - th_ne / 2])
            net_c_ne = _np.array([v_one, v_one])
            net_e_n = _np.array([v_one - th_ne / 2, v_one])
            net_middle = _np.array([v_one - th_ne / 3, v_one - th_ne / 3])

            triangle_ne_w = _np.array([net_c_w, net_e_sw, net_e_n, net_middle])
            triangle_ne_s = _np.array([net_e_sw, net_c_s, net_middle, net_e_e])
            triangle_ne_ne = _np.array(
                [net_middle, net_e_e, net_e_n, net_c_ne]
            )

            # NE bar
            neb_sw = _np.array([crossing_e_x, crossing_e_y])
            neb_se = _np.copy(net_c_s)
            neb_nw = _np.array([crossing_n_x, crossing_n_y])
            neb_ne = _np.copy(net_c_w)
            neb_s = (neb_sw + neb_se) / 2
            neb_e = (neb_se + neb_ne) / 2
            neb_n = (neb_nw + neb_ne) / 2
            neb_w = (neb_sw + neb_nw) / 2
            neb_middle = (neb_e + neb_w) / 2

            bar_ne_sw = _np.array([neb_sw, neb_s, neb_w, neb_middle])
            bar_ne_se = _np.array([neb_s, neb_se, neb_middle, neb_e])
            bar_ne_nw = _np.array([neb_w, neb_middle, neb_nw, neb_n])
            bar_ne_ne = _np.array([neb_middle, neb_e, neb_n, neb_ne])

            # Middle pieces
            m_sw = _np.array([crossing_s_x, crossing_s_y])
            m_se = _np.array([crossing_e_x, crossing_e_y])
            m_nw = _np.array([crossing_w_x, crossing_w_y])
            m_ne = _np.array([crossing_n_x, crossing_n_y])
            m_s = (m_sw + m_se) / 2
            m_e = (m_se + m_ne) / 2
            m_n = (m_nw + m_ne) / 2
            m_w = (m_sw + m_nw) / 2
            m_middle = (m_e + m_w) / 2

            bar_m_sw = _np.array([m_sw, m_s, m_w, m_middle])
            bar_m_se = _np.array([m_s, m_se, m_middle, m_e])
            bar_m_nw = _np.array([m_w, m_middle, m_nw, m_n])
            bar_m_ne = _np.array([m_middle, m_e, m_n, m_ne])

            triangle_cps_list = [
                triangle_sw_sw,
                triangle_sw_se,
                triangle_sw_n,
                triangle_nw_s,
                triangle_nw_e,
                triangle_nw_nw,
                triangle_se_sw,
                triangle_se_se,
                triangle_se_n,
                triangle_ne_w,
                triangle_ne_s,
                triangle_ne_ne,
            ]

            bar_cps_list = [
                bar_sw_sw,
                bar_sw_se,
                bar_sw_nw,
                bar_sw_ne,
                bar_nw_sw,
                bar_nw_se,
                bar_nw_nw,
                bar_nw_ne,
                bar_se_sw,
                bar_se_se,
                bar_se_nw,
                bar_se_ne,
                bar_ne_sw,
                bar_ne_se,
                bar_ne_nw,
                bar_ne_ne,
                bar_m_sw,
                bar_m_se,
                bar_m_nw,
                bar_m_ne,
            ]

            return triangle_cps_list, bar_cps_list

        e = _np.ones((4, 1))
        splines = []
        for i_derivative in range(n_derivatives + 1):
            if i_derivative == 0:
                (
                    th_swf,
                    th_sef,
                    th_nwf,
                    th_nef,
                    th_swb,
                    th_seb,
                    th_nwb,
                    th_neb,
                ) = parameters.flatten()

                v_zero = 0.0
                v_one_half = 0.5
                v_one = 1.0

            else:
                raise NotImplementedError("Derivatives not yet implemented")

            spline_list = []

            # Depth/x control points
            x_cps_front = _np.vstack((v_zero * e, v_one_half * e))
            x_cps_back = _np.vstack((v_one_half * e, v_one * e))

            triangle_cps_front_list, bar_cps_front_list = determine_yz_cps(
                v_zero, v_one, th_nef, th_sef, th_swf, th_nwf
            )
            triangle_cps_back_list, bar_cps_back_list = determine_yz_cps(
                v_zero, v_one, th_neb, th_seb, th_swb, th_nwb
            )

            # Go through the triangle pieces and create patches front to back
            # Also include the middle pieces which go front to back too
            for triangle_yz_cps_front, triangle_yz_cps_back in zip(
                triangle_cps_front_list + bar_cps_front_list[-4:],
                triangle_cps_back_list + bar_cps_back_list[-4:],
            ):
                # Cps are defined in xy-plane, but physical plane is zy-plane
                triangle_yz_cps_front = _np.fliplr(triangle_yz_cps_front)
                triangle_yz_cps_back = _np.fliplr(triangle_yz_cps_back)
                # Yz-control points at the middle yz-plane of the tile
                middle_yz_cps = (
                    triangle_yz_cps_front + triangle_yz_cps_back
                ) * 0.5
                # Go through front and back and create patches
                for x_cps, yz_cps in zip(
                    [x_cps_front, x_cps_back],
                    [
                        _np.vstack((triangle_yz_cps_front, middle_yz_cps)),
                        _np.vstack((middle_yz_cps, triangle_yz_cps_back)),
                    ],
                ):
                    control_points = _np.hstack((x_cps, yz_cps))
                    spline_list.append(
                        _Bezier(
                            degrees=[1, 1, 1], control_points=control_points
                        )
                    )

            # Define whether to put patches at front or at back. The list corresponds
            # to the return values in create_bar_yz_cps
            is_bar_at_front_list = _np.repeat(
                _np.array([True, False, False, True]), 4
            )

            for bar_yz_front, bar_yz_back, is_bar_at_front in zip(
                bar_cps_front_list[:-4],
                bar_cps_back_list[:-4],
                is_bar_at_front_list,
            ):
                # Control points are defined in xy-plane, but physical points should
                # lie in zy-plane
                bar_yz_front = _np.fliplr(bar_yz_front)
                bar_yz_back = _np.fliplr(bar_yz_back)
                middle_yz_cps = (bar_yz_front + bar_yz_back) * 0.5
                if is_bar_at_front:
                    x_cps = x_cps_front
                    yz_cps = _np.vstack((bar_yz_front, middle_yz_cps))
                else:
                    x_cps = x_cps_back
                    yz_cps = _np.vstack((middle_yz_cps, bar_yz_back))
                control_points = _np.hstack((x_cps, yz_cps))
                spline_list.append(
                    _Bezier(degrees=[1, 1, 1], control_points=control_points)
                )

            if i_derivative == 0:
                splines = spline_list.copy()
            else:
                derivatives.append(spline_list)

        return (splines, derivatives)

    def compute_inverse_connection_points(self, parameters):
        """Compute the necessary control points of the patches of a connection piece.

        Parameters
        -----------
        parameters: np.ndarray
            Parameter array

        Returns
        ----------
        patches_min: list<sp.Bezier>
            List of Bezier patches which belong to the connection at the inlet
        patches_max: list<sp.Bezier>
            List of Bezier patches which belong to the connection at the outlet
        """
        (
            th_swf,
            th_sef,
            th_nwf,
            th_nef,
            th_swb,
            th_seb,
            th_nwb,
            th_neb,
        ) = parameters.flatten()

        v_zero = 0.0
        v_one_half = 0.5
        v_one = 1.0

        # Front: x = 0; south: y = 0
        # South front
        sf_cps_min_w = _np.array(
            [
                [v_zero, v_zero, v_zero],
                [v_one_half, v_zero, v_zero],
                [v_zero, th_swf / 2, v_one / 2],
                [v_one_half, (th_swf + th_swb) / 4, v_one / 2],
                [v_zero, v_zero, v_one / 2],
                [v_one_half, v_zero, v_one / 2],
                [v_zero, th_swf / 3, 2 / 3 * v_one],
                [v_one_half, (th_swf + th_swb) / 6, 2 / 3 * v_one],
            ]
        )

        sf_cps_min_se = _np.array(
            [
                sf_cps_min_w[4, :],
                sf_cps_min_w[5, :],
                [v_zero, v_zero, v_one],
                [v_one_half, v_zero, v_one],
                sf_cps_min_w[6, :],
                sf_cps_min_w[7, :],
                [v_zero, th_swf / 2, v_one],
                [v_one_half, (th_swf + th_swb) / 4, v_one],
            ]
        )

        sf_cps_min_ne = _np.array(
            [
                sf_cps_min_w[6, :],
                sf_cps_min_w[7, :],
                sf_cps_min_se[6, :],
                sf_cps_min_se[7, :],
                sf_cps_min_w[2, :],
                sf_cps_min_w[3, :],
                [v_zero, th_swf, v_one],
                [v_one_half, (th_swf + th_swb) / 2, v_one],
            ]
        )

        # South back
        sb_cps_min_w = _np.array(
            [
                sf_cps_min_w[1, :],
                [v_one, v_zero, v_zero],
                sf_cps_min_w[3, :],
                [v_one, th_swb / 2, v_one / 2],
                sf_cps_min_w[5, :],
                [v_one, v_zero, v_one / 2],
                sf_cps_min_w[7, :],
                [v_one, th_swb / 3, 2 / 3 * v_one],
            ]
        )

        sb_cps_min_se = _np.array(
            [
                sb_cps_min_w[4, :],
                sb_cps_min_w[5, :],
                sf_cps_min_se[3, :],
                [v_one, v_zero, v_one],
                sb_cps_min_w[6, :],
                sb_cps_min_w[7, :],
                sf_cps_min_se[7, :],
                [v_one, th_swb / 2, v_one],
            ]
        )

        sb_cps_min_ne = _np.array(
            [
                sb_cps_min_w[6, :],
                sb_cps_min_w[7, :],
                sb_cps_min_se[6, :],
                sb_cps_min_se[7, :],
                sb_cps_min_w[2, :],
                sb_cps_min_w[3, :],
                sf_cps_min_ne[7, :],
                [v_one, th_swb, v_one],
            ]
        )

        # North front
        nf_cps_min_w = _np.array(
            [
                [v_zero, v_one, v_zero],
                [v_one_half, v_one, v_zero],
                [v_zero, v_one - th_nwf / 2, v_one / 2],
                [v_one_half, v_one - (th_nwf + th_nwb) / 4, v_one / 2],
                [v_zero, v_one, v_one / 2],
                [v_one_half, v_one, v_one / 2],
                [v_zero, v_one - th_nwf / 3, 2 / 3 * v_one],
                [v_one_half, v_one - (th_nwf + th_nwb) / 6, 2 / 3 * v_one],
            ]
        )

        nf_cps_min_se = _np.array(
            [
                nf_cps_min_w[2, :],
                nf_cps_min_w[3, :],
                [v_zero, v_one - th_nwf, v_one],
                [v_one_half, v_one - (th_nwf + th_nwb) / 2, v_one],
                nf_cps_min_w[6, :],
                nf_cps_min_w[7, :],
                [v_zero, v_one - th_nwf / 2, v_one],
                [v_one_half, v_one - (th_nwf + th_nwb) / 4, v_one],
            ]
        )

        nf_cps_min_ne = _np.array(
            [
                nf_cps_min_se[4, :],
                nf_cps_min_se[5, :],
                nf_cps_min_se[6, :],
                nf_cps_min_se[7, :],
                nf_cps_min_w[4, :],
                nf_cps_min_w[5, :],
                [v_zero, v_one, v_one],
                [v_one_half, v_one, v_one],
            ]
        )

        # North back
        nb_cps_min_w = _np.array(
            [
                nf_cps_min_w[1, :],
                [v_one, v_one, v_zero],
                nf_cps_min_w[3, :],
                [v_one, v_one - th_nwb / 2, v_one / 2],
                nf_cps_min_w[5, :],
                [v_one, v_one, v_one / 2],
                nf_cps_min_w[7, :],
                [v_one, v_one - th_nwb / 3, 2 / 3 * v_one],
            ]
        )

        nb_cps_min_se = _np.array(
            [
                nb_cps_min_w[2, :],
                nb_cps_min_w[3, :],
                nf_cps_min_se[3, :],
                [v_one, v_one - th_nwb, v_one],
                nb_cps_min_w[6, :],
                nb_cps_min_w[7, :],
                nf_cps_min_se[7, :],
                [v_one, v_one - th_nwb / 2, v_one],
            ]
        )

        nb_cps_min_ne = _np.array(
            [
                nb_cps_min_w[6, :],
                nb_cps_min_w[7, :],
                nb_cps_min_se[6, :],
                nb_cps_min_se[7, :],
                nb_cps_min_w[4, :],
                nb_cps_min_w[5, :],
                nf_cps_min_ne[7, :],
                [v_one, v_one, v_one],
            ]
        )

        # Max definitions: ChatGPT generated from above definitions
        # South front
        sf_cps_max_e = _np.array(
            [
                [v_zero, v_zero, v_one],
                [v_one_half, v_zero, v_one],
                [v_zero, th_sef / 2, v_one / 2],
                [v_one_half, (th_sef + th_seb) / 4, v_one / 2],
                [v_zero, v_zero, v_one / 2],
                [v_one_half, v_zero, v_one / 2],
                [v_zero, th_sef / 3, v_one / 3],
                [v_one_half, (th_sef + th_seb) / 6, v_one / 3],
            ]
        )

        sf_cps_max_sw = _np.array(
            [
                sf_cps_max_e[4, :],
                sf_cps_max_e[5, :],
                [v_zero, v_zero, v_zero],
                [v_one_half, v_zero, v_zero],
                sf_cps_max_e[6, :],
                sf_cps_max_e[7, :],
                [v_zero, th_sef / 2, v_zero],
                [v_one_half, (th_sef + th_seb) / 4, v_zero],
            ]
        )

        sf_cps_max_nw = _np.array(
            [
                sf_cps_max_e[6, :],
                sf_cps_max_e[7, :],
                sf_cps_max_sw[6, :],
                sf_cps_max_sw[7, :],
                sf_cps_max_e[2, :],
                sf_cps_max_e[3, :],
                [v_zero, th_sef, v_zero],
                [v_one_half, (th_sef + th_seb) / 2, v_zero],
            ]
        )

        # South back
        sb_cps_max_e = _np.array(
            [
                sf_cps_max_e[1, :],
                [v_one, v_zero, v_one],
                sf_cps_max_e[3, :],
                [v_one, th_seb / 2, v_one / 2],
                sf_cps_max_e[5, :],
                [v_one, v_zero, v_one / 2],
                sf_cps_max_e[7, :],
                [v_one, th_seb / 3, v_one / 3],
            ]
        )

        sb_cps_max_sw = _np.array(
            [
                sb_cps_max_e[4, :],
                sb_cps_max_e[5, :],
                sf_cps_max_sw[3, :],
                [v_one, v_zero, v_zero],
                sb_cps_max_e[6, :],
                sb_cps_max_e[7, :],
                sf_cps_max_sw[7, :],
                [v_one, th_seb / 2, v_zero],
            ]
        )

        sb_cps_max_nw = _np.array(
            [
                sb_cps_max_e[6, :],
                sb_cps_max_e[7, :],
                sb_cps_max_sw[6, :],
                sb_cps_max_sw[7, :],
                sb_cps_max_e[2, :],
                sb_cps_max_e[3, :],
                sf_cps_max_nw[7, :],
                [v_one, th_seb, v_zero],
            ]
        )

        # North front
        nf_cps_max_e = _np.array(
            [
                [v_zero, v_one, v_one],
                [v_one_half, v_one, v_one],
                [v_zero, v_one - th_nef / 2, v_one / 2],
                [v_one_half, v_one - (th_nef + th_neb) / 4, v_one / 2],
                [v_zero, v_one, v_one / 2],
                [v_one_half, v_one, v_one / 2],
                [v_zero, v_one - th_nef / 3, v_one / 3],
                [v_one_half, v_one - (th_nef + th_neb) / 6, v_one / 3],
            ]
        )

        nf_cps_max_sw = _np.array(
            [
                nf_cps_max_e[2, :],
                nf_cps_max_e[3, :],
                [v_zero, v_one - th_nef, v_zero],
                [v_one_half, v_one - (th_nef + th_neb) / 2, v_zero],
                nf_cps_max_e[6, :],
                nf_cps_max_e[7, :],
                [v_zero, v_one - th_nef / 2, v_zero],
                [v_one_half, v_one - (th_nef + th_neb) / 4, v_zero],
            ]
        )

        nf_cps_max_nw = _np.array(
            [
                nf_cps_max_sw[4, :],
                nf_cps_max_sw[5, :],
                nf_cps_max_sw[6, :],
                nf_cps_max_sw[7, :],
                nf_cps_max_e[4, :],
                nf_cps_max_e[5, :],
                [v_zero, v_one, v_zero],
                [v_one_half, v_one, v_zero],
            ]
        )

        # North back
        nb_cps_max_e = _np.array(
            [
                nf_cps_max_e[1, :],
                [v_one, v_one, v_one],
                nf_cps_max_e[3, :],
                [v_one, v_one - th_neb / 2, v_one / 2],
                nf_cps_max_e[5, :],
                [v_one, v_one, v_one / 2],
                nf_cps_max_e[7, :],
                [v_one, v_one - th_neb / 3, v_one / 3],
            ]
        )

        nb_cps_max_sw = _np.array(
            [
                nb_cps_max_e[2, :],
                nb_cps_max_e[3, :],
                nf_cps_max_sw[3, :],
                [v_one, v_one - th_neb, v_zero],
                nb_cps_max_e[6, :],
                nb_cps_max_e[7, :],
                nf_cps_max_sw[7, :],
                [v_one, v_one - th_neb / 2, v_zero],
            ]
        )

        nb_cps_max_nw = _np.array(
            [
                nb_cps_max_e[6, :],
                nb_cps_max_e[7, :],
                nb_cps_max_sw[6, :],
                nb_cps_max_sw[7, :],
                nb_cps_max_e[4, :],
                nb_cps_max_e[5, :],
                nf_cps_max_nw[7, :],
                [v_one, v_one, v_zero],
            ]
        )

        patches_min = []
        patches_max = []
        for cps_min in [
            sf_cps_min_w,
            sf_cps_min_se,
            sf_cps_min_ne,
            sb_cps_min_w,
            sb_cps_min_se,
            sb_cps_min_ne,
            nf_cps_min_w,
            nf_cps_min_se,
            nf_cps_min_ne,
            nb_cps_min_w,
            nb_cps_min_se,
            nb_cps_min_ne,
        ]:
            patches_min.append(
                _Bezier(degrees=[1, 1, 1], control_points=cps_min)
            )
        for cps_max in [
            sf_cps_max_e,
            sf_cps_max_sw,
            sf_cps_max_nw,
            sb_cps_max_e,
            sb_cps_max_sw,
            sb_cps_max_nw,
            nf_cps_max_e,
            nf_cps_max_sw,
            nf_cps_max_nw,
            nb_cps_max_e,
            nb_cps_max_sw,
            nb_cps_max_nw,
        ]:
            patches_max.append(
                _Bezier(degrees=[1, 1, 1], control_points=cps_max)
            )

        return patches_min, patches_max

    def compute_inverse_linkage_patches(self, parameters):
        """Compute the necessary control points of the patches of a connection piece.

        Parameters
        -----------
        parameters: np.ndarray
            Parameter array

        Returns
        ----------
        patches_min: list<sp.Bezier>
            List of Bezier patches which belong to the connection at the inlet
        patches_max: list<sp.Bezier>
            List of Bezier patches which belong to the connection at the outlet
        """
        (
            th_swf,
            th_sef,
            th_nwf,
            th_nef,
            th_swb,
            th_seb,
            th_nwb,
            th_neb,
        ) = parameters.flatten()

        v_zero = 0.0
        v_one_half = 0.5
        v_one = 1.0

        spline_list = []

        # Front and back wedges
        # Mainly takes the definition from compute_inverse_connection points and applies
        # the following transformation: change x- and y-coordinate and change the
        # thickness definitions: swf->sef, nwf->seb, swb->nef, nwb->neb
        # South front
        sf_cps_xz_w = _np.array(
            [
                [v_zero, v_zero, v_zero],
                [v_zero, v_one_half, v_zero],
                [th_sef / 2, v_zero, v_one / 2],
                [(th_sef + th_nef) / 4, v_one_half, v_one / 2],
                [v_zero, v_zero, v_one / 2],
                [v_zero, v_one_half, v_one / 2],
                [th_sef / 3, v_zero, 2 / 3 * v_one],
                [(th_sef + th_nef) / 6, v_one_half, 2 / 3 * v_one],
            ]
        )

        sf_cps_xz_se = _np.array(
            [
                sf_cps_xz_w[4, :],
                sf_cps_xz_w[5, :],
                [v_zero, v_zero, v_one],
                [v_zero, v_one_half, v_one],
                sf_cps_xz_w[6, :],
                sf_cps_xz_w[7, :],
                [th_sef / 2, v_zero, v_one],
                [(th_sef + th_nef) / 4, v_one_half, v_one],
            ]
        )

        sf_cps_xz_ne = _np.array(
            [
                sf_cps_xz_w[6, :],
                sf_cps_xz_w[7, :],
                sf_cps_xz_se[6, :],
                sf_cps_xz_se[7, :],
                sf_cps_xz_w[2, :],
                sf_cps_xz_w[3, :],
                [th_sef, v_zero, v_one],
                [(th_sef + th_nef) / 2, v_one_half, v_one],
            ]
        )

        # South back
        sb_cps_xz_w = _np.array(
            [
                sf_cps_xz_w[1, :],
                [v_zero, v_one, v_zero],
                sf_cps_xz_w[3, :],
                [th_nef / 2, v_one, v_one / 2],
                sf_cps_xz_w[5, :],
                [v_zero, v_one, v_one / 2],
                sf_cps_xz_w[7, :],
                [th_nef / 3, v_one, 2 / 3 * v_one],
            ]
        )

        sb_cps_xz_se = _np.array(
            [
                sb_cps_xz_w[4, :],
                sb_cps_xz_w[5, :],
                sf_cps_xz_se[3, :],
                [v_zero, v_one, v_one],
                sb_cps_xz_w[6, :],
                sb_cps_xz_w[7, :],
                sf_cps_xz_se[7, :],
                [th_nef / 2, v_one, v_one],
            ]
        )

        sb_cps_xz_ne = _np.array(
            [
                sb_cps_xz_w[6, :],
                sb_cps_xz_w[7, :],
                sb_cps_xz_se[6, :],
                sb_cps_xz_se[7, :],
                sb_cps_xz_w[2, :],
                sb_cps_xz_w[3, :],
                sf_cps_xz_ne[7, :],
                [th_nef, v_one, v_one],
            ]
        )

        # North front
        nf_cps_xz_w = _np.array(
            [
                [v_one, v_zero, v_zero],
                [v_one, v_one_half, v_zero],
                [v_one - th_seb / 2, v_zero, v_one / 2],
                [v_one - (th_seb + th_neb) / 4, v_one_half, v_one / 2],
                [v_one, v_zero, v_one / 2],
                [v_one, v_one_half, v_one / 2],
                [v_one - th_seb / 3, v_zero, 2 / 3 * v_one],
                [v_one - (th_seb + th_neb) / 6, v_one_half, 2 / 3 * v_one],
            ]
        )

        nf_cps_xz_se = _np.array(
            [
                nf_cps_xz_w[2, :],
                nf_cps_xz_w[3, :],
                [v_one - th_seb, v_zero, v_one],
                [v_one - (th_seb + th_neb) / 2, v_one_half, v_one],
                nf_cps_xz_w[6, :],
                nf_cps_xz_w[7, :],
                [v_one - th_seb / 2, v_zero, v_one],
                [v_one - (th_seb + th_neb) / 4, v_one_half, v_one],
            ]
        )

        nf_cps_xz_ne = _np.array(
            [
                nf_cps_xz_se[4, :],
                nf_cps_xz_se[5, :],
                nf_cps_xz_se[6, :],
                nf_cps_xz_se[7, :],
                nf_cps_xz_w[4, :],
                nf_cps_xz_w[5, :],
                [v_one, v_zero, v_one],
                [v_one, v_one_half, v_one],
            ]
        )

        # North back
        nb_cps_xz_w = _np.array(
            [
                nf_cps_xz_w[1, :],
                [v_one, v_one, v_zero],
                nf_cps_xz_w[3, :],
                [v_one - th_neb / 2, v_one, v_one / 2],
                nf_cps_xz_w[5, :],
                [v_one, v_one, v_one / 2],
                nf_cps_xz_w[7, :],
                [v_one - th_neb / 3, v_one, 2 / 3 * v_one],
            ]
        )

        nb_cps_xz_se = _np.array(
            [
                nb_cps_xz_w[2, :],
                nb_cps_xz_w[3, :],
                nf_cps_xz_se[3, :],
                [v_one - th_neb, v_one, v_one],
                nb_cps_xz_w[6, :],
                nb_cps_xz_w[7, :],
                nf_cps_xz_se[7, :],
                [v_one - th_neb / 2, v_one, v_one],
            ]
        )

        nb_cps_xz_ne = _np.array(
            [
                nb_cps_xz_w[6, :],
                nb_cps_xz_w[7, :],
                nb_cps_xz_se[6, :],
                nb_cps_xz_se[7, :],
                nb_cps_xz_w[4, :],
                nb_cps_xz_w[5, :],
                nf_cps_xz_ne[7, :],
                [v_one, v_one, v_one],
            ]
        )

        # Top and bottom row wedges: there is a slimming of the x-coordinates into the
        # z-direction
        # South front
        sf_cps_yz_w = _np.array(
            [
                [th_sef * v_one, v_zero, v_one],
                [v_one_half, v_zero, v_one],
                [th_sef * v_one / 2, th_swf / 2, v_one / 2],
                [v_one_half, (th_swf + th_swb) / 4, v_one / 2],
                [th_sef * v_one / 2, v_zero, v_one / 2],
                [v_one_half, v_zero, v_one / 2],
                [th_sef * v_one / 3, th_swf / 3, v_one / 3],
                [v_one_half, (th_swf + th_swb) / 6, v_one / 3],
            ]
        )

        sf_cps_yz_se = _np.array(
            [
                sf_cps_yz_w[4, :],
                sf_cps_yz_w[5, :],
                [v_zero, v_zero, v_zero],
                [v_one_half, v_zero, v_zero],
                sf_cps_yz_w[6, :],
                sf_cps_yz_w[7, :],
                [v_zero, th_swf / 2, v_zero],
                [v_one_half, (th_swf + th_swb) / 4, v_zero],
            ]
        )

        sf_cps_yz_ne = _np.array(
            [
                sf_cps_yz_w[6, :],
                sf_cps_yz_w[7, :],
                sf_cps_yz_se[6, :],
                sf_cps_yz_se[7, :],
                sf_cps_yz_w[2, :],
                sf_cps_yz_w[3, :],
                [v_zero, th_swf, v_zero],
                [v_one_half, (th_swf + th_swb) / 2, v_zero],
            ]
        )

        # South back
        sb_cps_yz_w = _np.array(
            [
                sf_cps_yz_w[1, :],
                [v_one - th_seb * v_one, v_zero, v_one],
                sf_cps_yz_w[3, :],
                [v_one - th_seb * v_one / 2, th_swb / 2, v_one / 2],
                sf_cps_yz_w[5, :],
                [v_one - th_seb * v_one / 2, v_zero, v_one / 2],
                sf_cps_yz_w[7, :],
                [v_one - th_seb * v_one / 3, th_swb / 3, v_one / 3],
            ]
        )

        sb_cps_yz_se = _np.array(
            [
                sb_cps_yz_w[4, :],
                sb_cps_yz_w[5, :],
                sf_cps_yz_se[3, :],
                [v_one, v_zero, v_zero],
                sb_cps_yz_w[6, :],
                sb_cps_yz_w[7, :],
                sf_cps_yz_se[7, :],
                [v_one, th_swb / 2, v_zero],
            ]
        )

        sb_cps_yz_ne = _np.array(
            [
                sb_cps_yz_w[6, :],
                sb_cps_yz_w[7, :],
                sb_cps_yz_se[6, :],
                sb_cps_yz_se[7, :],
                sb_cps_yz_w[2, :],
                sb_cps_yz_w[3, :],
                sf_cps_yz_ne[7, :],
                [v_one, th_swb, v_zero],
            ]
        )

        # North front
        nf_cps_yz_w = _np.array(
            [
                [th_nef * v_one, v_one, v_one],
                [v_one_half, v_one, v_one],
                [th_nef * v_one / 2, v_one - th_nwf / 2, v_one / 2],
                [v_one_half, v_one - (th_nwf + th_nwb) / 4, v_one / 2],
                [th_nef * v_one / 2, v_one, v_one / 2],
                [v_one_half, v_one, v_one / 2],
                [th_nef * v_one / 3, v_one - th_nwf / 3, v_one / 3],
                [v_one_half, v_one - (th_nwf + th_nwb) / 6, v_one / 3],
            ]
        )

        nf_cps_yz_se = _np.array(
            [
                nf_cps_yz_w[2, :],
                nf_cps_yz_w[3, :],
                [v_zero, v_one - th_nwf, v_zero],
                [v_one_half, v_one - (th_nwf + th_nwb) / 2, v_zero],
                nf_cps_yz_w[6, :],
                nf_cps_yz_w[7, :],
                [v_zero, v_one - th_nwf / 2, v_zero],
                [v_one_half, v_one - (th_nwf + th_nwb) / 4, v_zero],
            ]
        )

        nf_cps_yz_ne = _np.array(
            [
                nf_cps_yz_se[4, :],
                nf_cps_yz_se[5, :],
                nf_cps_yz_se[6, :],
                nf_cps_yz_se[7, :],
                nf_cps_yz_w[4, :],
                nf_cps_yz_w[5, :],
                [v_zero, v_one, v_zero],
                [v_one_half, v_one, v_zero],
            ]
        )

        # North back
        nb_cps_yz_w = _np.array(
            [
                nf_cps_yz_w[1, :],
                [v_one - th_neb * v_one, v_one, v_one],
                nf_cps_yz_w[3, :],
                [v_one - th_neb * v_one / 2, v_one - th_nwb / 2, v_one / 2],
                nf_cps_yz_w[5, :],
                [v_one - th_neb * v_one / 2, v_one, v_one / 2],
                nf_cps_yz_w[7, :],
                [v_one - th_neb * v_one / 3, v_one - th_nwb / 3, v_one / 3],
            ]
        )

        nb_cps_yz_se = _np.array(
            [
                nb_cps_yz_w[2, :],
                nb_cps_yz_w[3, :],
                nf_cps_yz_se[3, :],
                [v_one, v_one - th_nwb, v_zero],
                nb_cps_yz_w[6, :],
                nb_cps_yz_w[7, :],
                nf_cps_yz_se[7, :],
                [v_one, v_one - th_nwb / 2, v_zero],
            ]
        )

        nb_cps_yz_ne = _np.array(
            [
                nb_cps_yz_w[6, :],
                nb_cps_yz_w[7, :],
                nb_cps_yz_se[6, :],
                nb_cps_yz_se[7, :],
                nb_cps_yz_w[4, :],
                nb_cps_yz_w[5, :],
                nf_cps_yz_ne[7, :],
                [v_one, v_one, v_zero],
            ]
        )

        for cps in [
            sf_cps_xz_w,
            sf_cps_xz_se,
            sf_cps_xz_ne,
            sb_cps_xz_w,
            sb_cps_xz_se,
            sb_cps_xz_ne,
            nf_cps_xz_w,
            nf_cps_xz_se,
            nf_cps_xz_ne,
            nb_cps_xz_w,
            nb_cps_xz_se,
            nb_cps_xz_ne,
            sf_cps_yz_w,
            sf_cps_yz_se,
            sf_cps_yz_ne,
            sb_cps_yz_w,
            sb_cps_yz_se,
            sb_cps_yz_ne,
            nf_cps_yz_w,
            nf_cps_yz_se,
            nf_cps_yz_ne,
            nb_cps_yz_w,
            nb_cps_yz_se,
            nb_cps_yz_ne,
        ]:
            spline_list.append(_Bezier(degrees=[1, 1, 1], control_points=cps))

        return spline_list
