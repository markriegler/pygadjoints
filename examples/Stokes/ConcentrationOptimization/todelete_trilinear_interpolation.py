import numpy as np
import splinepy as sp

def _trilinear_interp(unit_points, corners):
    """
    Map points from a unit cube to an arbitrary cuboid using trilinear interpolation.
    
    Parameters:
        unit_points: Nx3 array of points with coordinates in [0, 1]
        corners:     8x3 array of cuboid corners in order:
                    [000], [100], [010], [110], [001], [101], [011], [111]
                    
                    where e.g. [100] means x=1,y=0,z=0.
                    
    Returns:
        Nx3 array of mapped points.
    """
    # u = unit_points[:, 0]
    # v = unit_points[:, 1]
    # w = unit_points[:, 2]
    u = unit_points[:, 2]
    v = unit_points[:, 1]
    w = unit_points[:, 0]

    # The following would work on standard labeling
    weights = np.stack([
        (1-u)*(1-v)*(1-w),
        u*(1-v)*(1-w),
        (1-u)*v*(1-w),
        u*v*(1-w),
        (1-u)*(1-v)*w,
        u*(1-v)*w,
        (1-u)*v*w,
        u*v*w
    ], axis=2 if len(unit_points.shape) > 2 else -2) # shape: N x 8
    
    # The SulzerSMXInverse tiles have a weird ordering
    reordering_indices = np.array([0,4, 2,6, 1,5, 3,7])
    weights = weights[reordering_indices]

    # Shape matching: weights N x 8; corners:   8 x 3 --> result N x 3
    mapped_pts = weights @ corners
    
    # Reorder them back
    mapped_pts = mapped_pts[reordering_indices]
    mapped_pts = np.fliplr(mapped_pts)
    
    return mapped_pts

def trilinear_interp_orig(unit_points, corners):
    """
    Map points from a unit cube to an arbitrary cuboid using trilinear interpolation.
    
    Parameters:
        unit_points: Nx3 array of points with coordinates in [0, 1]
        corners:     8x3 array of cuboid corners in order:
                    [000], [100], [010], [110], [001], [101], [011], [111]
                    
                    where e.g. [100] means x=1,y=0,z=0.
                    
    Returns:
        Nx3 array of mapped points.
    """
    u = unit_points[:, 0]
    v = unit_points[:, 1]
    w = unit_points[:, 2]

    # The following would work on standard labeling
    weights = np.stack([
        (1-u)*(1-v)*(1-w),
        u*(1-v)*(1-w),
        (1-u)*v*(1-w),
        u*v*(1-w),
        (1-u)*(1-v)*w,
        u*(1-v)*w,
        (1-u)*v*w,
        u*v*w
    ], axis=2 if len(unit_points.shape) > 2 else -2) # shape: N x 8

    # Shape matching: weights N x 8; corners:   8 x 3 --> result N x 3
    mapped_pts = weights @ corners
    
    return mapped_pts

if __name__ == "__main__":
    cps_para = np.array([
        [0.,         0.,         0.1       ],
        [0.,         0.,         0.42857143],
        [0.,         0.16186191, 0.29240189],
        [0.,         0.10790794, 0.44731554],
        [0.5,        0.,         0.1       ],
        [0.5,        0.,         0.42857143],
        [0.5,        0.16186191, 0.29240189],
        [0.5,        0.10790794, 0.44731554]]
    )
    
    # new_indices = np.array([0,4, 2,6, 1,5, 3,7])
    # cps_para = cps_para[new_indices]
    # print(cps_para)
    
    tile_corners = np.array([
        [0.,         0.,         0.        ],
        [0.02,       0.,         0.        ],
        [0.,         0.02,       0.        ],
        [0.02,       0.02,       0.        ],
        [0.,         0.,         0.03333333],
        [0.02,       0.,         0.03333333],
        [0.,         0.02,       0.03333333],
        [0.02,      0.02,       0.03333333]
    ])
    
    # tile_corners = sp.helpme.create.box(3,2,1).cps

    patch_parametric = sp.Bezier(degrees=[1,1,1], control_points=cps_para)
    
    # patch_parametric.show()
    
    # # new_patch.cps = trilinear_interp_orig(sp.helpme.create.box(1,1,1).cps, tile_corners)
    
    # cps_transformed = trilinear_interp_orig(cps_para, tile_corners)
    from scipy.interpolate import RegularGridInterpolator, LinearNDInterpolator
    e = np.array([0,1])
    E = sp.utils.data.cartesian_product([e,e,e])
    interp = LinearNDInterpolator(E, tile_corners)
    cps_transformed = interp(cps_para)
    
    new_patch = sp.Bezier(degrees=[1,1,1], control_points=cps_transformed)
    
    tile_outline = sp.Bezier(degrees=[1,1,1], control_points=tile_corners)
    
    new_patch_shifted = new_patch.copy()
    new_patch.cps[:,0] += tile_outline.cps[:,0].max()
    
    sp.show(
        ["Patch (parametric)", patch_parametric],
        ["Tile corners", tile_outline],
        ["New patch", new_patch],
        ["Combined", sp.Multipatch([tile_outline, new_patch])]
    )