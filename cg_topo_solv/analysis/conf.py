import numpy as np

def gyration_tensor_shape_descriptors(coords):
    """Compute shape descriptors from the gyration tensor of a 3D point cloud."""
    mass_center = np.mean(coords, axis=0)
    deltas = coords - mass_center

    S_xx = np.mean(deltas[:, 0] * deltas[:, 0])
    S_yy = np.mean(deltas[:, 1] * deltas[:, 1])
    S_zz = np.mean(deltas[:, 2] * deltas[:, 2])
    S_xy = np.mean(deltas[:, 0] * deltas[:, 1])
    S_xz = np.mean(deltas[:, 0] * deltas[:, 2])
    S_yz = np.mean(deltas[:, 1] * deltas[:, 2])

    tensor = np.array([[S_xx, S_xy, S_xz],
                       [S_xy, S_yy, S_yz],
                       [S_xz, S_yz, S_zz]])

    evals = np.linalg.eigvalsh(tensor)
    l1, l2, l3 = np.sort(evals)[::-1]

    I1 = l1 + l2 + l3
    I2 = l1 * l2 + l2 * l3 + l3 * l1

    asphericity = 1 - 3 * (I2 / (I1 ** 2)) if I1 != 0 else 0
    prolateness = (1 / (I1 ** 3)) * (3 * l1 - I1) * (3 * l2 - I1) * (3 * l3 - I1) if I1 != 0 else 0
    acylindricity = (l2 - l3) / I1 if I1 != 0 else 0

    return [S_xx, S_yy, S_zz, S_xy, S_xz, S_yz, I1, asphericity, prolateness, acylindricity]
