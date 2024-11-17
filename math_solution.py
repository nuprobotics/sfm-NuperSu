import numpy as np


def triangulation(
        camera_matrix: np.ndarray,
        camera_position1: np.ndarray,
        camera_rotation1: np.ndarray,
        camera_position2: np.ndarray,
        camera_rotation2: np.ndarray,
        image_points1: np.ndarray,
        image_points2: np.ndarray
):
    """
    :param camera_matrix: first and second camera matrix, np.ndarray 3x3
    :param camera_position1: first camera position in world coordinate system, np.ndarray 3x1
    :param camera_rotation1: first camera rotation matrix in world coordinate system, np.ndarray 3x3
    :param camera_position2: second camera position in world coordinate system, np.ndarray 3x1
    :param camera_rotation2: second camera rotation matrix in world coordinate system, np.ndarray 3x3
    :param image_points1: points in the first image, np.ndarray Nx2
    :param image_points2: points in the second image, np.ndarray Nx2
    :return: triangulated points, np.ndarray Nx3
    """

    pos1 = camera_position1.reshape(3, 1)
    pos2 = camera_position2.reshape(3, 1)
    rot1 = camera_rotation1.T
    rot2 = camera_rotation2.T
    trans1 = -rot1 @ pos1
    trans2 = -rot2 @ pos2
    proj_matrix1 = camera_matrix @ np.hstack((rot1, trans1))
    proj_matrix2 = camera_matrix @ np.hstack((rot2, trans2))
    num_points = image_points1.shape[0]
    triangulated_points = np.zeros((num_points, 3))

    for idx in range(num_points):
        x1, y1 = image_points1[idx]
        x2, y2 = image_points2[idx]
        A_matrix = np.zeros((4, 4))
        A_matrix[0] = x1 * proj_matrix1[2, :] - proj_matrix1[0, :]
        A_matrix[1] = y1 * proj_matrix1[2, :] - proj_matrix1[1, :]
        A_matrix[2] = x2 * proj_matrix2[2, :] - proj_matrix2[0, :]
        A_matrix[3] = y2 * proj_matrix2[2, :] - proj_matrix2[1, :]
        _, _, Vh = np.linalg.svd(A_matrix)
        homogeneous_point = Vh[-1]
        euclidean_point = homogeneous_point / homogeneous_point[3]
        triangulated_points[idx] = euclidean_point[:3]

    return triangulated_points
