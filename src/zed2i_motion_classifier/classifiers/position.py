"""file containing functions for classifying position."""

import numpy as np


def compute_knee_angle(hip, knee, ankle) -> float:
    """This function computes the angle of knee flexion in teh sagittal plane.

    Args:
        hip: 3d keypoint values of the hip joint in 1 frame.
        knee: 3d keypoint values of the knee joint in 1 frame.
        ankle: 3d keypoint values of the ankles joint in 1 frame.

    Returns:
        angle (in degrees) of knee flexion.
    """
    thigh = hip - knee
    shin = ankle - knee
    if np.linalg.norm(thigh) == 0 or np.linalg.norm(shin) == 0:
        return 0
    cosine_angle = np.dot(thigh, shin) / (np.linalg.norm(thigh) * np.linalg.norm(shin))
    angle_rad = np.arccos(np.clip(cosine_angle, -1.0, 1.0))

    return np.degrees(angle_rad)


def classify_position(skeleton) -> str:
    """This function classifies position based on knee flexion.

    Args:
        skeleton: zed tracked body.

    Returns:
        str describing position.
    """
    left_hip = skeleton.keypoint[18]
    left_knee = skeleton.keypoint[20]
    left_ankle = skeleton.keypoint[22]

    right_hip = skeleton.keypoint[19]
    right_knee = skeleton.keypoint[21]
    right_ankle = skeleton.keypoint[23]

    def valid(p):
        return not np.allclose(p, [0, 0, 0])

    angles = []
    if valid(left_hip) and valid(left_knee) and valid(left_ankle):
        angles.append(compute_knee_angle(left_hip, left_knee, left_ankle))
    if valid(right_hip) and valid(right_knee) and valid(right_ankle):
        angles.append(compute_knee_angle(right_hip, right_knee, right_ankle))

    if len(angles) == 0:
        return "unknown"

    mean_angle = np.mean(angles)

    if mean_angle < 165:  # you can tune this threshold
        return "sitting"
    else:
        return "standing"
