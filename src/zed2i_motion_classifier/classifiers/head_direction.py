"""file containing the functions for classifying head direction."""

import numpy as np
import core.config as config
from collections import deque
from typing import List, Union
from scipy.spatial.transform import Rotation as R


def extract_orientation_features(skeleton) -> np.ndarray:
    """This function calculates the quaternion of each joint.

    Args:
        skeleton: zed tracked body.

    Returns:
        an array of quaternions of all joints for the current frame.
    """
    features = []
    for joint_id in config.POSTURE_JOINTS:
        quat = skeleton.local_orientation_per_joint[joint_id]  # [x, y, z, w]
        rot = R.from_quat(quat)
        # Convert quaternion to Euler angles (radians)
        euler = rot.as_euler("xyz", degrees=True)
        features.extend(euler)  # flatten into feature vector
    return np.array(features)


def detect_postural_shift(person) -> Union[bool, List[bool]]:
    """This function determines if postural shift has occured.

    A sliding window of past postures features is saved in posture_history. This function checks if the differences
    in the current frame's features and the posture history is larger than the posture threshold.

    Args:
        person: zed tracked body.

    Returns:
        single boolean value or list of boolean values.
    """
    features = extract_orientation_features(person)

    if person.id not in config.posture_history:
        config.posture_history[person.id] = deque(maxlen=config.POSTURE_WINDOW)

    config.posture_history[person.id].append(features)

    # Compare latest posture to average of earlier
    if len(config.posture_history[person.id]) < config.POSTURE_WINDOW:
        return False  # Not enough data yet

    baseline = np.mean(list(config.posture_history[person.id])[:-5], axis=0)
    latest = features
    diff = np.linalg.norm(latest - baseline)

    return diff > config.POSTURE_THRESHOLD


def normalize(v):
    """This function calculates the norm to a vector (v).

    Args:
        v: vector to normalize.

    Returns:
        normalized vector (length 1).
    """
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-6 else v


def get_head_orientation_with_posture(body) -> str | bool:
    """Function to classify head orientation.

    This function calculates the directional vetors from the tracked joints. Those vectors are used to calculates the head's pitch and yaw.
    detect_postural_shift is then called in order to offset any changes in head direction with a postural shift and a final head orientation
    classification is made after checking against threshold values.

    Args:
        body: zed tracked body.

    Returns:
        orientation: str containing pitch and/or yaw classifications.
        postural_shift: Boolean value or list of boolean values if postural shift occurred in the frame.
    """
    if any(np.allclose(body.keypoint[j], [0, 0, 0]) for j in [5, 4, 3, 12, 13]):
        return "unknown"

    neck = body.keypoint[4]
    nose = body.keypoint[5]
    spine3 = body.keypoint[3]
    left_shoulder = body.keypoint[12]
    right_shoulder = body.keypoint[13]

    # Vectors
    neck_to_nose = nose - neck
    neck_to_spine = spine3 - neck
    shoulder_axis = right_shoulder - left_shoulder

    # Pitch (up/down)
    neck_to_nose_u = normalize(neck_to_nose)
    neck_to_spine_u = normalize(neck_to_spine)
    dot_product = np.clip(np.dot(neck_to_nose_u, neck_to_spine_u), -1.0, 1.0)
    pitch_angle = np.degrees(np.arccos(dot_product))

    # Yaw (left/right) - project onto shoulder plane
    shoulder_axis_norm = normalize(shoulder_axis)
    yaw_proj = np.dot(normalize(neck_to_nose), shoulder_axis_norm)

    # Detect postural shift
    postural_shift = detect_postural_shift(body)

    # Thresholds
    pitch_up_thresh = 150
    pitch_down_thresh = 130
    yaw_thresh = 0.2

    # Determine pitch (ignore pitch if postural shift detected)
    if postural_shift:
        pitch = "neutral"
    elif pitch_angle > pitch_up_thresh:
        pitch = "up"
    elif pitch_angle < pitch_down_thresh:
        pitch = "down"
    else:
        pitch = "neutral"

    # Determine yaw
    if yaw_proj > yaw_thresh:
        yaw = "right"
    elif yaw_proj < -yaw_thresh:
        yaw = "left"
    else:
        yaw = "forward"

    # Combine result
    if pitch == "neutral":
        orientation = yaw
    elif yaw == "forward":
        orientation = pitch
    else:
        orientation = f"{pitch}-{yaw}"

    return orientation, postural_shift
