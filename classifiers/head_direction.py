"""file containing the functions for classifying head direction."""

import numpy as np
import core.config as config
from collections import deque
from scipy.spatial.transform import Rotation as R


def extract_orientation_features(skeleton):
    features = []
    for joint_id in config.POSTURE_JOINTS:
        quat = skeleton.local_orientation_per_joint[joint_id]  # [x, y, z, w]
        rot = R.from_quat(quat)
        # Convert quaternion to Euler angles (radians)
        euler = rot.as_euler("xyz", degrees=True)  # or 'zyx' depending on preference
        features.extend(euler)  # flatten into feature vector
    return np.array(features)


def detect_postural_shift(person_id, skeleton):
    global posture_history

    features = extract_orientation_features(skeleton)

    if person_id not in posture_history:
        posture_history[person_id] = deque(maxlen=config.POSTURE_WINDOW)

    posture_history[person_id].append(features)

    # Compare latest posture to average of earlier
    if len(posture_history[person_id]) < config.POSTURE_WINDOW:
        return False  # Not enough data yet

    baseline = np.mean(list(posture_history[person_id])[:-5], axis=0)
    latest = features
    diff = np.linalg.norm(latest - baseline)

    return diff > config.POSTURE_THRESHOLD


def normalize(v):
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-6 else v


def angle_between(v1, v2):
    """Returns the angle in degrees between vectors 'v1' and 'v2'"""
    v1_u = normalize(v1)
    v2_u = normalize(v2)
    dot_product = np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)
    return np.degrees(np.arccos(dot_product))


def get_head_orientation_with_posture(body):
    kp = body.keypoint

    NOSE = 5
    NECK = 4
    SPINE3 = 3
    LEFT_SHOULDER = 12
    RIGHT_SHOULDER = 13

    if any(
        np.allclose(kp[j], [0, 0, 0])
        for j in [NOSE, NECK, SPINE3, LEFT_SHOULDER, RIGHT_SHOULDER]
    ):
        return "unknown"

    neck = kp[NECK]
    nose = kp[NOSE]
    spine3 = kp[SPINE3]
    left_shoulder = kp[LEFT_SHOULDER]
    right_shoulder = kp[RIGHT_SHOULDER]

    # Vectors
    neck_to_nose = nose - neck
    neck_to_spine = spine3 - neck
    shoulder_axis = right_shoulder - left_shoulder

    # Pitch (up/down)
    pitch_angle = angle_between(neck_to_nose, neck_to_spine)

    # Yaw (left/right) - project onto shoulder plane
    shoulder_axis_norm = normalize(shoulder_axis)
    yaw_proj = np.dot(normalize(neck_to_nose), shoulder_axis_norm)

    # Detect postural shift
    postural_shift = detect_postural_shift(body.id, body)

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
