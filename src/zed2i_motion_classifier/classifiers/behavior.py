"""file containing functions for classifying behavior."""

import numpy as np
import core.config as config
from collections import deque


def update_joint_history(person) -> None:
    """Funciton to update joint history dictionary with new keypoints from frame.

    Args:
        person: zed tracked body.
    """
    if person.id not in config.joint_histories:
        config.joint_histories[person.id] = {
            joint: deque(maxlen=config.WINDOW_SIZE) for joint in config.FIDGET_JOINTS
        }
    for joint in config.FIDGET_JOINTS:
        kp = person.keypoint[joint]
        if not np.allclose(kp, [0, 0, 0]):
            config.joint_histories[person.id][joint].append(np.array(kp))


def compute_velocity(positions) -> np.array:
    """Compute velocities between frames for all saved frames for 1 joint.

    Args:
        positions: 3D keypoint values for a joint for all saved frames in joint_history.

    Returns:
        np.array containing all between-frame velocities for 1 joint across saved frames in joint_history.
    """
    velocities = []
    for i in range(1, len(positions)):
        v = np.linalg.norm(positions[i] - positions[i - 1])
        velocities.append(v)
    return np.array(velocities)


def classify_fidgeting(joint_history) -> float:
    """Calculates a fidgeting "score" to determine degree of fidgeting.

    Args:
        joint_history: dictionary of past joint keypoints over config.WINDOW_SIZE.

    Returns:
        float value of the mean velocities of tracked joints over the last saved frames.
    """
    motion_energies = []
    for _, positions in joint_history.items():
        if len(positions) < 2:
            continue
        positions_np = np.array(positions)
        velocities = compute_velocity(positions_np)
        above_thresh = velocities > config.MIN_MOTION_THRESHOLD
        if np.sum(above_thresh) >= config.MIN_ACTIVE_FRAMES:
            motion_energies.append(np.mean(velocities))

    if not motion_energies:
        return 0.0
    return np.mean(motion_energies)


def get_behavior(person) -> str:
    """Function to determine a person's behavior.

    Args:
        person: zed tracked body.

    Returns:
        str describing a person's behavior.
    """
    update_joint_history(person)

    if str(person.action_state) == "MOVING":
        return "moving"

    elif str(person.action_state) == "IDLE":
        if person.id not in config.joint_histories:
            return "still"  # Not enough data yet, assume still

        score = classify_fidgeting(config.joint_histories[person.id])

        if score != 0.0:
            norm_score = np.clip((score - 0.0) / (0.1 - 0.0), 0, 1)
            if norm_score < 0.2:
                return "slightly fidgety"
            elif norm_score < 0.5:
                return "fidgety"
            else:
                return "very fidgety"
        else:
            return "still"

    else:
        return "unknown"
