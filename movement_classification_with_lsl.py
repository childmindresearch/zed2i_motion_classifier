"""This file takes in live zed feed to  perform movement classification on the target
and stream frame by frame classifications over lsl."""

import cv2
import os
import pyzed.sl as sl
import tracking_viewer as cv_viewer
import numpy as np
import argparse
from collections import deque
from scipy.spatial.transform import Rotation as R
from pylsl import StreamInfo, StreamOutlet
import pathlib as pl
from datetime import datetime


# GLOBAL PARAMETERS AND THRESHOLDS
POSTURE_WINDOW = 30  # frames (1 second @ 30fps)
POSTURE_THRESHOLD = 10  # quaternion difference threshold (tune this)
POSTURE_JOINTS = [0, 1, 2, 3, 4]  # pelvis, spine1, spine2, spine3, neck

WINDOW_SIZE = 15  # longer window = smoother signal
MIN_MOTION_THRESHOLD = 0.01  # higher = only count more deliberate movement
MIN_ACTIVE_FRAMES = 5  # require at least 5 frames with motion > threshold

FIDGET_JOINTS = [16, 17, 32, 33, 34, 35, 36, 37]  # wrists and fingers

posture_history = {}  # person_id: deque of posture feature vectors
joint_histories = {}  # person_id: joint position history
frame_behavior_log = []  # to save classifications: (frame_id, person_id, behavior)


def compute_velocity(positions):
    velocities = []
    for i in range(1, len(positions)):
        v = np.linalg.norm(positions[i] - positions[i - 1])
        velocities.append(v)
    return np.array(velocities)


def classify_fidgeting(joint_history):
    motion_energies = []
    for joint, positions in joint_history.items():
        if len(positions) < 2:
            continue
        positions_np = np.array(positions)
        velocities = compute_velocity(positions_np)
        above_thresh = velocities > MIN_MOTION_THRESHOLD
        if np.sum(above_thresh) >= MIN_ACTIVE_FRAMES:
            motion_energies.append(np.mean(velocities))  # or np.sum(velocities)

    if not motion_energies:
        return 0.0
    return np.mean(motion_energies)


def update_joint_history(person_id, skeleton):
    if person_id not in joint_histories:
        joint_histories[person_id] = {
            joint: deque(maxlen=WINDOW_SIZE) for joint in FIDGET_JOINTS
        }
    for joint in FIDGET_JOINTS:
        kp = skeleton.keypoint[joint]
        if not np.allclose(kp, [0, 0, 0]):
            joint_histories[person_id][joint].append(np.array(kp))


def get_behavior(person):
    pid = person.id
    action = person.action_state
    update_joint_history(pid, person)

    if str(action) == "MOVING":
        return "moving"

    elif str(action) == "IDLE":
        if pid not in joint_histories:
            return "still"  # Not enough data yet, assume still

        score = classify_fidgeting(joint_histories[pid])

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


def compute_knee_angle(hip, knee, ankle):
    # Vector from knee to hip (thigh) and knee to ankle (shin)
    thigh = hip - knee
    shin = ankle - knee
    if np.linalg.norm(thigh) == 0 or np.linalg.norm(shin) == 0:
        return 0
    cosine_angle = np.dot(thigh, shin) / (np.linalg.norm(thigh) * np.linalg.norm(shin))
    angle_rad = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    angle_deg = np.degrees(angle_rad)
    return angle_deg


def classify_position(skeleton):
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


def extract_orientation_features(skeleton):
    features = []
    for joint_id in POSTURE_JOINTS:
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
        posture_history[person_id] = deque(maxlen=POSTURE_WINDOW)

    posture_history[person_id].append(features)

    # Compare latest posture to average of earlier
    if len(posture_history[person_id]) < POSTURE_WINDOW:
        return False  # Not enough data yet

    baseline = np.mean(list(posture_history[person_id])[:-5], axis=0)
    latest = features
    diff = np.linalg.norm(latest - baseline)

    return diff > POSTURE_THRESHOLD


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


def draw_head_bounding_box(img_np, body, image_scale):
    bbox = np.array(body.head_bounding_box_2d, dtype=np.int32)  # shape (4, 2)

    if bbox.shape != (4, 2):
        return

    bbox[:, 0] = (bbox[:, 0] * image_scale[0]).astype(int)
    bbox[:, 1] = (bbox[:, 1] * image_scale[1]).astype(int)

    bbox_reshaped = bbox.reshape((-1, 1, 2))
    cv2.polylines(
        img_np, [bbox_reshaped], isClosed=True, color=(0, 255, 255), thickness=2
    )

    top_left = bbox[0]
    text_pos = (top_left[0], max(top_left[1] - 10, 0))
    cv2.putText(
        img_np,
        f"ID: {body.id}",
        text_pos,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
    )


def record_svo(participant_ID, zed, lsl_outlet):
    output_dir = os.makedirs(pl.Path("motion_classification_SVO_files"), exist_ok=True)
    output_svo_file = (
        pl.Path(output_dir)
        / f"{participant_ID}_{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')}.svo2"
    )

    lsl_outlet.push_sample([f"SVO_recording_path: {output_svo_file}"])

    recording_param = sl.RecordingParameters(
        output_svo_file, sl.SVO_COMPRESSION_MODE.H265
    )  # Enable recording with the filename specified in argument

    err = zed.enable_recording(recording_param)
    if err != sl.ERROR_CODE.SUCCESS:
        print("Recording ZED : ", err)
        exit(1)

    # Start Recording
    svo_start_time = datetime.now()
    lsl_outlet.push_sample([
        f"SVO_recording_start: {svo_start_time.strftime('%Y-%m-%d %H:%M:%S.%f')}"
    ])

    sl.RuntimeParameters()


# --- Main Pipeline ---


def main():
    # Create a Camera object
    zed = sl.Camera()

    # Create new stream info for lsl, stream camera_open, change source_id from "zed2i-harlem" to appropriate device, ex: "zed2i-midtown"
    info = StreamInfo("MotionTracking", "Markers", 1, 0, "string", "zed2i-midtown")
    lsl_outlet = StreamOutlet(info)

    while True:
        key = input(
            "Press 'c' to continue after starting lsl stream in LabRecorder: "
        ).strip()
        if key == "c":
            break

    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD1080  # Use HD1080 video mode
    init_params.coordinate_units = sl.UNIT.METER  # Set coordinate units
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP

    # Open the camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        exit(1)

    start_time = datetime.now()
    lsl_outlet.push_sample([
        f"camera_open: {start_time.strftime('%Y-%m-%d %H:%M:%S.%f')}"
    ])

    record_svo(opt.participant_id, zed, lsl_outlet)

    # Enable Positional tracking (mandatory for object detection)
    positional_tracking_parameters = sl.PositionalTrackingParameters()
    positional_tracking_parameters.set_as_static = True  # camera is static
    zed.enable_positional_tracking(positional_tracking_parameters)

    # Set Body Tracking parameters
    body_param = sl.BodyTrackingParameters()
    body_param.enable_tracking = True  # Track people across images flow
    body_param.enable_body_fitting = True  # Smooth skeleton move
    body_param.detection_model = sl.BODY_TRACKING_MODEL.HUMAN_BODY_ACCURATE
    body_param.body_format = (
        sl.BODY_FORMAT.BODY_38
    )  # Choose the BODY_FORMAT you wish to use
    zed.enable_body_tracking(body_param)

    # Set Body Tracking Runtime parameters
    body_runtime_param = sl.BodyTrackingRuntimeParameters()
    body_runtime_param.detection_confidence_threshold = 50

    # Get ZED camera information
    camera_info = zed.get_camera_information()

    # 2D viewer utilities
    display_resolution = sl.Resolution(
        min(camera_info.camera_configuration.resolution.width, 1280),
        min(camera_info.camera_configuration.resolution.height, 720),
    )
    image_scale = [
        display_resolution.width / camera_info.camera_configuration.resolution.width,
        display_resolution.height / camera_info.camera_configuration.resolution.height,
    ]

    # Create ZED objects filled in the main loop
    image = sl.Mat()
    key = ""
    bodies = sl.Bodies()
    image = sl.Mat()
    f = 0

    # Start body tracking
    print("Press 'q' to QUIT without saving. Press 'd' when DONE and save.")

    while True:  # Infinite loop
        key = cv2.waitKey(1) & 0xFF  # Non-blocking key press check

        if key == ord("q"):
            print("You pressed 'q', quitting...")
            key_press = datetime.now()
            lsl_outlet.push_sample([
                f"quit_key_press: {key_press.strftime('%Y-%m-%d %H:%M:%S.%f')}"
            ])
            break

        if key == ord("d"):
            print("You pressed 'd', motion tracking is done.")
            key_press = datetime.now()
            lsl_outlet.push_sample([
                f"done_key_press: {key_press.strftime('%Y-%m-%d %H:%M:%S.%f')}"
            ])
            break

        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            # count frames
            f += 1

            zed.retrieve_bodies(bodies, body_runtime_param)
            zed.retrieve_image(image, sl.VIEW.LEFT, sl.MEM.CPU, display_resolution)
            if len(bodies.body_list) == 0:
                print("MOVE INTO FRAME")
            else:
                selected_body = bodies.body_list[0]
                behavior = get_behavior(selected_body)
                orientation, postural_shift = get_head_orientation_with_posture(
                    selected_body
                )

                lsl_outlet.push_sample([
                    f"Frame {f} - Action: {selected_body.action_state}, Behavior: {behavior}, Head Direction: {orientation}"
                ])
                print(
                    f"Frame {f} - Action: {selected_body.action_state}, Behavior: {behavior}, Head Direction: {orientation}"
                )

                if postural_shift:
                    lsl_outlet.push_sample([f"Frame {f} - POSTURAL SHIFT detected"])
                    print(f"Frame {f} - POSTURAL SHIFT detected")

                # Display skeletons
                zed.retrieve_image(image, sl.VIEW.LEFT, sl.MEM.CPU, display_resolution)
                image_left_ocv = image.get_data()
                cv_viewer.render_2D(
                    image_left_ocv,
                    image_scale,
                    [selected_body],
                    body_param.enable_tracking,
                    body_param.body_format,
                )
                cv2.imshow("ZED | 2D View", image_left_ocv)
                cv2.moveWindow("ZED | 2D View", 100, 100)

        # Zed connection failed
        elif zed.grab() != sl.ERROR_CODE.SUCCESS:
            zed_err = datetime.now()
            lsl_outlet.push_sample([
                f"failed_zed_connection: {zed_err.strftime('%Y-%m-%d %H:%M:%S.%f')}"
            ])
            print("Failed ZED connection")
            break

    image.free(sl.MEM.CPU)
    zed.disable_body_tracking()
    zed.disable_positional_tracking()
    zed.close()

    end_time = datetime.now()
    lsl_outlet.push_sample([
        f"camera_close: {end_time.strftime('%Y-%m-%d %H:%M:%S.%f')}"
    ])

    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--participant_id", required=True, type=str, help="Participant ID"
    )
    opt = parser.parse_args()

    main()
