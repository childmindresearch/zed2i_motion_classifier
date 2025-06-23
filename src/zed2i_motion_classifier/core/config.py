"""File containing all global parameters, variables, thresholds, and dictionaries for classifications."""

POSTURE_WINDOW = 30  # frames (1 second @ 30fps)
POSTURE_THRESHOLD = 10  # quaternion difference threshold (tune this)
POSTURE_JOINTS = [0, 1, 2, 3, 4]  # pelvis, spine1, spine2, spine3, neck

WINDOW_SIZE = 15  # longer window = smoother signal
MIN_MOTION_THRESHOLD = 0.01  # higher = only count more deliberate movement (1 m/frame)
MIN_ACTIVE_FRAMES = 5  # require at least 5 frames with motion > threshold

FIDGET_JOINTS = [16, 17, 32, 33, 34, 35, 36, 37]  # wrists and fingers

posture_history = {}  # person_id: deque of posture feature vectors
joint_histories = {}  # person_id: joint position history
frame_behavior_log = []  # to save classifications: (frame_id, person_id, behavior)

DEPTH_MODE = "NEURAL"
DETECTION_MODEL = "HUMAN_BODY_ACCURATE"
