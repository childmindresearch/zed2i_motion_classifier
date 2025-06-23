"""Python based runner."""

import cv2
import pyzed.sl as sl
from pylsl import StreamInfo, StreamOutlet

from datetime import datetime
from classifiers import behavior, head_direction
from display import tracking_viewer
from core import export, initialize_parameters


def run(participant_id):
    """This function is responsible for the main processing of the pipeline.

    The pipeline first waits for the user to initiate the lsl stream in LabRecorder. Enter "c" once complete.
    An svo recording and the zed parameters are then initialized. The main processing loop then begins where for every frame,
    the position, behavior, and ehad direction classificatiosn are calculated and returned in the terminal. A live
    display with a skeleton overlay also appears. The pipeline will run indefinitely until user intervention when the user either
    clicks "q" to quit or "d" when done.

    Args:
        participant_id: cli user input str containing ID number.
    """
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

    # Initialize parameters and start svo recording
    initialize_parameters.initialize_zed_parameters(zed, lsl_outlet)
    export.record_svo(participant_id, zed, lsl_outlet)
    positional_tracking_parameters = sl.PositionalTrackingParameters()
    positional_tracking_parameters.set_as_static = True  # camera is static
    zed.enable_positional_tracking(positional_tracking_parameters)

    # Set Body Tracking parameters
    body_param = sl.BodyTrackingParameters()
    body_param.enable_tracking = True  # Track people across images flow
    body_param.enable_body_fitting = True  # Smooth skeleton move
    body_param.detection_model = getattr(sl.BODY_TRACKING_MODEL, config.DETECTION_MODEL)
    body_param.body_format = (
        sl.BODY_FORMAT.BODY_38
    )  # Choose the BODY_FORMAT you wish to use
    zed.enable_body_tracking(body_param)

    # Set Body Tracking Runtime parameters
    body_runtime_param = sl.BodyTrackingRuntimeParameters()
    body_runtime_param.detection_confidence_threshold = 50
    display_resolution, image_scale = initialize_parameters.display_utilities(zed)

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
                behavior_result = behavior.get_behavior(selected_body)
                orientation, postural_shift = (
                    head_direction.get_head_orientation_with_posture(selected_body)
                )

                lsl_outlet.push_sample([
                    f"Frame {f} - Action: {selected_body.action_state}, Behavior: {behavior_result}, Head Direction: {orientation}"
                ])
                print(
                    f"Frame {f} - Action: {selected_body.action_state}, Behavior: {behavior_result}, Head Direction: {orientation}"
                )

                if postural_shift:
                    lsl_outlet.push_sample([f"Frame {f} - POSTURAL SHIFT detected"])
                    print(f"Frame {f} - POSTURAL SHIFT detected")

                # Display skeletons
                zed.retrieve_image(image, sl.VIEW.LEFT, sl.MEM.CPU, display_resolution)
                image_left_ocv = image.get_data()
                tracking_viewer.render_2D(
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
