import pyzed.sl as sl
from datetime import datetime
from core import config


def initialize_zed_parameters(zed, lsl_outlet):
    """Create a InitParameters object and set configuration parameters.

    Args:
        zed: zed camera object.
        lsl_outlet: pylsl object to stream markers.
    """
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD1080  # Use HD1080 video mode
    init_params.camera_fps = 30
    init_params.coordinate_units = sl.UNIT.METER  # Set coordinate units
    init_params.depth_mode = getattr(sl.DEPTH_MODE, config.DEPTH_MODE)
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP

    # Open the camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print("Camera Open", err, "Exit program.")
        exit(1)

    start_time = datetime.now()

    lsl_outlet.push_sample([
        f"camera_open: {start_time.strftime('%Y-%m-%d %H:%M:%S.%f')}",
        "",
        "",
        "",
        ""
        ])


def display_utilities(zed):
    """Set the opencv display resolution and scale.

    Args:
        zed: zed camera object.
    """
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

    return display_resolution, image_scale

