"""utils for the opencv viewer."""

import pyzed.sl as sl


ID_COLORS = [
    (232, 176, 59),
    (175, 208, 25),
    (102, 205, 105),
    (185, 0, 255),
    (99, 107, 252),
]


def render_object(object_data, is_tracking_on) -> any:
    """Check to render object if object is tracked.

    Args:
        object_data: zed parameters for tracked person.
        is_tracking_on: boolean value determining if a body is a tracked object in zed sdk.
    """
    if is_tracking_on:
        return object_data.tracking_state == sl.OBJECT_TRACKING_STATE.OK
    else:
        return (object_data.tracking_state == sl.OBJECT_TRACKING_STATE.OK) or (
            object_data.tracking_state == sl.OBJECT_TRACKING_STATE.OFF
        )


def generate_color_id_u(idx) -> list[int] | list:
    """Generate color for skeleton display.

    Args:
        idx: object or tracked body id value.

    Returns:
        arr: list of RGB values for skeleton display.
    """
    arr = []
    if idx < 0:
        arr = [236, 184, 36, 255]
    else:
        color_idx = idx % 5
        arr = [
            ID_COLORS[color_idx][0],
            ID_COLORS[color_idx][1],
            ID_COLORS[color_idx][2],
            255,
        ]
    return arr
