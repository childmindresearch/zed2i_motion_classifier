"""Opencv viewer for real time motion tracking."""

import cv2
import pyzed.sl as sl

from display import utils


def cvt(pt, scale) -> list:
    """Function that scales point coordinates to the image scale

    Args:
        pt: 3d keypoint of joint.
        scale: image display scale.

    Return:
        out: list of scaled keypoint values.
    """
    out = [pt[0] * scale[0], pt[1] * scale[1]]
    return out


def render_sk(left_display, img_scale, obj, color, BODY_BONES) -> None:
    """Render the skeleton on top of opencv RGB camera feed.

    Args:
        left_display: left RGB image from zed camera.
        img_scale: image display scale.
        obj: skeleton of tracked person.
        color: display color for tracked skeleton.
        BODY_BONES: Skeleton type to display (18, 34, or 38).
    """
    for part in BODY_BONES:
        kp_a = cvt(obj.keypoint_2d[part[0].value], img_scale)
        kp_b = cvt(obj.keypoint_2d[part[1].value], img_scale)
        # Check that the keypoints are inside the image
        if (
            kp_a[0] < left_display.shape[1]
            and kp_a[1] < left_display.shape[0]
            and kp_b[0] < left_display.shape[1]
            and kp_b[1] < left_display.shape[0]
            and kp_a[0] > 0
            and kp_a[1] > 0
            and kp_b[0] > 0
            and kp_b[1] > 0
        ):
            cv2.line(
                left_display,
                (int(kp_a[0]), int(kp_a[1])),
                (int(kp_b[0]), int(kp_b[1])),
                color,
                1,
                cv2.LINE_AA,
            )

    # Skeleton joints
    for kp in obj.keypoint_2d:
        cv_kp = cvt(kp, img_scale)
        if cv_kp[0] < left_display.shape[1] and cv_kp[1] < left_display.shape[0]:
            cv2.circle(left_display, (int(cv_kp[0]), int(cv_kp[1])), 3, color, -1)


def render_2D(left_display, img_scale, objects, is_tracking_on, body_format) -> None:
    """Render joints and bones ontop of RGB live camera feed.

    Args:
        left_display: left RGB image from zed camera.
        img_scale: image display scale.
        objects: list of skeletons or tracked people in frame.
        is_tracking_on: Boolean value determining if a body is a tracked object in zed sdk.
        body_format: Zed skeleton type to display (18, 34, or 38).
    """
    overlay = left_display.copy()

    # Render skeleton joints and bones
    for obj in objects:
        if utils.render_object(obj, is_tracking_on):
            if len(obj.keypoint_2d) > 0:
                color = utils.generate_color_id_u(obj.id)
                if body_format == sl.BODY_FORMAT.BODY_18:
                    render_sk(left_display, img_scale, obj, color, sl.BODY_18_BONES)
                elif body_format == sl.BODY_FORMAT.BODY_34:
                    render_sk(left_display, img_scale, obj, color, sl.BODY_34_BONES)
                elif body_format == sl.BODY_FORMAT.BODY_38:
                    render_sk(left_display, img_scale, obj, color, sl.BODY_38_BONES)

    cv2.addWeighted(left_display, 0.9, overlay, 0.1, 0.0, left_display)
