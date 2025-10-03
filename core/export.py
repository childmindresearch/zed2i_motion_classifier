import os
import pathlib as pl
from datetime import datetime
import pyzed.sl as sl

def record_svo(participant_ID, zed, lsl_outlet):
    """This fucntion initiates an svo recording to be saved upon pipeline completion.

    Change the directory or filenaming scheme below to match your desired preferences.

    Args:
        participant_ID: str containing user-input ID number.
        zed: zed camera object.
        lsl_outlet: pylsl object to stream markers.
    """
    os.makedirs(pl.Path("motion_classification_SVO_files"), exist_ok=True)
    output_dir = "motion_classification_SVO_files"
    output_svo_file = (
        pl.Path(output_dir)
        / f"{participant_ID}_{datetime.now().strftime('%Y-%m-%d.%f')}.svo2"
    )

    lsl_outlet.push_sample([
        f"SVO_recording_path: {output_svo_file}",
            "",
            "",
            "",
            ""
            ])
    recording_param = sl.RecordingParameters()
    recording_param.compression_mode = sl.SVO_COMPRESSION_MODE.H264
    recording_param.video_filename = str(output_svo_file)

    err = zed.enable_recording(recording_param)
    if err != sl.ERROR_CODE.SUCCESS:
        print("Recording ZED : ", err)
        svo_start_time = datetime.now()

        lsl_outlet.push_sample([
            f"svo_recording_err: {svo_start_time.strftime('%Y-%m-%d %H:%M:%S.%f')}",
            "",
            "",
            "",
            ""
            ])
        exit(1)

    # Start Recording
    svo_start_time = datetime.now()

    lsl_outlet.push_sample([
        f"SVO_recording_start: {svo_start_time.strftime('%Y-%m-%d %H:%M:%S.%f')}",
            "",
            "",
            "",
            ""
            ])
    sl.RuntimeParameters()
