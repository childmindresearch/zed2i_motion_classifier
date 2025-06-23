import os
import pathlib as pl
from datetime import datetime
import pyzed as sl


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
