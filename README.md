# zed2i_motion_classifier
This repository takes in live zed feed to perform movement classification on the target body and streams frame by frame classifications over lsl.

![stability-experimental](https://img.shields.io/badge/stability-experimental-orange.svg)
[![LGPL--2.1 License](https://img.shields.io/badge/license-LGPL--2.1-blue.svg)](https://github.com/childmindresearch/mobi-motion-tracking/blob/main/LICENSE)
[![pages](https://img.shields.io/badge/api-docs-blue)](https://github.com/childmindresearch/zed2i_3d_lsl_capture)

Welcome to `zed2i_motion_classifier`, a Python Repository designed for real time motion classification using the ZED 2i stereo camera developed by StereoLabs (https://github.com/stereolabs/) and stream live marker events via LSL (https://labstreaminglayer.readthedocs.io/info/intro.html). This repository performs real time action, behavior, and position classification on a single person, streams these classifications over LSL, and records video to a .svo2 file. The markers streamed to LSL include the camera open, output svo path, frame by frame classifications, and camera close events.

## Supported software & devices

The package currently supports the ZED 2i and is reliant on proper installation of the `zed-sdk` (https://github.com/stereolabs/zed-sdk) and the `zed-python-api` (https://github.com/stereolabs/zed-python-api). It is also reliant on pylsl (https://labstreaminglayer.readthedocs.io/info/getting_started.html). If you want to run this data collection pipeline without LSL integration see (https://github.com/childmindresearch/zed2i_3d_capture).

**Special Note**
    The ZED SDK is only supported on Windows devices. Please see https://www.stereolabs.com/docs#supported-platforms for full details on ZED supported platforms.
    

## Processing pipeline implementation

The main processing pipeline of the `zed2i_motion_classifier` module can be described as follows:

- **Initiate LSL stream**: The user will be prompted to initiate the zed stream in LabRecorder.
- **Initiate the camera**: The zed camera will be triggered to open. If the camera cannot be accessed, an error will be thrown. 
- **Begin body tracking**: Skeletal joints will begin being captured at 30 fps. The pipeline can be manually interrupted by pressing the 'q' key.
- **Live Classifications**: For every frame, the target will be classified by ACTION (idle or moving), BEHAVIOR (still, slightly fidgety, fidgety, very fidgety, moving), and HEAD DIRECTION (up, down, left, right, forward).
- **Body tracking ends**: The pipeline can be concluded by pressing the 'd' key.
- **Export data**: The live recording of the participant will be saved as a .svo2 file located in motion_classification_SVO_files/.


## LSL Event Markers

Below is a complete list of all possible LSL event markers to be streamed dependent on various events that may occur during data collection:

- camera_open
- SVO_recording_start
- quit_key_press
- done_key_press
- classifications
- failed_zed_connection
- camera_close


## Installation

1. Install the ZED SDK from StereoLabs. Installation documentation can be found here: https://www.stereolabs.com/docs/installation/windows 
    - *** When prompted to select the folder location for the ZED SDK, you can use the default path ("C:\Program Files (x86)\ZED SDK") or change it based on your preference. However, this readme is based on the default path.

2. Grant administrative permissions to the ZED SDK. 
    - Navigate to the ZED SDK folder in "C:\Program Files (x86)" in file explorer
    - Right click on the folder -> select properties -> go to security tab -> click edit
    - Select the correct user to grant access to and tick the box next to full control under "Allow" 
    - Click apply and Ok
    - Restart your terminal

3. Navigate to the ZED SDK folder:
```sh
cd "C:\Program Files (x86)\ZED SDK"
```

4. Create a virtual environment. Any environment management tool can be used, but the following steps describe setting up a uv venv:

***NOTE:*** If you already have a zed2i_lsl_venv correctly setup from https://github.com/childmindresearch/zed2i_3d_lsl_capture, use the same environment and install the packages listed in step 6.

create a virtual environment named zed2i_lsl_venv
```sh
uv venv zed2i_lsl_venv
```
 activate the environment
```sh
zed2i_lsl_venv/Scripts/activate
```

5. Install the ZED Python API. Installation support documentation can be found here on the Stereolabs website (https://www.stereolabs.com/docs/app-development/python/install). However, follow our steps below for proper CMI/MoBI-specific API installation:

ensure pip is installed 
```sh
python -m ensurepip
```
install API dependencies
```sh
uv pip install cython numpy opencv-python requests
```
run get_python_api.py
```sh
uv run get_python_api.py
```

***NOTE:*** After running get_python_api.py, numpy>=2.3.0 will be automatically installed. This will cause additional dependency issues. Downgrade numpy to v2.1.0 In order to proceed with this package.

6. Install repository-dependent packages

```sh
uv pip install scipy==1.15.0 pylsl
```


## Quick start
Clone this repository inside ZED SDK:

```sh
git clone https://github.com/cgmaiorano/zed2i_motion_classifier.git
```

Navigate to root:

```sh
cd zed2i_motion_classifier
```

Run the pipeline with Participant ID 100 and live display

```sh
uv run __main__.py -p "100" -d
```
