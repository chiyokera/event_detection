# Play Event Detection Systems (:construction:This repository is constructing now)

This repository allows you to perform **Event Detection** and generate commentator-like text for your sample soccer videos. If you want to train your own models, you will also need to download the **SoccerNet** Video Dataset.

## Workflow for Applying Event Detection System to Your Video

### 0. Preliminary Setup

#### 0.1 Install Libraries

First, clone this git repository to your development environment.
Note that you need the following libraries to use this system for Team or Location Detection tasks. Specifically, PyNvCodec requires NVIDIA Drivers, NVIDIA GPUs, and supported operating systems (Windows and Linux):

- FFmpeg
- PyNvCodec

Please create a new Python environment using one of the following methods and install the requirements:

```bash
$ conda create -n event_detection python=3.8
$ conda activate event_detection
$ pip install -r requirements.txt
```

or

```bash
$ python -m venv event_detection
$ source event_detection/bin/activate  # On Windows: event_detection\Scripts\activate
$ pip install -r requirements.txt
```

If you encounter any issues with FFmpeg or PyNvCodec installation, please refer to the official documentation ([FFmpeg](https://www.ffmpeg.org/), [PyNvCodec](https://docs.nvidia.com/video-technologies/pynvvideocodec/pynvc-api-prog-guide/index.html)).

#### 0.2 Model Checkpoint Download

This system contains 3 types of modules. Download the `.pth` files from this [Google Drive](https://drive.google.com/drive/folders/13mNx6O6_T4fiPRG2WLQxfcWg1_5McOGa?usp=sharing)/checkpoints and place them in the indicated directories. (Team or Location Detection Model Weights: ~50MB, T-DEED_challenge: ~170MB, T-DEED_test: ~40MB)

- **T-DEED for Ball-Action Spotting**
  - SoccerNetBall_challenge2: Trained on 500 action spotting games and **7** ball-action spotting games
    - Place the checkpoint at `data/tdeed/checkpoints/SoccerNetBall_challenge2`
  - SoccerNetBall_test: Trained on 500 action spotting games and **5** ball-action spotting games
    - Place the checkpoint at `data/tdeed/checkpoints/SoccerNetBall_test`
- **Location Detection**
  - Easy: Detects ball location from input 15 frames among `Right(half), Left, OUT`
  - Hard: Detects ball location from input 15 frames among `Center midfield, Top midfield, Bottom midfield, Top corner, Bottom corner, Edge of the box, Top box, Bottom box`
- **Team Detection**
  - Detects the own half of the player with the ball from input 15 frames among `Right, Left`
  
### 1. Prepare Your Sample Video

- **Important**: Your original soccer video must have exactly 25 fps (even 25.01 fps will not work!)
  - If you want to use a trimmed video, use ffmpeg with the following command:
    - `$ ffmpeg -i input.mp4 -ss 00:02:00 -to 00:08:00 -c copy output.mp4` (Don't forget the `-c copy` flag)
- Place your video in the `data/sample/videos` directory
- `.mp4` format is preferred, but you can use other formats by modifying the source code
- You can place **multiple videos** in this directory

#### 1.1 Create Game Information File

Create a JSON file similar to `sample_game_info.json` containing metadata such as game name, video name, right-side team name, left-side team name, etc. Place this file in the `data/sample/results/video_info` directory. (An example is already provided in this repository for reference)

### 2. Execute Event Detection

The system performs 3 main tasks in a single Python script:

1. **Frame Extraction for T-DEED**
   - Counts the number of frames in each video and extracts individual images (frame0.jpg ~ frame{max_num-1}) to `data/sample/frames`

2. **Ball-Action Spotting**
   - Uses extracted frames to predict probable ball-actions from 12 classes for each image
   - Applies NMS (Non-Maximum Suppression) technique and custom filtering to generate final results in `data/sample/results/{video_name}/action`
   - The custom filtering method is crucial for improving results and has been carefully tuned. If you're interested in modifications, refer to the `my_filter_nms` method in `inference_tdeed.py`

3. **Location and Team Detection**
   - **Location**: Classifies into 17 classes:
     - "Right center midfield", "Right bottom midfield", "Right top midfield"
     - "Right bottom corner", "Right top corner", "Right bottom box"
     - "Right edge of the box", "Right top box", "0"
     - "Left center midfield", "Left bottom midfield", "Left top midfield"
     - "Left bottom corner", "Left top corner", "Left bottom box"
     - "Left edge of the box", "Left top box"
   - As mentioned earlier, the easy model detects ball location from {Right, Left, OUT}, and if Right or Left is detected, the hard model further classifies into specific field regions
   - **Team**: Determines the own half of the player involved in the event (with ball) from 2 classes {Right, Left}

```bash
$ cd src/scripts/sample_inference
$ python inference_all.py --model SoccerNetBall_challenge2 --seed 1 --save_filename results_spotting_integrated
```

**Arguments:**
- `--model`: Choose from {SoccerNetBall_challenge2, SoccerNetBall_test}


### 3. Generate SRT Files for Visualization

This step creates SRT subtitle files for your raw video.

- You can choose a specific task for subtitle generation. For example, setting "action" as the task will only include predicted actions in the SRT file.
- If you choose "all" as the task, template-based event texts will be generated (these templates are custom-designed).
  - **OpenAI Integration**: If you have an OpenAI account, create an API key file named `OpenAI_API_key.txt` in the parent directory (next to src, data, or configs). GPT-4o will then convert the event texts into more natural commentator-like descriptions.
  - **Voice Generation**: To create voice narration and integrate it into your video, run `src/scripts/sample_inference/srt_to_wav_to_video.py`. You'll need to prepare a voice model and configuration first. This system uses [Piper](https://github.com/rhasspy/piper/blob/master/VOICES.md) for TTS - choose your preferred voice, download the model and config files, and place them in the `data/voice_model` directory.

```bash
$ cd src/scripts/sample_inference
$ python srt_making_subtitle.py --task {"action", "team", "location", "all"}
```

**Voice Generation and Integration (Individual Video):**
```bash
$ python srt_to_wav_to_video.py --task {"action", "team", "location", "all"}
```

### Extra 1: Training Location or Team Detection Models

If you want to train Location or Team Detection models using the SoccerNet2024 Dataset, follow the instructions below. (Note: T-DEED training files are included in this repository, but their functionality has not been fully verified yet.)

#### Extra 1.1 Download SoccerNet2024 Dataset

- **Videos**: Download Train, Validation, and Test videos (7 games) with annotations. You need an NDA password for downloading. Please complete this [Google Form](https://docs.google.com/forms/d/e/1FAIpQLSfYFqjZNm4IgwGnyJXDPk2Ko_lZcbVtYX73w5lf6din5nxfmA/viewform) and use the password with the `--password_videos` argument.
  ```bash
  $ cd src/scripts/train/team_location_detection
  $ python download_ball_data.py --password_dir $[NDA_PASSWORD] --without_challenge
  ```

- **Team and Location Labels**: Modified labels for training Location or Team detection models are available. Download the "england_efl/2019-2020/*" folder from this [Google Drive](https://drive.google.com/drive/folders/1vJ6i2vAl6XZk3NyErNGsKNfEFpECIWKh?usp=drive_link)/soccernet/ and place it in `data/team_location_detection/soccernet/england_efl/2019-2020/*(each game)`.

#### Extra 1.2 Train Models

Train the following models:
- **location_easy**: Detects which half the ball is in from {right, left, out}
- **location_hard**: Detects specific field regions from {center_midfield, top_midfield, bottom_midfield, top_corner, bottom_corner, edge_of_the_box, top_box, bottom_box}
- **location**: Detects one region from 17 classes (right or left × location_hard + out)
- **team**: Detects which half is the own half for the player involved in events

### Extra 2: Test Results Reconstruction

- **Benchmark**: Uses a modified SoccerNet Ball-Action Spotting Test Dataset to evaluate current model accuracy
- **Workflow**:
  1. Generate results using the inference pipeline
  2. Evaluate against ground truth annotations

```bash
$ cd src/scripts/test_inference
$ python test_inference.py
```