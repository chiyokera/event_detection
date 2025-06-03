# Ball-Action Event Detection Modules

This repository let you try **Event Detection** and making commentator-like text corresponding to your sample soccer video. If you wanna try to train your own models, You also need to download **Soccer Net** Video Dataset.

## Flow for Applying Event Detection System for Your Video

### 0. Preliminary Setting

#### 0.1 Install Library

First of all, clone this git repository in your CDK.
And note that you need following libraries to use this system for Team or Location Detection Task. Specifically, PyNvCodec calls for NVDIA Drivers, NVIDIA GPUs and Supported Operating Systems (Windows and Linux)

- FFmpeg
- PyNvCodec

So, please make new python env like following ways and run requirements.txt

```
$ conda activate event_detection
$ pip install -r requirements.txt
```

or

```
$ python -m venv event_detection
$ pip install -r requirements.txt
```

If you get something trouble on FFmpeg or PyNvCodec install, please refer to the official documents ([FFmpeg](https://www.ffmpeg.org/),  [PyNvCodec](https://docs.nvidia.com/video-technologies/pynvvideocodec/pynvc-api-prog-guide/index.html))

#### 0.2 Checkpoint(Model Param) Downloading

There are 3 kinds of modules in this system. Download the `.pth` files from this [Drive](https://drive.google.com/drive/folders/13mNx6O6_T4fiPRG2WLQxfcWg1_5McOGa?usp=sharing)/checkpoints respectively and set them indicted repository. (Team or Location Detection Model Weights: about 50MB, T-DEED_challenge: about 170MB, T-DEED_test: about 40MB)

- T-DEED for Ball-Action Spotting
  - SoccerNetBall_challenge2: Trained 500 action spotting games and **7** ball-action spotting games
    - Set the param at `data/tdeed/checkpoints/SoccerNetBall_challenge2`
  - SoccerNetBall_test: Trained 500 action spotting games and **5** ball-action spotting games
    - Set the param at `data/tdeed/checkpoints/SoccerNetBall_test`
- Location Detection
  - easy: To detect ball location of the input 15 frames from `Right(half), Left, OUT`
  - hard: To detect ball location of the input 15 frames from `Center midfield, Top midfield, Bottom midfield, Top corner, Bottom corner, Edge of the box, Top box, Bottom box`
- Team Detection
  - To detect the own half of the player with ball in the input 15 frames from `Right, Left`
  
### 1. Set Your Sample Video

- Note that your original soccer video'fps must be 25 fps (event though like 25.01 is OUT!!)
  - If you want to use trimed video, use ffmpeg and follow this command
    - `$ ffmpeg -i input.mp4 -ss 00:02:00 -to 00:08:00 -c copy output.mp4` (Don't forget to set `-c copy` in the command)
- Please set your video into `data/sample/videos`
- `.mp4` is preferable, but you can use any filetype by changing this source code.
- You can place **multiple videos** in it

#### 1.1 Write Game Info

As `sample_game_info.json`, write meta info like game name, video name, right-side team name, left-side team name etc. And then set it in `data/sample/results/video_info` directory. (As example, I've already set it on this repository. Please help yourself to change your own one)

### 2. Execute Event Detection

Now, we'll start 3 things at one python file `sample_inference.py`.

1. Extracting frames of the video you set for T-DEED
   - Count the number of frames of each video and put each image(frame0.jpg~frame{max_num-1}) into `data/sample/frames`
2. Ball-Action Spotting
   - Using extracted frames, predict plausible ball-action from 12classes for each image, and using nms technique and my own filter, you can get final results in `data/sample/results/{video_name}/action`. My own filtering way is also important to make the results better, so I spend so mach time to adjust the hyperparameters. If you have interests or want modified, please go to `my_filter_nms` method in `inference_tdeed.py`
3. Location and Team Detection
   - Location: 17 classes {"Right center midfield",
    "Right bottom midfield",
    "Right top midfield",
    "Right bottom corner",
    "Right top corner",
    "Right bottom box",
    "Right edge of the box",
    "Right top box",
    "0",
    "Left center midfield",
    "Left bottom midfield",
    "Left top midfield",
    "Left bottom corner",
    "Left top corner",
    "Left bottom box",
    "Left edge of the box",
    "Left top box"}. As mentioned, at easy part, model detects where ball is from {Right, Left, OUT} and if Right or Left, as the hard part, from {center mid field, top midfield, bottom midfield, top corner, bottom corner, edge of the box, top box, bottom box}

    - Team: Choose the own side half of the one player who did event(with ball) from 2 classes {Right, Left}


```
$ cd src/scripts/sample_inference
$ python inference_all.py --model SoccerNetBall_challenge2 --seed 1 --save_filename results_spotting_integrated 

```

- --model: Choose from {SoccerNetBall_challenge2, SoccerNetBall_test}


### 3. SRT File Making for Visualization

- In this part, you can make SRT file which is ued for subtitle of the raw video.
- At argument, you can choose one task. For example, if you set "action" as task, only predicted action is written in the srt file.
  - If you choose "all" as task, template-based event texts are written. Theses are tailored by author me.
    - Additionally, if you have the OPENAI Account, after set API key `.txt` file on just below this parent dir(meaning next to src, data, or configs as `OpenAI_API_key.txt`), GPT-4o will change the event texts into even more commentator-like short texts. Of course, the srt filed will be written.
    - Furthermore, if you wanna make Voice speech and integrate it into your own sample video, run `src/scripts/sample_inference/srt_to_wav_to_video.py`. Before that you need prepare for the Voice model and config. For the tts, I used [Pipper](https://github.com/rhasspy/piper/blob/master/VOICES.md) so choose your preferable voice and download model and config. Then set them `data/voice_model`dir.

```
$ cd src/scripts/sample_inference
$ python srt_making_subtitle.py --task {"action", "team", "location", "all"}
```

Voice making and integration (Individual Video)
```
$ python srt_to_wav_to_video.py --task {"action", "team", "location", "all"}
```

### Extra 1: Training of Location or Team Detection Model

If you want to train Location or Team Detection Model with SoccerNet2024 Dataset, please comply below indication. (※ Tdeed train file is also in this repository, but I haven't yet confirm that it works well.Sorry)

#### Extra 1.1 Downloading SoccerNet2024 Dataset

- Video: Download Train, Valid and Test 7games video and annotation. For downloading, you need NDA password, so please fulfill this [Google form](https://docs.google.com/forms/d/e/1FAIpQLSfYFqjZNm4IgwGnyJXDPk2Ko_lZcbVtYX73w5lf6din5nxfmA/viewform) and set it at `--password_videos` argument.
  ```
  $ cd src/scripts/train/team_location_detection
  $ python download_ball_data.py --password_dir $[NDA] --without_challenge 
  ```
- Team and Location Label: I modified original Label you can download by above file in order to train Location or Team detection model. Click this [Drive](https://drive.google.com/drive/folders/1vJ6i2vAl6XZk3NyErNGsKNfEFpECIWKh?usp=drive_link)/soccernet/ and then download "england_efl/2019-2020/*" folder. Subsequently, set it in `data/team_location_detection/soccernet/england_efl/2019-2020/*(each game)`

#### Extra 1.2 Training Some Module

- Next, Let's train {location_easy, location_hard, location, team} model!:
  - location_easy: Detect which half the ball is from  {right, left, out} 
  - location_hard: Detect {center_midfield, top_midfield, bottom_midfield, top_corner, bottom_corner, edge of the box, top_box, bottom_box}
  - location: Detect one region from 17 classes (right or left x location_hard + out)
  - team: Detect which half is own half for the player involved in the events

### Extra 2: Reconstruction of Test Results

- **Benchmark**: To know the current model accuracy, we use modified SoccerNet Ball-Action Spotting Test Dataset
- **Flow**: 
  1. Making Results

```
cd inference_all.py 
```

  1. 