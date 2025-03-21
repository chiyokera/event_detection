import argparse
import json
import os
import sys
from multiprocessing import Pool

import cv2
import moviepy.editor
from tqdm import tqdm
from pathlib import Path

cv2.setNumThreads(0)
sys.path.append(str(Path(__file__).parents[2]))
from tdeed.util.dataset import read_fps

"""
This script extracts frames from SoccerNetv2 Ball Action Spotting dataset by introducing the path where the downloaded videos are (at 720 resolution), the path to
write the frames, the sample fps, and the number of workers to use. The script will create a folder for each video in the out_dir path and save the frames as .jpg files in
the desired resolution.

python extract_frames_snb.py --video_dir video_dir
        --out_dir out_dir
        --sample_fps 25 --num_workers 5
"""

RECALC_FPS_ONLY = False
FRAME_RETRY_THRESHOLD = 1000
TARGET_HEIGHT = 448
TARGET_WIDTH = 796


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v",
        "--video_dir",
        default="../../../data/sample/videos",
        help="Path to the sample videos",
    )
    parser.add_argument(
        "-o",
        "--frames_out_dir",
        default="../../../data/sample/frames",
        help="Path to write frames. Dry run if None.",
    )
    parser.add_argument("--sample_fps", type=int, default=25)
    parser.add_argument("--recalc_fps", action="store_true")
    parser.add_argument("-j", "--num_workers", type=int, default=os.cpu_count() // 4)
    return parser.parse_args()


def get_duration(video_path):
    # Copied from SoccerNet repo
    return moviepy.editor.VideoFileClip(video_path).duration


def worker(args):
    video_name, video_path, out_dir, sample_fps = args

    def get_stride(src_fps):
        if sample_fps <= 0:
            stride = 1
        else:
            stride = int(src_fps / sample_fps)
        return stride

    vc = cv2.VideoCapture(video_path)
    fps = vc.get(cv2.CAP_PROP_FPS)

    num_frames = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(
        "{} -- fps: {} -- num_frames: {} -- w: {} -- h: {}".format(
            video_name, fps, num_frames, w, h
        )
    )
    oh = TARGET_HEIGHT
    ow = TARGET_WIDTH

    time_in_s = get_duration(video_path)

    fps_path = None
    if out_dir is not None:
        fps_path = os.path.join(out_dir, "fps.txt")
        if not RECALC_FPS_ONLY:
            if os.path.exists(fps_path):
                print("Already done:", video_name)
                vc.release()
                return
        else:
            if str(read_fps(out_dir)) == str(fps / get_stride(fps)):
                print("FPS is already consistent:", video_name)
                vc.release()
                return
            else:
                # Recalculate FPS in cases where the actual frame count does not
                # match the metadata
                print("Inconsistent FPS:", video_name)

        os.makedirs(out_dir, exist_ok=True)

    not_done = True
    while not_done:
        stride = get_stride(fps)
        est_out_fps = fps / stride
        print(
            "{} -- effective fps: {} (stride: {})".format(
                video_name, est_out_fps, stride
            )
        )

        i = 0
        while True:
            ret, frame = vc.read()
            if not ret:
                # fps and num_frames are wrong
                if i != num_frames:
                    print(
                        "Failed to decode: {} -- {} / {}".format(
                            video_path, i, num_frames
                        )
                    )

                    if i + FRAME_RETRY_THRESHOLD < num_frames:
                        num_frames = i
                        adj_fps = num_frames / time_in_s
                        if get_stride(adj_fps) == stride:
                            # Stride would not change so nothing to do
                            not_done = False
                        else:
                            print("Retrying:", video_path)
                            # Stride changes, due to large error in fps.
                            # Use adjusted fps instead.
                            vc.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            fps = adj_fps
                    else:
                        not_done = False
                        num_frames = i-1
                else:
                    not_done = False
                break

            if i % stride == 0:
                if not RECALC_FPS_ONLY:
                    if frame.shape[0] != oh or frame.shape[1] != ow:
                        frame = cv2.resize(frame, (ow, oh))
                    if out_dir is not None:
                        frame_path = os.path.join(out_dir, "frame{}.jpg".format(i))
                        cv2.imwrite(frame_path, frame)
            i += 1
    vc.release()

    out_fps = fps / get_stride(fps)
    if fps_path is not None:
        with open(fps_path, "w") as fp:
            fp.write(str(out_fps))

    print("{} - done".format(video_name))
    return video_name, num_frames

def inference_extract(args):
    video_dir = args.video_dir
    out_dir = args.frame_dir
    sample_fps = args.sample_fps
    recalc_fps = args.recalc_fps
    num_workers = args.num_workers

    global RECALC_FPS_ONLY
    RECALC_FPS_ONLY = recalc_fps
    VIDEO_INFO_PATH = os.path.join(
        "../../../data/sample/results/video_info",
        "sample_video_info.json",
    )
    worker_args = []
    label_files = []
    for sample_video in os.listdir(video_dir):
        if (not (".mp4" in sample_video)) or ("224" in sample_video):
            continue
        if os.path.exists(VIDEO_INFO_PATH):
            frame_exists = False
            with open(VIDEO_INFO_PATH, "r") as f:
                label_files = json.load(f)
                for label in label_files:
                    if label["video"] == sample_video.split(".")[0]:
                        frame_exists = True
                        break
            if frame_exists:
                continue
                
        # label_dict = {}
        video_name = sample_video
        out_frames_dir = os.path.join(out_dir, video_name.split(".")[0])
        os.makedirs(out_frames_dir, exist_ok=True)
        # label_dict["video"] = video_name.split(".")[0]
        video_path = os.path.join(video_dir, sample_video)
        vc = cv2.VideoCapture(video_path)
        # label_dict["num_frames"] = num_frames
        # label_files.append(label_dict)
        vc.release()
        
        worker_args.append(
            (
                video_name,
                video_path,
                out_frames_dir,  # out_dir to save frames
                sample_fps,
            )
        )
    print("Total videos to process:", worker_args)
    if len(worker_args) == 0:
        print("All videos are already processed!")
        return
    
    with Pool(num_workers) as p:
        for video_name, num_frame in tqdm(p.imap_unordered(worker, worker_args), total=len(worker_args)):
            label_files.append({"video": video_name.split(".")[0], "num_frames": num_frame})
            pass
    
    with open(VIDEO_INFO_PATH, "w") as f:
        json.dump(label_files, f)
    print("Done!")


if __name__ == "__main__":
    args = get_args()
    inference_extract(args)
