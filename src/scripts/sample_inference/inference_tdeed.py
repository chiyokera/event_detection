#!/usr/bin/env python3
"""
File containing the main training script for T-DEED.
"""
# Standard imports
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from pprint import pprint

import numpy as np
import torch
import wandb

sys.path.append(str(Path(__file__).parents[2]))
from tdeed.dataset.frame import FrameReaderVideo, InferenceActionSpotVideoDataset
from tdeed.model.model import TDEEDModel
from SoccerNet.Evaluation.ActionSpotting import evaluate as evaluate_SN
from torch.utils.data import DataLoader
from tqdm import tqdm
from tdeed.util.dataset import load_classes
from tdeed.util.eval import (
    evaluate,
    evaluate_SNB,
    non_maximum_supression,
    process_frame_predictions_challenge,
)
from tdeed.util.io import load_json, load_text, store_json, store_json_snb
from scripts.train.tdeed.train_tdeed import update_args

# Constants
STRIDE = 1
STRIDE_SN = 12
STRIDE_SNB = 2
TOLERANCES_SNB = [6, 12]
WINDOWS_SNB = [6, 12]
FPS_SN = 25
TDEED_DATA_DIR = "../../../data/tdeed"
SAMPLE_DIR = "../../../data/sample"
CONFIG_DIR = "../../../configs"


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)  # SoccerNetBall_challenge2
    parser.add_argument(
        "-ag", "--acc_grad_iter", type=int, default=1, help="Use gradient accumulation"
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--frame_dir",
        type=str,
        default="../../../data/sample/frames",
        help="Path to the frames",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="../../../data/sample/results",
        help="Path to save the predictions",
    )
    return parser.parse_args()


def action_spotting(args):
    # Set seed
    initial_time = time.time()
    print("Setting seed to: ", args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    config_path = os.path.join(CONFIG_DIR, "tdeed", args.model + ".json")
    config = load_json(config_path)
    args = update_args(args, config)

    assert args.dataset in ["soccernetball"]  # Only SoccerNet Ball is supported

    # # Variables for SN & SNB label paths if datastes
    # if (args.dataset == "soccernet") | (args.dataset == "soccernetball"):
    #     global LABELS_SN_PATH
    #     global LABELS_SNB_PATH
    #     LABELS_SN_PATH = load_text(
    #         os.path.join("data", "soccernet", "labels_path.txt")
    #     )[0]
    #     LABELS_SNB_PATH = load_text(
    #         os.path.join("data", "soccernetball", "labels_path.txt")
    #     )[0]

    assert args.batch_size % args.acc_grad_iter == 0
    if args.crop_dim <= 0:
        args.crop_dim = None

    # # initialize wandb
    # wandb.login()
    # wandb.init(
    #     config=args,
    #     dir=args.save_dir + "/wandb_logs",
    #     project="TDEED-ballaction-inference",
    #     name=args.model + "-" + str(args.seed),
    # )

    # Model
    model = TDEEDModel(args=args)

    # If pretrain -> 2 prediction heads
    # if args.pretrain != None:
    classes = load_classes(os.path.join(TDEED_DATA_DIR, "ballaction_class.txt"))
    pretrain_classes = load_classes(os.path.join(TDEED_DATA_DIR, "action_class.txt"))
    n_classes = [len(classes) + 1, len(pretrain_classes) + 1]
    model._model.update_pred_head(n_classes)
    model._num_classes = np.array(n_classes).sum()

    print("START INFERENCE")
    model.load(
        torch.load(
            os.path.join(
                TDEED_DATA_DIR,
                "checkpoints",
                "soccernetball",
                args.model,
                "checkpoint_best.pt",
            )
        )
    )

    frame_dir = Path(args.frame_dir)
    label_file = os.path.join(
        SAMPLE_DIR, "results", "video_info", "sample_video_info.json"
    )
    stride = STRIDE
    if args.dataset == "soccernet":
        stride = STRIDE_SN
    if args.dataset == "soccernetball":
        stride = STRIDE_SNB
    video_info = load_json(label_file)
    if video_info == []:
        print("No videos found in the dataset")
        return
    num_frames = video_info[0]["num_frames"]
    if os.path.exists(frame_dir):
        split_data = InferenceActionSpotVideoDataset(
            classes=classes,
            label_file=label_file,
            frame_dir=frame_dir,
            modality=args.modality,
            clip_len=args.clip_len,
            overlap_len=(
                args.clip_len // 4 * 3
                if args.dataset != "soccernet"
                else args.clip_len // 2
            ),  # 3/4 overlap for video dataset, 1/2 overlap for soccernet
            stride=stride,
            dataset="soccernetball",
        )

        # save_pred = os.path.join(
        #     "data",
        #     args.dataset,
        #     "sample-results",
        #     "preds.json",
        # )
        pred_dict = {}
        for (
            video,
            video_len,
            _,
        ) in (
            split_data.videos
        ):  # video = video name, video_len = number of frames in video
            pred_dict[video] = (
                np.zeros((video_len, len(classes) + 1), np.float32),
                np.zeros(video_len, np.int32),
            )

        # Do not up the batch size if the dataset augments
        batch_size = 1
        h = 0
        for clip in tqdm(
            DataLoader(
                split_data,
                num_workers=1,
                pin_memory=True,
                batch_size=batch_size,
            )
        ):
            # Batched by dataset
            scores, support = pred_dict[clip["video"][0]]
            start = clip["start"][0].item()
            _, pred_scores = model.predict(clip["frame"])
            if start < 0:
                pred_scores = pred_scores[:, -start:, :]
                start = 0
            end = start + pred_scores.shape[1]
            if end >= scores.shape[0]:
                end = scores.shape[0]
                pred_scores = pred_scores[:, : end - start, :]

            scores[start:end, :] += np.sum(pred_scores, axis=0)
            support[start:end] += pred_scores.shape[0]

            # Additional view with horizontal flip
            for i in range(1):
                start = clip["start"][0].item()
                _, pred_scores = model.predict(clip["frame"], augment_inference=True)
                if start < 0:
                    pred_scores = pred_scores[:, -start:, :]
                    start = 0
                end = start + pred_scores.shape[1]
                if end >= scores.shape[0]:
                    end = scores.shape[0]
                    pred_scores = pred_scores[:, : end - start, :]

                scores[start:end, :] += np.sum(pred_scores, axis=0)
                support[start:end] += pred_scores.shape[0]

        pred_events, pred_events_high_recall, pred_scores = (
            process_frame_predictions_challenge(
                split_data, classes, pred_dict, high_recall_score_threshold=0.01
            )
        )
        # print(
        #     f"pred_events_high_recall[video]: {pred_events_high_recall[0]['video']},{pred_events_high_recall[1]['video']}"
        # )
        windows = WINDOWS_SNB
        pred_events_high_recall_store = non_maximum_supression(
            pred_events_high_recall, window=windows[0], threshold=0.10
        )
        # print(
        #     f"pred_events_high_recall_store[video]: {pred_events_high_recall_store[0]['video']},{pred_events_high_recall_store[1]['video']}"
        # )
        save_dir = Path(args.save_dir)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        store_json(save_dir, pred_events_high_recall_store)

        # if not os.path.exists("/".join(save_pred.split("/")[:-1])):
        #     os.makedirs("/".join(save_pred.split("/")[:-1]))
        # store_json(save_pred, pred_events_high_recall_store)
        store_json_snb(save_dir, pred_events_high_recall_store, stride=stride)
        my_filter_nms(save_dir, pred_events_high_recall_store, stride, num_frames)

    print("CORRECTLY FINISHED INFERENCE")


def my_filter_nms(pred_dir, games, stride, num_frames):
    for game in games:
        gameDict = dict()
        gameDict["UrlLocal"] = game["video"]
        gameDict["predictions"] = []
        video_seconds = num_frames / FPS_SN
        one_seconde_window = [
            (FPS_SN * i, FPS_SN * (i + 1)) for i in range(int(video_seconds))
        ]
        filter_list = [{window: list()} for window in one_seconde_window]
        for i, event in enumerate(game["events"]):
            # print(f"event_frame: {event['frame']}")
            for j, window in enumerate(one_seconde_window):
                if (
                    event["frame"] * stride >= window[0]
                    and event["frame"] * stride < window[1]
                ):
                    filter_list[j][window].append((i, event))
                    break
        # filter1(全イベントの内の候補のindexを入れたい)
        max_idxes = set()
        for window_dict in filter_list:
            for window, events in window_dict.items():
                event_score_list = list()
                for _, event in events:
                    event_score_list.append(event["score"])

                if len(event_score_list) == 0:
                    continue

                max_idx = event_score_list.index(max(event_score_list))
                idx, max_event = events[max_idx]
                if max_event["score"] >= 0.15:
                    # print(
                    #     f"event_frame: {max_event['frame']*stride}, event_label: {max_event['label']}, event_score: {max_event['score']}"
                    # )
                    if max_event["label"] == "DRIVE":
                        if max_event["score"] >= 0.21:
                            max_idxes.add(idx)
                        else:
                            max_idx = -1
                    elif max_event["label"] == "HEADER":
                        if max_event["score"] >= 0.25:
                            max_idxes.add(idx)
                        else:
                            max_idx = -1
                    elif max_event["label"] == "PASS":
                        if max_event["score"] >= 0.18:
                            max_idxes.add(idx)
                        else:
                            if len(events) == 1:
                                max_idxes.add(idx)
                            else:
                                max_idx = -1
                    elif max_event["label"] == "SHOT":
                        if max_event["score"] >= 0.2:
                            max_idxes.add(idx)
                        else:
                            max_idx = -1
                    elif max_event["label"] == "CROSS":
                        if max_event["score"] >= 0.3:
                            max_idxes.add(idx)
                        else:
                            max_idx = -1
                    elif max_event["label"] == "THROW IN":
                        if max_event["score"] >= 0.61:
                            max_idxes.add(idx)
                    else:
                        max_idxes.add(idx)
                else:
                    if max_event["label"] == "OUT":
                        if max_event["score"] >= 0.1:
                            max_idxes.add(idx)
                    else:
                        max_idx = -1
                for i, (idx, event) in enumerate(events):
                    if i == max_idx:
                        continue
                    # 1秒以内だけど、特殊なイベントの場合はそのまま採用
                    if event["label"] == "HEADER":
                        if events[i][1]["score"] >= 0.32:
                            max_idxes.add(idx)
                    elif event["label"] == "GOAL":
                        if events[i][1]["score"] >= 0.12:
                            max_idxes.add(idx)
                    elif event["label"] == "HIGH PASS":
                        if events[i][1]["score"] >= 0.2:
                            max_idxes.add(idx)
                    elif event["label"] == "BALL PLAYER BLOCK":
                        if events[i][1]["score"] >= 0.17:
                            max_idxes.add(idx)
                    elif event["label"] == "PLAYER SUCCESSFUL TACKLE":
                        if events[i][1]["score"] >= 0.2:
                            max_idxes.add(idx)
                    elif event["label"] == "DRIVE":
                        if max_event["label"] == "PASS":
                            if events[i][1]["score"] >= 0.24:
                                max_idxes.add(idx)
                        elif events[i][1]["score"] >= 0.9:
                            max_idxes.add(idx)

                    elif event["label"] == "SHOT":
                        if events[i][1]["score"] >= 0.4:
                            max_idxes.add(idx)
                    elif event["label"] == "PASS":
                        if max_event["label"] == "DRIVE":
                            if events[i][1]["score"] >= 0.2:
                                max_idxes.add(idx)
                        elif events[i][1]["score"] >= 0.85:
                            max_idxes.add(idx)
                    elif event["label"] == "CROSS":
                        if events[i][1]["score"] >= 0.1:
                            max_idxes.add(idx)
        # print(
        #     f"game: {game['video']}, num_events: {len(game['events'])}, num_max_idxes: {len(max_idxes)}"
        # )
        # print(max_idxes)
        # for i, idx in enumerate(sorted(list(max_idxes))):
        #     print(game["events"][idx]["gameTime"], game["events"][idx]["label"])
        # filter2
        # もしあるイベントの前後1秒以内にイベントがある場合、scoreが高い方を採用,ただし、そのイベントがHEADERかGOALの場合はそのまま採用
        max_idxes2 = set()
        out_idxes = set()
        for i, idx in enumerate(sorted(list(max_idxes))):
            if idx in out_idxes:
                continue
            event = game["events"][idx]
            position = int(event["frame"] / FPS_SN * 1000) * stride
            event_label = event["label"]
            event_score = event["score"]
            conf_dict = dict()
            conf_dict[idx] = event_score
            if event_label == "OUT":
                if i == len(list(max_idxes)) - 1:
                    max_idxes2.add(idx)
                    continue
                t = i + 1
                while True:
                    if t >= len(list(max_idxes)):
                        break
                    next_idx = sorted(list(max_idxes))[t]
                    next_event = game["events"][next_idx]
                    next_event_label = next_event["label"]
                    if next_event_label == "OUT":
                        t += 1
                        conf_dict[next_idx] = next_event["score"]
                    else:
                        break
                # conf_dictの中で最も高いscoreを持つOUTを採用し、それ以外のidxはout_idxesに追加
                max_idx = max(conf_dict, key=conf_dict.get)
                max_idxes2.add(max_idx)
                for k in conf_dict.keys():
                    if k != max_idx:
                        out_idxes.add(k)
            else:
                if i < len(list(max_idxes)) - 1:
                    next_idx = sorted(list(max_idxes))[i + 1]
                    next_event = game["events"][next_idx]
                    next_position = int(next_event["frame"] / FPS_SN * 1000) * stride
                    next_event_label = next_event["label"]

                    if (
                        event_label == "HEADER"
                        or event_label == "GOAL"
                        or event_label == "HIGH PASS"
                        or event_label == "BALL PLAYER BLOCK"
                        or event_label == "PLAYER SUCCESSFUL TACKLE"
                        or event_label == "CROSS"
                    ):
                        if event_label == next_event_label:
                            if event_score >= next_event["score"]:
                                max_idxes2.add(idx)
                                if next_event["score"] >= 0.9:
                                    continue
                                else:
                                    out_idxes.add(next_idx)
                            else:
                                if event_score >= 0.9:
                                    max_idxes2.add(idx)
                        else:
                            max_idxes2.add(idx)
                            continue
                    if next_position - position <= 1000:
                        if next_position - position <= 500:
                            if event_score >= next_event["score"]:
                                if (
                                    event_label == "PASS"
                                    and next_event_label == "HIGH PASS"
                                ) or (
                                    event_label == "HIGH PASS"
                                    and next_event_label == "PASS"
                                ):
                                    continue

                                max_idxes2.add(idx)

                                if next_event["score"] >= 0.9:
                                    continue
                                if (
                                    next_event_label == "PLAYER SUCCESSFUL TACKLE"
                                    or next_event_label == "BALL PLAYER BLOCK"
                                ):
                                    continue

                                if (
                                    event_label == "PASS"
                                    and next_event_label == "CROSS"
                                ):
                                    continue
                                if (
                                    event_label == "PASS"
                                    and next_event_label == "DRIVE"
                                ):
                                    continue
                                if (
                                    event_label == "SHOT"
                                    and next_event_label == "BALL PLAYER BLOCK"
                                ):
                                    continue
                                else:
                                    out_idxes.add(next_idx)
                            else:
                                if event_score >= 0.9:
                                    max_idxes2.add(idx)
                                if (
                                    next_event_label == "GOAL"
                                    or next_event_label == "HIGH PASS"
                                    or next_event_label == "BALL PLAYER BLOCK"
                                    or next_event_label == "PLAYER SUCCESSFUL TACKLE"
                                ):
                                    max_idxes2.add(idx)
                        else:
                            if event_score >= next_event["score"]:
                                max_idxes2.add(idx)
                                if next_event["score"] >= 0.8:
                                    continue
                                elif (
                                    event_label == "PASS"
                                    and next_event_label == "DRIVE"
                                ):
                                    if next_event["score"] >= 0.25:
                                        continue
                                elif (
                                    event_label == "DRIVE"
                                    and next_event_label == "PASS"
                                ):
                                    if next_event["score"] >= 0.21:
                                        continue
                                elif (
                                    event_label == "PASS" and next_event_label == "PASS"
                                ):
                                    if next_event["score"] >= 0.22:
                                        continue
                                else:
                                    out_idxes.add(next_idx)
                            else:
                                if event_score >= 0.8:
                                    max_idxes2.add(idx)
                                if (
                                    event_label == "DRIVE"
                                    and next_event_label == "SHOT"
                                ):
                                    max_idxes2.add(idx)
                                if (
                                    event_label == "DRIVE"
                                    and next_event_label == "PASS"
                                ):
                                    max_idxes2.add(idx)
                                if (
                                    event_label == "PASS"
                                    and next_event_label == "DRIVE"
                                ):
                                    max_idxes2.add(idx)
                    else:
                        if (
                            next_position - position <= 1500
                            and event_label == "DRIVE"
                            and next_event_label == "DRIVE"
                        ):
                            if event_score >= next_event["score"]:
                                max_idxes2.add(idx)
                                if next_event["score"] >= 0.9:
                                    continue
                                else:
                                    out_idxes.add(next_idx)
                            else:
                                if event_score >= 0.9:
                                    max_idxes2.add(idx)
                        elif (
                            next_position - position <= 1500
                            and event_label == "THROW IN"
                            and next_event_label == "THROW IN"
                        ):
                            if event_score >= next_event["score"]:
                                max_idxes2.add(idx)
                                if next_event["score"] >= 0.9:
                                    continue
                                else:
                                    out_idxes.add(next_idx)
                            else:
                                if event_score >= 0.9:
                                    max_idxes2.add(idx)
                        else:
                            max_idxes2.add(idx)
                else:
                    max_idxes2.add(idx)
        for i, idx in enumerate(sorted(list(max_idxes2))):
            event = game["events"][idx]
            eventDict = dict()

            position = int(event["frame"] / FPS_SN * 1000) * stride
            if position == 0:
                continue
            eventDict["gameTime"] = "1 - {}:{}".format(
                position // 60000, int((position % 60000) // 1000)
            )
            eventDict["action"] = event["label"]
            eventDict["position"] = position
            eventDict["frame"] = event["frame"] * stride
            eventDict["action_score"] = event["score"]
            eventDict["half"] = 1
            gameDict["predictions"].append(eventDict)
        # path = os.path.join("/".join(pred_path.split("/")[:-1]), game["video"], "preds")
        pred_game_dir = os.path.join(pred_dir, game["video"], "action")
        os.makedirs(pred_game_dir, exist_ok=True)
        with open(
            os.path.join(pred_game_dir, "results_spotting_my_filtered_action.json"), "w"
        ) as fp:
            json.dump(gameDict, fp, indent=4)


if __name__ == "__main__":
    args = get_args()
    action_spotting(args)
