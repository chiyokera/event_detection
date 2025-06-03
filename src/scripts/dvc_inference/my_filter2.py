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

STRIDE = 1
STRIDE_SN = 12
STRIDE_SNB = 2
TOLERANCES_SNB = [6, 12]
WINDOWS_SNB = [6, 12]
FPS_SN = 25


def my_filter_nms2(pred_dir, games, stride, num_frames):
    for game in games:
        gameDict = dict()
        gameDict["UrlLocal"] = game["video"]
        gameDict["predictions"] = []
        video_seconds = num_frames / FPS_SN
        get_idxes = set()

        # THROW IN 0.6未満は除外
        for i, event in enumerate(game["events"]):
            if event["label"] == "THROW IN" and event["score"] < 0.6:
                continue
            else:
                get_idxes.add(i)

        out_idxes = set()
        for i, idx in enumerate(sorted(list(get_idxes))):
            if idx in out_idxes:
                continue
            event = game["events"][idx]
            position = int(event["frame"] / FPS_SN * 1000) * stride
            event_label = event["label"]
            event_score = event["score"]
            if i != len(list(get_idxes)) - 1:
                next_idx = sorted(list(get_idxes))[i + 1]
                next_event = game["events"][next_idx]
                next_position = int(next_event["frame"] / FPS_SN * 1000) * stride
                next_event_label = next_event["label"]
                next_time_diff = next_position - position
                if next_time_diff == 0:
                    # 連続しているイベントはスコアが高い方を採用
                    if event_score >= next_event["score"]:
                        out_idxes.add(next_idx)
                    else:
                        out_idxes.add(idx)

            # 連続しているOUTはスコアが高い方を採用
            conf_dict = dict()
            conf_dict[idx] = event_score
            if event_label == "OUT":
                if i == len(list(get_idxes)) - 1:
                    continue
                t = i + 1
                while True:
                    if t >= len(list(get_idxes)):
                        break
                    next_idx = sorted(list(get_idxes))[t]
                    next_event = game["events"][next_idx]
                    next_event_label = next_event["label"]
                    if next_event_label == "OUT":
                        t += 1
                        conf_dict[next_idx] = next_event["score"]
                    else:
                        break
                # conf_dictの中で最も高いscoreを持つOUTを採用し、それ以外のidxはout_idxesに追加
                max_idx = max(conf_dict, key=conf_dict.get)
                for k in conf_dict.keys():
                    if k != max_idx:
                        out_idxes.add(k)

            elif event["frame"] < 15:
                out_idxes.add(idx)

            elif event_label == "THROW IN":
                # THROW INはスコアが0.6以上のものを採用
                if event_score >= 0.6:
                    continue
                else:
                    out_idxes.add(idx)

        get_idxes = get_idxes - out_idxes
        out_idxes2 = set()
        for i, idx in enumerate(sorted(list(get_idxes))):
            if idx in out_idxes2:
                continue
            event = game["events"][idx]
            position = int(event["frame"] / FPS_SN * 1000) * stride
            event_label = event["label"]
            event_score = event["score"]
            # 前後とのPositionの差が600以内かつスコアが0.15未満のPASSは除外
            # ただしPASSとPASSで連続しているときはスコアが低い方を除外
            if event_label == "PASS":
                next_event_list = []
                for j in range(i + 1, len(list(get_idxes))):
                    if j in out_idxes2:
                        continue
                    next_idx = sorted(list(get_idxes))[j]
                    next_event = game["events"][next_idx]
                    next_position = int(next_event["frame"] / FPS_SN * 1000) * stride
                    next_event_label = next_event["label"]
                    next_time_diff = next_position - position
                    if next_time_diff <= 600:
                        next_event_list.append(next_event)
                    else:
                        break
                if game["video"] == "2_09_07":
                    print("event_frame", event["frame"] * stride)
                    print("next_event_list", len(next_event_list))
                for j, next_event in enumerate(next_event_list):
                    next_idx = sorted(list(get_idxes))[i + j + 1]
                    next_position = int(next_event["frame"] / FPS_SN * 1000) * stride
                    next_event_label = next_event["label"]
                    next_time_diff = next_position - position
                    if next_event_label == "PASS":
                        if event_score >= next_event["score"]:
                            out_idxes2.add(next_idx)
                            continue
                        else:
                            out_idxes2.add(idx)
                            continue
                    else:
                        if event_score >= 0.15:
                            continue
                        else:
                            out_idxes2.add(idx)
                            continue

        if game["video"] == "2_09_07":
            frame_list = []
            for i, idx in enumerate(sorted(list(get_idxes - out_idxes2))):
                event = game["events"][idx]
                frame_list.append(event["frame"] * stride)
            print("frame_list", frame_list)
        get_idxes = get_idxes - out_idxes2
        out_idxes2 = set()
        for i, idx in enumerate(sorted(list(get_idxes))):
            if idx in out_idxes2:
                continue
            event = game["events"][idx]
            position = int(event["frame"] / FPS_SN * 1000) * stride
            event_label = event["label"]
            event_score = event["score"]
            # 前後とのPositionの差が600以内かつスコアが0.15未満のPASSは除外
            # ただしPASSとPASSで連続しているときはスコアが低い方を除外
            if event_label == "PASS":
                prev_event_list = []
                for j in range(i - 1, -1, -1):
                    if j in out_idxes2:
                        continue
                    prev_idx = sorted(list(get_idxes))[j]
                    prev_event = game["events"][prev_idx]
                    prev_position = int(prev_event["frame"] / FPS_SN * 1000) * stride
                    prev_event_label = prev_event["label"]
                    prev_time_diff = position - prev_position
                    if prev_time_diff <= 600:
                        prev_event_list.append(prev_event)
                    else:
                        break
                for j, prev_event in enumerate(prev_event_list):
                    prev_idx = sorted(list(get_idxes))[i - j - 1]
                    prev_position = int(prev_event["frame"] / FPS_SN * 1000) * stride
                    prev_event_label = prev_event["label"]
                    prev_time_diff = position - prev_position
                    if event_score >= 0.15:
                        continue
                    else:
                        out_idxes2.add(idx)
                        continue
        if game["video"] == "2_09_07":
            frame_list = []
            for i, idx in enumerate(sorted(list(get_idxes - out_idxes2))):
                event = game["events"][idx]
                frame_list.append(event["frame"] * stride)
            print("frame_list2", frame_list)
        get_idxes = get_idxes - out_idxes2
        out_idxes3 = set()
        for i, idx in enumerate(sorted(list(get_idxes))):
            if idx in out_idxes3:
                continue
            event = game["events"][idx]
            position = int(event["frame"] / FPS_SN * 1000) * stride
            event_label = event["label"]
            event_score = event["score"]
            next_event_list = []
            # 次のイベントまでの時間幅が500以内で連続するまでそのイベントを格納
            for j in range(i + 1, len(list(get_idxes))):
                next_idx = sorted(list(get_idxes))[j]
                if next_idx in out_idxes3:
                    continue
                next_event = game["events"][next_idx]
                next_position = int(next_event["frame"] / FPS_SN * 1000) * stride
                next_event_label = next_event["label"]
                next_time_diff = next_position - position
                if next_time_diff <= 600:
                    next_event_list.append(next_event)
                if event_label == "HIGH PASS":
                    if next_event_label == "HIGH PASS":
                        if next_time_diff <= 1000:
                            next_event_list.append(next_event)
                        else:
                            break
                    else:
                        continue
                else:
                    break
            for j, next_event in enumerate(next_event_list):
                next_idx = sorted(list(get_idxes))[i + j + 1]
                next_position = int(next_event["frame"] / FPS_SN * 1000) * stride
                next_event_label = next_event["label"]
                next_time_diff = next_position - position
                # 600以内で連続している場合，そのイベントがBLOCK,GOAL,SHOT,CROSS,TACKLE以外はスコアが小さいほうを除外
                # ただし，DRIVEとDRIVEが連続している場合は除外しない
                if event_label == "HIGH PASS":
                    if next_event_label == "HIGH PASS":
                        if event_score >= next_event["score"]:
                            out_idxes3.add(next_idx)
                            continue
                        else:
                            out_idxes3.add(idx)
                            continue
                if next_time_diff <= 600:
                    if (
                        event_label == "BALL PLAYER BLOCK"
                        or event_label == "GOAL"
                        or event_label == "SHOT"
                        or event_label == "CROSS"
                        or event_label == "PLAYER SUCCESSFUL TACKLE"
                    ):
                        continue
                    elif event_label == "DRIVE" and next_event_label == "DRIVE":
                        # 両方とも0.15以下なら両方消す
                        if event_score <= 0.15 and next_event["score"] <= 0.15:
                            out_idxes3.add(idx)
                            out_idxes3.add(next_idx)
                            continue
                        else:
                            continue
                    else:
                        if event_score >= next_event["score"]:
                            if (
                                next_event_label == "GOAL"
                                or next_event_label == "SHOT"
                                or next_event_label == "CROSS"
                                or next_event_label == "PLAYER SUCCESSFUL TACKLE"
                                or next_event_label == "BALL PLAYER BLOCK"
                            ):
                                if event_label == "PASS":
                                    out_idxes3.add(idx)
                                    continue
                                continue
                            else:
                                out_idxes3.add(next_idx)
                                continue
                        else:
                            out_idxes3.add(idx)
                            continue
        # Final Filtering
        # 次のイベントまでの時間幅が200以内で連続するときはスコアが高い方を採用
        if game["video"] == "2_09_07":
            frame_list = []
            for i, idx in enumerate(sorted(list(get_idxes - out_idxes3))):
                event = game["events"][idx]
                frame_list.append(event["frame"] * stride)
            print("frame_list3", frame_list)
        get_idxes = get_idxes - out_idxes3
        out_idxes4 = set()
        for i, idx in enumerate(sorted(list(get_idxes))):
            if idx in out_idxes4:
                continue
            event = game["events"][idx]
            position = int(event["frame"] / FPS_SN * 1000) * stride
            event_label = event["label"]
            event_score = event["score"]
            if i != len(list(get_idxes)) - 1:
                next_idx = sorted(list(get_idxes))[i + 1]
                next_event = game["events"][next_idx]
                next_position = int(next_event["frame"] / FPS_SN * 1000) * stride
                next_event_label = next_event["label"]
                next_time_diff = next_position - position
                if next_time_diff <= 200:
                    if event_score >= next_event["score"]:
                        out_idxes4.add(next_idx)
                        continue
                    else:
                        out_idxes4.add(idx)
                        continue
        # Result Making
        get_idxes = get_idxes - out_idxes4
        for i, idx in enumerate(sorted(list(get_idxes))):
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
            os.path.join(pred_game_dir, "results_spotting_my_filtered_action3.json"),
            "w",
        ) as fp:
            json.dump(gameDict, fp, indent=4)
