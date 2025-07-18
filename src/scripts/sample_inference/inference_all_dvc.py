import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from inference_extract_frames import inference_extract
from inference_tdeed import action_spotting
from tqdm import tqdm

sys.path.append(str(Path(__file__).parents[2]))
from team_location_detection.others import constants
from team_location_detection.frame_fetchers import NvDecFrameFetcher
from team_location_detection.predictors import MultiDimStackerPredictor
from team_location_detection.utils import get_best_model_path

# Constants
STRIDE = 1
STRIDE_SN = 12
STRIDE_SNB = 2
TOLERANCES_SNB = [6, 12]
WINDOWS_SNB = [6, 12]
FPS_SN = 25
TTA = False
INDEX_SAVE_ZONE = 1


def parse_arguments():
    parser = argparse.ArgumentParser()
    # For extracting frames
    parser.add_argument(
        "-d",
        "--data_dir",
        default="../../../data/sample",
        help="Path to the sample data",
    )
    parser.add_argument(
        "-v",
        "--video_dir",
        default="../../../data/sample/videos",
        help="Path to the sample videos",
    )
    parser.add_argument(
        "-o",
        "--frame_dir",
        default="../../../data/sample/frames",
        help="Path to write frames. Dry run if None.",
    )
    parser.add_argument(
        "-a",
        "--save_dir",
        default="../../../data/sample/results",
        help="Path to write results. Dry run if None.",
    )
    parser.add_argument("--sample_fps", type=int, default=25)
    parser.add_argument("--recalc_fps", action="store_true")
    parser.add_argument("-j", "--num_workers", type=int, default=os.cpu_count() // 4)

    # For ball-action spotting
    parser.add_argument("--folds", default="train", type=str, required=True)
    parser.add_argument("--gpu_id", default=0, type=int)

    parser.add_argument(
        "--model", type=str, default="SoccerNetBall_challenge2"
    )  # SoccerNetBall_challenge2
    parser.add_argument(
        "-ag", "--acc_grad_iter", type=int, default=1, help="Use gradient accumulation"
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--gsr", action="store_true", help="Get the player names from gsr results"
    )
    parser.add_argument(
        "--save_filename",
        type=str,
        default="results_spotting_my_filtered_integrated",
    )
    return parser.parse_args()


def predict_fold(video_dir, fold, gpu_id: int):
    """
    モデル宣言と動画特定
    """
    experiments = [
        "ball_tuning_location_easy",
        "ball_tuning_location_hard",
        "ball_tuning_team",
    ]
    targets = ["location_easy", "location_hard", "team"]
    for experiment, target in zip(experiments, targets):
        print(f"Predict games: {fold=}, {gpu_id=}")
        print(f"Experiment: {experiment}, {target=}")
        # experimentを_で区切って前から2番目以降を取得
        experiment_dir = (
            constants.experiments_dir + "/" + experiment + f"/fold_{target}_{fold}"
        )
        print("Experiment dir:", experiment_dir)
        model_path = get_best_model_path(experiment_dir)
        print("Model path:", model_path)

        # Model宣言
        predictor = MultiDimStackerPredictor(
            model_path, device=f"cuda:{gpu_id}", tta=TTA
        )
        # データセットの特定
        video_info_path = os.path.join(
            DATA_DIR, "results", "video_info", "video_info.json"
        )
        with open(video_info_path) as file:
            video_infos = json.load(file)

        for video_info in video_infos:
            # 各映像ごとに予測
            video_path = os.path.join(video_dir, f"{video_info['video']}.mp4")
            prediction_dir = os.path.join(
                DATA_DIR,
                "results",
                GAME_NAME,
                video_info["video"],
                experiment.replace("ball_tuning_", ""),
            )
            if not os.path.exists(prediction_dir):
                os.makedirs(prediction_dir)
            else:
                print(f"Folder {prediction_dir} already exists.")
            frame_fetcher = NvDecFrameFetcher(video_path, gpu_id=predictor.device.index)
            results_path = os.path.join(
                DATA_DIR,
                "results",
                GAME_NAME,
                video_info["video"],
                "action",
                "results_spotting_my_filtered_action.json",
            )
            if os.path.exists(results_path):
                with open(results_path, "r") as file:
                    results = json.load(file)
            else:
                raise FileNotFoundError(f"File {results_path} not found.")

            if results["predictions"] == []:
                continue
            key_frame_indexes = [result["frame"] for result in results["predictions"]]

            pred_list = get_predictions(
                predictor, frame_fetcher, key_frame_indexes, video_info, target
            )
            save_path = os.path.join(
                prediction_dir,
                f"results_spotting_my_filtered_{target}.json",
            )
            save_json_snb(pred_list, video_info, save_path, target)


def get_predictions(
    predictor: MultiDimStackerPredictor,
    frame_fetcher: NvDecFrameFetcher,
    key_frame_indexes: list,
    video_info: dict,
    target: str,
):
    """
    Get location for the key frames of the actions
    """
    if target == "location_easy":
        target2class_dict = constants.target2location_easy
    elif target == "location_hard":
        target2class_dict = constants.target2location_hard
    elif target == "team":
        target2class_dict = constants.target2team
    frame_count = video_info["num_frames"]

    pred_list = []

    for key_frame_index in tqdm(key_frame_indexes):
        pred_dict = {}
        indexes_generator = predictor.indexes_generator
        min_frame_index = indexes_generator.clip_index(0, frame_count, INDEX_SAVE_ZONE)
        max_frame_index = indexes_generator.clip_index(
            frame_count, frame_count, INDEX_SAVE_ZONE
        )
        if key_frame_index < min_frame_index or key_frame_index >= max_frame_index:
            continue
        predictor.reset_buffers()
        prediction = predictor.target_predict(
            frame_fetcher=frame_fetcher, key_index=key_frame_index
        )
        probs = prediction.cpu().numpy()
        max_prob = np.max(probs, axis=0)
        predicted_class = target2class_dict[int(np.argmax(probs))]
        pred_dict["frame"] = key_frame_index
        pred_dict["label"] = predicted_class
        pred_dict["score"] = max_prob
        pred_list.append(pred_dict)

    return pred_list


def save_json_snb(pred_list, video_info, save_path, target="location"):
    """
    Save the predictions in the json format
    """

    game = video_info["video"]
    game_dict = {}
    game_dict["UrlLocal"] = game
    game_dict["predictions"] = []
    for pred in pred_list:
        event_dict = {}
        position = int(pred["frame"] / FPS_SN * 1000)
        event_dict["gameTime"] = "{} - {}:{}".format(
            1, position // 60000, int((position % 60000) // 1000)
        )
        event_dict["frame"] = pred["frame"]
        if "location" in target:
            event_dict["location"] = pred["label"]
        elif target == "team":
            event_dict["team"] = pred["label"]
        event_dict["confidence"] = float(pred["score"])
        game_dict["predictions"].append(event_dict)

    with open(save_path, "w") as file:
        json.dump(game_dict, file, indent=4)


def get_name(df_game, frame_num, range_list, team_name):
    frame_dic = {}
    for i in range_list:
        df_frame = df_game[df_game["image_id"] == frame_num + i]
        # さらにteamコラムがteam_nameと一致する行を抽出
        df_frame = df_frame[df_frame["team"] == team_name]
        dic = {}
        # df_frame一つ目の行のuclidが欠損値でない場合
        if not df_frame.empty and df_frame.iloc[0]["uclid"] != "#NUM!":
            df_last = df_frame[df_frame["uclid"] == df_frame["uclid"].min()]
            dic["name"] = "No_name"
            df_name = df_last["name"].to_string(index=False).replace("\n", "")

            if df_name != "NaN":
                dic["name"] = df_name
        else:
            dic["name"] = "No_name"
        frame_dic[frame_num + i] = dic
    return frame_dic


def goal_replacement(match_info, results_path: str):

    with open(results_path, "r") as file:
        results = json.load(file)
    for i, result in enumerate(results["predictions"]):
        if result["action"] == "GOAL":
            if not i == len(results["predictions"]) - 1:
                next_action = results["predictions"][i + 1]["action"]
                if next_action == "BALL PLAYER BLOCK":
                    # gameTime, position, frame以外、入れ替える
                    results["predictions"][i]["action"] = "BALL PLAYER BLOCK"
                    results["predictions"][i]["team"] = results["predictions"][i + 1][
                        "team"
                    ]
                    results["predictions"][i]["team_own_side"] = results["predictions"][
                        i + 1
                    ]["team_own_side"]
                    results["predictions"][i]["location"] = results["predictions"][
                        i + 1
                    ]["location"]
                    results["predictions"][i]["action_confidence"] = results[
                        "predictions"
                    ][i + 1]["action_confidence"]
                    results["predictions"][i]["team_confidence"] = results[
                        "predictions"
                    ][i + 1]["team_confidence"]
                    results["predictions"][i]["easy_confidence"] = results[
                        "predictions"
                    ][i + 1]["easy_confidence"]
                    results["predictions"][i]["hard_confidence"] = results[
                        "predictions"
                    ][i + 1]["hard_confidence"]
                    results["predictions"][i + 1]["action"] = "GOAL"
                    results["predictions"][i + 1]["team"] = results["predictions"][i][
                        "team"
                    ]
                    results["predictions"][i + 1]["team_own_side"] = results[
                        "predictions"
                    ][i]["team_own_side"]
                    results["predictions"][i + 1]["location"] = results["predictions"][
                        i
                    ]["location"]
                    results["predictions"][i + 1]["action_confidence"] = results[
                        "predictions"
                    ][i]["action_confidence"]
                    results["predictions"][i + 1]["team_confidence"] = results[
                        "predictions"
                    ][i]["team_confidence"]
                    results["predictions"][i + 1]["easy_confidence"] = results[
                        "predictions"
                    ][i]["easy_confidence"]
                    results["predictions"][i + 1]["hard_confidence"] = results[
                        "predictions"
                    ][i]["hard_confidence"]

        if result["action"] == "BALL PLAYER BLOCK":
            previous_team_own_side = results["predictions"][i - 1]["team_own_side"]
            if previous_team_own_side == "right":
                results["predictions"][i]["team_own_side"] = "left"
                results["predictions"][i]["team"] = match_info["left"]
            elif previous_team_own_side == "left":
                results["predictions"][i]["team_own_side"] = "right"
                results["predictions"][i]["team"] = match_info["right"]

    with open(results_path, "w") as file:
        json.dump(results, file, indent=4)


if __name__ == "__main__":
    time_start = time.time()
    args = parse_arguments()
    DATA_DIR = Path(args.data_dir)
    LEAGUE_NAME = args.video_dir.split("/")[-2]
    GAME_NAME = args.video_dir.split("/")[-1]
    # inference_extract(args)
    # action_spotting(args)
    # predict_fold(args.video_dir, args.folds, args.gpu_id)
    video_info_path = os.path.join(DATA_DIR, "results", "video_info", "video_info.json")
    game_info_path = os.path.join(DATA_DIR, "results", "video_info", "game_info.json")
    with open(video_info_path) as file:
        video_infos = json.load(file)
    with open(game_info_path) as file:
        game_infos = json.load(file)

    if args.gsr:
        csv_path = os.path.join(
            DATA_DIR, "gsr", "players_in_frames_sn_gamestate_noname.csv"
        )
        df = pd.read_csv(csv_path)
        save_filename = args.save_filename + "_name" + ".json"
    else:
        save_filename = args.save_filename + ".json"

    for game_info in game_infos:
        game = game_info["game"].split("/")[-1]
        for video_info in video_infos:
            video_name = video_info["video"]
            video_half = video_name.split("_")[0]
            half_team_dict = game_info["half"][video_half]
            game_dir = os.path.join(DATA_DIR, "results", GAME_NAME, video_name)
            action_path = os.path.join(
                game_dir, "action", "results_spotting_my_filtered_action.json"
            )
            easy_path = os.path.join(
                game_dir,
                "location_easy",
                "results_spotting_my_filtered_location_easy.json",
            )

            if not os.path.exists(easy_path):
                continue

            hard_path = os.path.join(
                game_dir,
                "location_hard",
                "results_spotting_my_filtered_location_hard.json",
            )
            team_path = os.path.join(
                game_dir, "team", "results_spotting_my_filtered_team.json"
            )
            save_path = os.path.join(game_dir, save_filename)

            out_put_dict = {}
            with open(action_path, "r") as f:
                action_dict = json.load(f)
            if action_dict["predictions"][0]["frame"] < 15:
                action_dict["predictions"] = action_dict["predictions"][1:]
            with open(easy_path, "r") as f:
                easy_dict = json.load(f)
            with open(team_path, "r") as f:
                team_dict = json.load(f)

            out_put_dict["UrlLocal"] = easy_dict["UrlLocal"]
            out_put_dict["predictions"] = []
            with open(hard_path, "r") as f:
                hard_dict = json.load(f)

            for i, (action_pred, easy_pred, hard_pred, team_pred) in enumerate(
                zip(
                    action_dict["predictions"],
                    easy_dict["predictions"],
                    hard_dict["predictions"],
                    team_dict["predictions"],
                )
            ):
                pred_dict = {}
                pred_dict["gameTime"] = easy_pred["gameTime"]
                pred_dict["position"] = action_pred["position"]
                pred_dict["frame"] = easy_pred["frame"]
                pred_dict["action"] = action_pred["action"]
                pred_dict["team_own_side"] = team_pred["team"]
                pred_dict["team"] = half_team_dict[team_pred["team"]]
                if args.gsr:  # teamは求まっているので、nameのみ
                    frame_num = int(action_pred["frame"])
                    frame_num = frame_num + 1  # frame_スタートを1からにする
                    frame_num = frame_num - 4  # 4フレーム前の情報をキーフレームと調整

                    if action_pred["action"] == "OUT":
                        pred_dict["name"] = "OUT"
                        continue
                    elif action_pred["action"] == "PASS":
                        range_list = range(-15, 2)
                    elif action_pred["action"] == "DRIVE":
                        range_list = range(-5, 15)
                    else:
                        range_list = range(-5, 10)
                    frame_info = get_name(df, frame_num, range_list, pred_dict["team"])

                    # すべてのframe_info[i]["name"]がNo_nameの場合、キーフレームから一番近い名前を取得
                    k = 5
                    while all([frame_info[i]["name"] == "No_name" for i in frame_info]):
                        range_list = range(-15 - k, 15 + k)
                        frame_info = get_name(
                            df, frame_num, range_list, pred_dict["team"]
                        )

                        k += 5
                    # frame_infoのnameの中で最も多いvalueを出力
                    name = []
                    for i in frame_info:
                        if frame_info[i]["name"] == "No_name":
                            continue
                        name.append(frame_info[i]["name"])
                    # 最も多い要素を取得
                    if name:
                        name = max(set(name), key=name.count)
                    pred_dict["name"] = name

                # locationの処理
                if easy_pred["location"] == "0":
                    pred_dict["location"] = "OUT"
                else:
                    pred_dict["location"] = (
                        easy_pred["location"] + " " + hard_pred["location"]
                    )
                pred_dict["action_confidence"] = action_pred["action_score"]
                pred_dict["team_confidence"] = team_pred["confidence"]
                pred_dict["easy_confidence"] = easy_pred["confidence"]
                pred_dict["hard_confidence"] = hard_pred["confidence"]
                out_put_dict["predictions"].append(pred_dict)

            print("Save path:", save_path)
            with open(save_path, "w") as f:
                json.dump(out_put_dict, f, indent=4)
            goal_replacement(half_team_dict, save_path)

    print("Finished")
    print("Time:", time.time() - time_start)
