import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.append(str(Path(__file__).parents[2]))
from scripts.test_inference.test_inference_extract_frames import test_inference_extract
from scripts.test_inference.test_inference_tdeed import test_action_spotting
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

DATA_DIR = Path("../../../data/team_location_detection/soccernet/england_efl/2019-2020")
INDEX_SAVE_ZONE = 1


def parse_arguments():
    parser = argparse.ArgumentParser()
    # For extracting frames
    parser.add_argument(
        "--video_dir",
        default="../../../data/team_location_detection/soccernet/england_efl/2019-2020",
        help="Path to the soccernet data directory",
    )
    parser.add_argument("--sample_fps", type=int, default=25)
    parser.add_argument("--recalc_fps", action="store_true")
    parser.add_argument("-j", "--num_workers", type=int, default=os.cpu_count() // 4)

    # For ball-action spotting
    parser.add_argument("--folds", default="train", type=str)
    parser.add_argument("--gpu_id", default=0, type=int)

    parser.add_argument(
        "--model", type=str, default="SoccerNetBall_test"
    )  # SoccerNetBall_challenge2
    parser.add_argument(
        "-ag", "--acc_grad_iter", type=int, default=1, help="Use gradient accumulation"
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--save_filename",
        type=str,
        default="results_spotting_my_filtered_integrated",
    )
    return parser.parse_args()


def predict_fold(fold, gpu_id: int):
    """
    モデル宣言と動画特定
    """
    experiments = [
        "ball_tuning_location_easy",
        "ball_tuning_location_hard",
        "ball_tuning_team",
    ]
    targets = ["location_easy", "location_hard", "location", "team"]
    for experiment, target in zip(experiments, targets):
        print(f"Predict games: {fold=}, {gpu_id=}")
        print(f"Experiment: {experiment}, {target=}")
        # experimentを_で区切って前から2番目以降を取得

        experiment_dir = Path(
            "../../../data/team_location_detection/experiments"
            + f"/{experiment}"
            + f"/fold_{target}_{fold}",
        )
        print("Experiment dir:", experiment_dir)
        model_path = get_best_model_path(experiment_dir)
        print("Model path:", model_path)

        # Model宣言
        predictor = MultiDimStackerPredictor(
            model_path, device=f"cuda:{gpu_id}", tta=TTA
        )
        # データセットの特定
        video_info_path = os.path.join(DATA_DIR, "test_video_info.json")
        with open(video_info_path) as file:
            video_infos = json.load(file)

        for video_info in video_infos:
            # 各映像ごとに予測
            video_path = os.path.join(DATA_DIR, video_info["video"], "720p.mp4")
            prediction_dir = os.path.join(
                DATA_DIR,
                video_info["video"],
                "train",
                experiment.replace("ball_tuning_", ""),
            )
            if not os.path.exists(prediction_dir):
                os.makedirs(prediction_dir)
            else:
                print(f"Folder {prediction_dir} already exists.")
            frame_fetcher = NvDecFrameFetcher(video_path, gpu_id=predictor.device.index)
            results_path = os.path.join(
                DATA_DIR,
                video_info["video"],
                "train",
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


def goal_replacement(results_path: str):

    with open(results_path, "r") as file:
        results = json.load(file)
    for i, result in enumerate(results["predictions"]):
        if result["action"] == "GOAL":
            if not i == len(results["predictions"]) - 1:
                next_action = results["predictions"][i + 1]["action"]
                if next_action == "BALL PLAYER BLOCK":
                    # gameTime, position, frame以外、入れ替える
                    results_i_dict = results["predictions"][i].copy()
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
                    results["predictions"][i + 1]["team"] = results_i_dict["team"]
                    results["predictions"][i + 1]["team_own_side"] = results_i_dict[
                        "team_own_side"
                    ]
                    results["predictions"][i + 1]["location"] = results_i_dict[
                        "location"
                    ]
                    results["predictions"][i + 1]["action_confidence"] = results_i_dict[
                        "action_confidence"
                    ]
                    results["predictions"][i + 1]["team_confidence"] = results_i_dict[
                        "team_confidence"
                    ]
                    results["predictions"][i + 1]["easy_confidence"] = results_i_dict[
                        "easy_confidence"
                    ]
                    results["predictions"][i + 1]["hard_confidence"] = results_i_dict[
                        "hard_confidence"
                    ]

    with open(results_path, "w") as file:
        json.dump(results, file, indent=4)


if __name__ == "__main__":
    time_start = time.time()
    args = parse_arguments()
    test_inference_extract(args)
    test_action_spotting(args)
    predict_fold(args.folds, args.gpu_id)
    match_info_path = os.path.join(DATA_DIR, "spotting-2024-right_or_left.json")
    test_game = [
        "2019-10-01 - Reading - Fulham",
        "2019-10-01 - Stoke City - Huddersfield Town",
    ]
    with open(match_info_path) as file:
        match_infos = json.load(file)

    save_filename = args.save_filename + ".json"

    for match_info in match_infos["matches"]:
        if match_info["game"] not in test_game:
            continue
        game = match_info["game"]
        game_dir = os.path.join(DATA_DIR, game)
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
        for i in range(len(action_dict["predictions"])):
            if action_dict["predictions"][i]["frame"] >= 15:
                start_idx = i
                break
        action_dict["predictions"] = action_dict["predictions"][start_idx:]
        with open(easy_path, "r") as f:
            easy_dict = json.load(f)
        with open(team_path, "r") as f:
            team_dict = json.load(f)

        out_put_dict["UrlLocal"] = easy_dict["UrlLocal"]
        out_put_dict["predictions"] = []
        with open(hard_path, "r") as f:
            hard_dict = json.load(f)

        assert len(action_dict["predictions"]) == len(
            easy_dict["predictions"]
        ), "action and easy length mismatch"

        for i, (action_pred, easy_pred, hard_pred, team_pred) in enumerate(
            zip(
                action_dict["predictions"],
                easy_dict["predictions"],
                hard_dict["predictions"],
                team_dict["predictions"],
            )
        ):

            pred_dict = {}
            frame = action_pred["frame"]
            pred_dict["gameTime"] = easy_pred["gameTime"]
            pred_dict["position"] = action_pred["position"]
            pred_dict["frame"] = easy_pred["frame"]
            pred_dict["action"] = action_pred["action"]
            pred_dict["team_own_side"] = team_pred["team"]
            if frame < match_info["second_start"]:
                pred_dict["team"] = match_info[
                    match_info["halves"]["1st"][team_pred["team"]]
                ]
            else:
                pred_dict["team"] = match_info[
                    match_info["halves"]["2nd"][team_pred["team"]]
                ]
            if pred_dict["action"] == "BALL PLAYER BLOCK":
                previous_team_own_side = team_dict["predictions"][i - 1]["team"]
                if previous_team_own_side == "right":
                    pred_dict["team_own_side"] = "left"
                    if frame < match_info["second_start"]:
                        pred_dict["team"] = match_info[
                            match_info["halves"]["1st"]["left"]
                        ]
                    else:
                        pred_dict["team"] = match_info[
                            match_info["halves"]["2nd"]["left"]
                        ]
                elif previous_team_own_side == "left":
                    pred_dict["team_own_side"] = "right"
                    if frame < match_info["second_start"]:
                        pred_dict["team"] = match_info[
                            match_info["halves"]["1st"]["right"]
                        ]
                    else:
                        pred_dict["team"] = match_info[
                            match_info["halves"]["2nd"]["right"]
                        ]

            if pred_dict["action"] == "SHOT":
                previous_team_own_side = team_dict["predictions"][i - 1]["team"]
                if previous_team_own_side == "right":
                    pred_dict["team_own_side"] = "right"
                    if frame < match_info["second_start"]:
                        pred_dict["team"] = match_info[
                            match_info["halves"]["1st"]["right"]
                        ]
                    else:
                        pred_dict["team"] = match_info[
                            match_info["halves"]["2nd"]["right"]
                        ]
                elif previous_team_own_side == "left":
                    pred_dict["team_own_side"] = "left"
                    if frame < match_info["second_start"]:
                        pred_dict["team"] = match_info[
                            match_info["halves"]["1st"]["left"]
                        ]
                    else:
                        pred_dict["team"] = match_info[
                            match_info["halves"]["2nd"]["left"]
                        ]

            # locationの処理
            if easy_pred["location"] == "0":
                pred_dict["location"] = "OUT"
                pred_dict["team"] = "OUT"
                pred_dict["action"] = "OUT"
                pred_dict["team_own_side"] = "OUT"
                pred_dict["name"] = "OUT"
            else:
                pred_dict["location"] = (
                    easy_pred["location"] + " " + hard_pred["location"]
                )
                if pred_dict["action"] == "OUT":
                    pred_dict["location"] = "OUT"
            pred_dict["action_confidence"] = action_pred["action_score"]
            pred_dict["team_confidence"] = team_pred["confidence"]
            pred_dict["easy_confidence"] = easy_pred["confidence"]
            pred_dict["hard_confidence"] = hard_pred["confidence"]
            out_put_dict["predictions"].append(pred_dict)

        out_idxes = []
        for i in range(len(out_put_dict["predictions"])):
            if i != len(out_put_dict["predictions"]) - 1:
                if i == 0 and out_put_dict["predictions"][i]["action"] == "OUT":
                    out_idxes.append(i)
                if (
                    out_put_dict["predictions"][i]["action"] == "OUT"
                    and out_put_dict["predictions"][i + 1]["action"] == "OUT"
                ):
                    out_idxes.append(i + 1)
                if (
                    out_put_dict["predictions"][i]["action"] == "CROSS"
                    and out_put_dict["predictions"][i + 1]["action"] == "CROSS"
                ):
                    if (
                        out_put_dict["predictions"][i]["action_confidence"]
                        >= out_put_dict["predictions"][i + 1]["action_confidence"]
                    ):
                        out_idxes.append(i + 1)
                    else:
                        out_idxes.append(i)
        out_put_dict["predictions"] = [
            pred
            for i, pred in enumerate(out_put_dict["predictions"])
            if i not in out_idxes
        ]

        print("Save path:", save_path)
        with open(save_path, "w") as f:
            json.dump(out_put_dict, f, indent=4)
        goal_replacement(save_path)

    print("Finished")
    print("Time:", time.time() - time_start)
