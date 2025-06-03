# action特定から、location、team特定全てを行い、終わるまでの時間を測る
import argparse
import json
import os
import sys
import time
from pathlib import Path
import numpy as np
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
    parser.add_argument(
        "--data_dir",
        default="../../../data/team_location_detection/soccernet/england_efl/2019-2020",
        help="Path to the soccernet data directory",
    )
    parser.add_argument("--folds", default="train", choices=["train", "all"], type=str)
    parser.add_argument("--gpu_id", default=0, type=int)
    parser.add_argument(
        "--save_filename",
        type=str,
        default="results_spotting_eval_integrated2",
    )
    return parser.parse_args()


def predict_fold(fold: str, gpu_id: int, data_dir: str):
    experiments = [
        "ball_tuning_location_easy",
        "ball_tuning_location_hard",
        "ball_tuning_location",
        "ball_tuning_team",
    ]
    targets = ["location_easy", "location_hard", "location", "team"]
    for experiment, target in zip(experiments, targets):
        print(f"Predict games: {fold=}, {gpu_id=}")
        # experimentを_で区切って前から2番目以降を取得
        experiment_dir = Path(
            "../../../data/team_location_detection/experiments"
            + f"/{experiment}"
            + f"/fold_{target}_{fold}",
        )
        print("Experiment dir:", experiment_dir)
        model_path = get_best_model_path(experiment_dir)
        print("Model path:", model_path)

        # Model declaration
        predictor = MultiDimStackerPredictor(
            model_path, device=f"cuda:{gpu_id}", tta=TTA
        )
        # Data declaration
        video_info_path = os.path.join(data_dir, "test_video_info.json")
        with open(video_info_path) as file:
            video_infos = json.load(file)

        for video_info in video_infos:
            video_path = os.path.join(data_dir, video_info["video"], "720p.mp4")
            prediction_dir = os.path.join(data_dir, video_info["video"], fold)

            if not os.path.exists(prediction_dir):
                os.makedirs(prediction_dir)
            else:
                print(f"Folder {prediction_dir} already exists.")
            frame_fetcher = NvDecFrameFetcher(video_path, gpu_id=predictor.device.index)

            # results_path = os.path.join(
            #     prediction_dir,
            #     "results_spotting_my_filtered_v2.json",
            # )
            # if os.path.exists(results_path):
            #     with open(results_path) as file:
            #         results = json.load(file)
            # key_frame_indexes = [result["frame"] for result in results["predictions"]]
            key_frame_path = os.path.join(
                data_dir, video_info["video"], "Labels-ball-location-team.json"
            )
            if os.path.exists(key_frame_path):
                with open(key_frame_path) as file:
                    results = json.load(file)

            key_frame_indexes = []
            for result in results["annotations"]:
                position = result["position"]
                frame = int((int(position) / 1000) * FPS_SN)
                key_frame_indexes.append(frame)

            # print(f"key_frame_indexes: {key_frame_indexes}")
            pred_list = get_predictions(
                predictor, frame_fetcher, key_frame_indexes, video_info, target
            )
            save_path = os.path.join(
                prediction_dir,
                f"results_spotting_eval_{target}.json",
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
    elif target == "location":
        target2class_dict = constants.target2location
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


if __name__ == "__main__":
    time_start = time.time()
    args = parse_arguments()
    predict_fold(args.folds, args.gpu_id, args.data_dir)
    match_info_path = "../../../data/team_location_detection/soccernet/england_efl/2019-2020/test_video_info.json"
    with open(match_info_path) as file:
        match_infos = json.load(file)
    save_filename = args.save_filename + ".json"
    for match_info in match_infos:
        game = match_info["video"]
        game_dir = os.path.join(args.data_dir, game, args.folds)
        easy_path = os.path.join(
            game_dir,
            "results_spotting_eval_location_easy.json",
        )
        hard_path = os.path.join(
            game_dir,
            "results_spotting_eval_location_hard.json",
        )
        loc_path = os.path.join(
            game_dir,
            "results_spotting_eval_location.json",
        )
        team_path = os.path.join(game_dir, "results_spotting_eval_team.json")
        save_path = os.path.join(game_dir, save_filename)
        if os.path.exists(save_path):
            continue
        out_put_dict = {}
        with open(easy_path, "r") as f:
            easy_dict = json.load(f)
        with open(hard_path, "r") as f:
            hard_dict = json.load(f)
        with open(loc_path, "r") as f:
            loc_dict = json.load(f)
        with open(team_path, "r") as f:
            team_dict = json.load(f)

        out_put_dict["UrlLocal"] = easy_dict["UrlLocal"]
        out_put_dict["predictions"] = []
        for easy_pred, hard_pred, loc_pred, team_pred in zip(
            easy_dict["predictions"],
            hard_dict["predictions"],
            loc_dict["predictions"],
            team_dict["predictions"],
        ):
            pred_dict = {}
            pred_dict["gameTime"] = easy_pred["gameTime"]
            pred_dict["position"] = int(easy_pred["frame"] * 1000 / FPS_SN)
            pred_dict["frame"] = easy_pred["frame"]
            pred_dict["team"] = team_pred["team"]

            # locationの処理
            if easy_pred["location"] == "0":
                pred_dict["location"] = "0"
            else:
                pred_dict["location"] = (
                    easy_pred["location"] + " " + hard_pred["location"]
                )
            pred_dict["location2"] = loc_pred["location"]
            pred_dict["team_confidence"] = team_pred["confidence"]
            pred_dict["easy_confidence"] = easy_pred["confidence"]
            pred_dict["hard_confidence"] = hard_pred["confidence"]
            pred_dict["location2_confidence"] = loc_pred["confidence"]
            out_put_dict["predictions"].append(pred_dict)

        with open(save_path, "w") as f:
            json.dump(out_put_dict, f, indent=4)

    print("Finished")
    print("Time:", time.time() - time_start)
