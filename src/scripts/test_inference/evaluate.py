import argparse
import json
import sys
from pathlib import Path
from os.path import join

import numpy as np

sys.path.append(str(Path(__file__).parents[2]))
# from SoccerNet.Evaluation.ActionSpotting import evaluate
from team_location_detection.others import constants
from team_location_detection.evaluate import evaluate


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True, type=str)
    parser.add_argument("--folds", default="all", type=str)
    parser.add_argument("--challenge", action="store_true")
    parser.add_argument("--target", default="label", type=str)
    parser.add_argument("--metric", default="at1", type=str)
    parser.add_argument("--prediction_file", default="results_spotting_height0.5.json")
    return parser.parse_args()


def evaluate_predictions(
    experiment: str,
    fold: str,
    challenge: bool,
    target: str,
    metric: str,
    prediction_file: str,
):

    if challenge:
        data_split = "challenge"
        games = constants.challenge_games
    else:
        if fold == "train":
            data_split = "test"
            games = constants.fold2games[5] + constants.fold2games[6]
            if target == "label":
                fold = "train"
            elif target == "location":
                fold = "location_train"
            elif target == "team":
                fold = "team_train"
        else:
            data_split = "cv"
            games = constants.fold2games[fold]

    predictions_path = constants.soccernet_dir
    print(f"Evaluate predictions: {experiment=}, {fold=}, {target=}, {metric=}")
    # predictions_path = constants.predictions_dir / experiment / "cv" / f"fold_{fold}"
    print("Predictions path", predictions_path)
    # games = constants.fold2games[fold]
    print("Evaluate games", games)
    if target == "label":
        event_dictionary = constants.class2target
    elif target == "location":
        event_dictionary = constants.location_class2target
    elif target == "team":
        event_dictionary = constants.team_class2target

    results = evaluate(
        SoccerNet_path=str(predictions_path),
        Predictions_path=str(predictions_path),
        list_games=games,
        prediction_file=prediction_file,
        label_files=constants.labels_filename,
        metric=metric,
        version=2,
        framerate=25,
        dataset=None,
        EVENT_DICTIONARY=event_dictionary,
    )

    print(f"Average mAP@{metric}: {results['a_mAP']}")
    print(f"Average mAP@{metric} per event: {results['a_mAP_per_class']}")
    evaluate_results_path = predictions_path / f"{metric}_evaluate_results.json"

    if metric == "tight" or metric == "loose":
        results_save = dict()
        for key, value in results.items():
            key = metric + "_" + key
            if np.isscalar(value):
                results_save[key] = float(value)
            else:
                for value_key, value_value in value.items():
                    new_key = key + "_" + value_key
                    results_save[new_key] = float(value_value)
    else:
        results_save = {
            key: (float(value) if np.isscalar(value) else list(value))
            for key, value in results.items()
        }

    with open(evaluate_results_path, "w") as outfile:
        json.dump(results_save, outfile, indent=4)
    return evaluate_results_path


if __name__ == "__main__":
    args = parse_arguments()

    if args.folds == "train":
        evaluate_predictions(
            args.experiment, args.folds, args.challenge, args.target, args.metric
        )
    else:
        if args.folds == "all":
            folds = constants.folds
        else:
            folds = [int(fold) for fold in args.folds.split(",")]

        for fold in folds:
            evaluate_predictions(args.experiment, fold, args.challenge)
