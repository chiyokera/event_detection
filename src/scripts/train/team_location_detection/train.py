import argparse
import json
import multiprocessing
import sys
import time
from importlib.machinery import SourceFileLoader
from pathlib import Path
from pprint import pprint
import os
import torch
import torch._dynamo
from argus.callbacks import CosineAnnealingLR, LambdaLR, LoggingToCSV, LoggingToFile
from argus.model import load_model

sys.path.append(str(Path(__file__).parents[3]))
from team_location_detection.argus_models import BallActionModel
from team_location_detection.others import constants
from team_location_detection.others.annotations import (
    get_videos_data,
    get_videos_sampling_weights,
)
from team_location_detection.others.augmentations import get_train_augmentations
from team_location_detection.data_loaders import (
    RandomSeekDataLoader,
    SequentialDataLoader,
)
from team_location_detection.datasets import TrainActionDataset, ValActionDataset
from team_location_detection.ema import EmaCheckpoint, ModelEma
from team_location_detection.frames import get_frames_processor
from team_location_detection.indexes import FrameIndexShaker, StackIndexesGenerator
from team_location_detection.metrics import Accuracy, AveragePrecision
from team_location_detection.mixup import TimmMixup
from team_location_detection.target import MaxWindowTargetsProcessor
from team_location_detection.utils import (
    get_best_model_path,
    get_lr,
    load_weights_from_pretrain,
)


def parse_arguments():
    # --experiment変数はball-tuning_xxxに対応
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        required=True,
        choices=[
            "ball_tuning_action",
            "ball_tuning_location",
            "ball_tuning_location_easy",
            "ball_tuning_location_hard",
            "ball_tuning_team",
        ],
        type=str,
    )
    parser.add_argument("--folds", default="train", type=str)
    return parser.parse_args()


def train_ball_action(
    config: dict,
    save_dir: Path,
    train_games: list[str],
    val_games: list[str],
    target: str,
):
    argus_params = config["argus_params"]
    model = BallActionModel(argus_params)
    if "pretrained" in model.params["nn_module"][1]:
        model.params["nn_module"][1]["pretrained"] = False

    # Load pretrained param
    pretrain_dir = ""
    if config["pretrain_ball_experiment"]:
        if "finetune" in config["pretrain_ball_experiment"]:
            pretrain_dir = os.path.join(
                constants.experiments_dir, config["pretrain_ball_experiment"], "fold_5"
            )
        elif "easy" in config["pretrain_ball_experiment"]:
            pretrain_dir = os.path.join(
                constants.experiments_dir,
                config["pretrain_ball_experiment"],
                "fold_location_easy_train",
            )
        elif "hard" in config["pretrain_ball_experiment"]:
            pretrain_dir = os.path.join(
                constants.experiments_dir,
                config["pretrain_ball_experiment"],
                "fold_location_hard_train",
            )
        elif "location" in config["pretrain_ball_experiment"]:
            pretrain_dir = os.path.join(
                constants.experiments_dir,
                config["pretrain_ball_experiment"],
                "fold_location_train",
            )
        if os.path.exists(pretrain_dir):
            print(f"Pretrain dir: {pretrain_dir}")
        else:
            print(f"Pretrain dir {pretrain_dir} not exists")

    if pretrain_dir:
        pretrain_model_path = get_best_model_path(pretrain_dir)
        print(f"Load pretrain model: {pretrain_model_path}")
        pretrain_model = load_model(pretrain_model_path, device=argus_params["device"])
        load_weights_from_pretrain(model.nn_module, pretrain_model.nn_module)
        del pretrain_model

    augmentations = get_train_augmentations(config["image_size"])
    model.augmentations = augmentations

    # MixUpとは、学習の画像データを混ぜ合わせることで、モデルの汎化性能を向上させる手法
    # 今回はどのconfigファイルでもmixup_paramsが設定されていないので、mixupは使われない
    if "mixup_params" in config:
        model.mixup = TimmMixup(**config["mixup_params"])

    # 名前的にラベルを処理するもの
    targets_processor = MaxWindowTargetsProcessor(
        window_size=config["max_targets_window_size"]
    )
    # 入力フレームを正規化しパディングする関数
    frames_processor = get_frames_processor(*argus_params["frames_processor"])

    # frame_stack_sizeは15, frame_stack_stepは2
    indexes_generator = StackIndexesGenerator(
        argus_params["frame_stack_size"],
        argus_params["frame_stack_step"],
    )
    frame_index_shaker = FrameIndexShaker(**config["frame_index_shaker"])

    # トレーニングのスキームとしてパラーメータの移動平均を求めて、効率的に重みを更新する
    print("EMA decay:", config["ema_decay"])
    model.model_ema = ModelEma(model.nn_module, decay=config["ema_decay"])

    # model.nn_moduleをコンパイルする
    if "torch_compile" in config:
        print("torch.compile:", config["torch_compile"])
        torch._dynamo.reset()
        model.nn_module = torch.compile(model.nn_module, **config["torch_compile"])

    device = torch.device(argus_params["device"][0])

    # train_dataは各試合のビデオパスとアノテーションを取得
    # ここでtargetはlabel, location, location_easy, teamのいずれか
    train_data = get_videos_data(games=train_games, target=target)

    # 各ビデオのフレームの初期重みを取得、ここで事前学習されたモデルの重みを考慮している
    videos_sampling_weights = get_videos_sampling_weights(
        train_data,
        **config["train_sampling_weights"],
    )
    if target == "label":
        classes = constants.classes
    elif target == "location":
        classes = constants.location_classes
    elif target == "location_easy":
        classes = constants.location_easy_classes
    elif target == "location_hard":
        classes = constants.location_hard_classes
    elif target == "team":
        classes = constants.team_classes
    # trainデータのみ、sampling_weightsを考慮してデータセットを作成
    train_dataset = TrainActionDataset(
        videos_data=train_data,
        classes=classes,
        indexes_generator=indexes_generator,
        epoch_size=config["train_epoch_size"],
        videos_sampling_weights=videos_sampling_weights,
        target_process_fn=targets_processor,
        frames_process_fn=frames_processor,
        frame_index_shaker=frame_index_shaker,
    )
    print(f"Train dataset len {len(train_dataset)}")
    val_data = get_videos_data(val_games, add_empty_actions=True, target=target)
    val_dataset = ValActionDataset(
        videos_data=val_data,
        classes=classes,
        indexes_generator=indexes_generator,
        target_process_fn=targets_processor,
        frames_process_fn=frames_processor,
    )
    print(f"Val dataset len {len(val_dataset)}")
    train_loader = RandomSeekDataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        num_nvdec_workers=config["num_nvdec_workers"],
        num_opencv_workers=config["num_opencv_workers"],
        gpu_id=device.index,
    )
    val_loader = SequentialDataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        frame_buffer_size=argus_params["frame_stack_size"]
        * argus_params["frame_stack_step"],
        gpu_id=device.index,
    )

    # num_epochs=[7, 35], stages=["warmup", "train"]、あわせて42エポックで訓練
    for num_epochs, stage in zip(config["num_epochs"], config["stages"]):
        callbacks = [
            LoggingToFile(os.path.join(save_dir, "log2.txt"), append=True),
            LoggingToCSV(os.path.join(save_dir, "log2.csv"), append=True),
        ]

        # num_iteration = アクションの総数 // 2 * num_epochs(7 or 3)
        num_iterations = (len(train_dataset) // config["batch_size"]) * num_epochs
        # warmupの場合は、学習率をnum_iterationsで割ることで、学習率を徐々に上げていく
        if stage == "warmup":
            # LambdaLRは学習率を変更するためのクラス
            # step_on_iteration=Trueで、学習率をiterationごとに変更する
            callbacks += [
                LambdaLR(lambda x: x / num_iterations, step_on_iteration=True),
            ]

            # 学習開始、学習率が小さい(ちょっとずつ探索)から大きく
            # (最初は初期の局所最適、そっから抜け出すになることを防ぐ)
            model.fit(train_loader, num_epochs=num_epochs, callbacks=callbacks)

        elif stage == "train":
            checkpoint_format = "model-{epoch:03d}-{val_average_precision:.6f}.pth"
            # Emaはモデルの重みをスムーズに更新していく。
            # EmaCheckpointはモデルの重みを保存するクラス
            # max_saves=1で、最新のモデルのみ保存、古いモデルは削除
            # CosineAnnealingLRは学習率をコサイン関数に基づいて調整するクラス
            # つまり最初は学習率が大きく、徐々に小さくしていく（最初は大まかに、だんだん細かく）
            # 後半の方は、モデルが過度にデータセットに適合しないようにするため、学習率を小さくする
            callbacks += [
                EmaCheckpoint(save_dir, file_format=checkpoint_format, max_saves=30),
                CosineAnnealingLR(
                    T_max=num_iterations,
                    eta_min=get_lr(config["min_base_lr"], config["batch_size"]),
                    step_on_iteration=True,
                ),
            ]

            metrics = [
                AveragePrecision(classes),
                Accuracy(classes, threshold=config["metric_accuracy_threshold"]),
            ]

            model.fit(
                train_loader,
                val_loader=val_loader,
                num_epochs=num_epochs,
                callbacks=callbacks,
                metrics=metrics,
            )

    train_loader.stop_workers()
    val_loader.stop_workers()


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    args = parse_arguments()
    target = "_".join(args.experiment.split("_")[2:])
    args.target = target
    print("Experiment:", args.experiment)
    print("Target:", args.target)
    # ここでconfigフォルダの中で--experimentで設定したファイルを読み込む
    config_path = os.path.join(constants.configs_dir, f"{args.experiment}.py")
    if not os.path.exists(config_path):
        raise ValueError(f"Config file '{config_path}' not exists")

    # data/ball-action/experiments/ball-tuning-xxx/config.jsonに対応
    config = SourceFileLoader(args.experiment, str(config_path)).load_module().config
    print("Experiment config:")
    pprint(config, sort_dicts=False)
    # experiments_dirはdata/ball_action/experiments/ball-tuning-xxxに対応
    experiments_dir = os.path.join(constants.experiments_dir, args.experiment)
    print("Experiment dir:", experiments_dir)
    if not os.path.exists(experiments_dir):
        os.makedirs(experiments_dir)
    else:
        print(f"Folder '{experiments_dir}' already exists.")

    # ここで現在のファイルを模したtrain.pyとconfig.jsonを作成している
    train_basename = "train.py"
    config_basename = "config.json"
    if args.target == "location":
        train_basename = "train_location.py"
        config_basename = "config_location_finetune_easy.json"
    elif args.target == "team":
        train_basename = "train_team.py"
        config_basename = "config_team.json"

    with open(os.path.join(experiments_dir, train_basename), "w") as outfile:
        outfile.write(open(__file__).read())

    with open(os.path.join(experiments_dir, config_basename), "w") as outfile:
        json.dump(config, outfile, indent=4)

    if args.folds == "train":
        train_folds = [0, 1, 2, 3, 4]
        val_folds = [5, 6]
        train_games = []
        val_games = constants.fold2games[5] + constants.fold2games[6]

        for train_fold in train_folds:
            train_games += constants.fold2games[train_fold]

        fold_experiment_dir = os.path.join(experiments_dir, "fold_train")
        if "location" in args.target:
            fold_experiment_dir = os.path.join(
                experiments_dir, f"fold_{args.target}_train"
            )
        elif args.target == "team":
            fold_experiment_dir = os.path.join(experiments_dir, "fold_team_train")

        print(f"Val folds: {val_folds}, train folds: {train_folds}")
        print(f"Val games: {val_games}, train games: {train_games}")
        print(f"Fold experiment dir: {fold_experiment_dir}")
        train_ball_action(
            config=config,
            save_dir=fold_experiment_dir,
            train_games=train_games,
            val_games=val_games,
            target=args.target,
        )

        torch._dynamo.reset()
        torch.cuda.empty_cache()
        time.sleep(12)

    else:
        if args.folds == "all":
            folds = constants.folds
        else:
            folds = [int(fold) for fold in args.folds.split(",")]

        for fold in folds:
            # 各foldでそのfold以外のデータセットで学習してそれをtestに用いる。valに設定してループ
            train_folds = list(set(constants.folds) - {fold})
            val_games = constants.fold2games[fold]
            train_games = []
            for train_fold in train_folds:
                train_games += constants.fold2games[train_fold]
            fold_experiment_dir = os.path.join(experiments_dir, f"fold_{fold}")
            if args.target == "location":
                fold_experiment_dir = os.path.join(
                    experiments_dir, f"fold_location_{fold}"
                )
            elif args.target == "team":
                fold_experiment_dir = os.path.join(experiments_dir, f"fold_team_{fold}")

            print(f"Val fold: {fold}, train folds: {train_folds}")
            print(f"Val games: {val_games}, train games: {train_games}")
            print(f"Fold experiment dir: {fold_experiment_dir}")
            # train_ball_action関数の引数はconfig, save_dir, train_games, val_games
            train_ball_action(
                config=config,
                save_dir=fold_experiment_dir,
                train_games=train_games,
                val_games=val_games,
                target=args.target,
            )

            torch._dynamo.reset()
            torch.cuda.empty_cache()
            time.sleep(12)
