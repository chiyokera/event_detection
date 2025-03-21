from itertools import islice
from pathlib import Path
from typing import Iterable, Optional

import argus
import torch
from kornia.geometry.transform import hflip
from team_location_detection.frame_fetchers import NvDecFrameFetcher
from team_location_detection.frames import get_frames_processor
from team_location_detection.indexes import StackIndexesGenerator


# Iterble(リストやタプルなど)をイテレータに変換(__iter__とか__next__を持つ)
# isliceはイテレータのスライスを取得する関数
# サイズが3の場合、[1, 2, 3, 4, 5, 6, 7, 8, 9] -> (1, 2, 3), (4, 5, 6), (7, 8, 9)
def batched(iterable: Iterable, size: int):
    iterator = iter(iterable)
    while batch := tuple(islice(iterator, size)):
        yield batch


# ここでModel.pathを指定し、実際にInferenceしていく
class MultiDimStackerPredictor:
    def __init__(self, model_path: Path, device: str = "cuda:0", tta: bool = False):
        self.model = argus.load_model(
            model_path, device=device, optimizer=None, loss=None
        )
        self.model.eval()
        self.device = self.model.device
        # ttaとはTest Time Augmentationの略でテスト時にデータ拡張を行うかどうかを指定するフラグ
        # 画像を水平反転させることで、データ拡張を行う
        self.tta = tta
        assert self.model.params["nn_module"][0] == "multidim_stacker"

        # frameを入力することで、入力画像の正規化(÷255)とゼロパディングを行う
        self.frames_processor = get_frames_processor(
            *self.model.params["frames_processor"]
        )
        self.frame_stack_size = self.model.params["frame_stack_size"]  # 15
        self.frame_stack_step = self.model.params["frame_stack_step"]  # 2
        # StackIndexesGeneratorは、周囲30フレームから1つ飛ばしで15フレーム番号を取得
        self.indexes_generator = StackIndexesGenerator(
            self.frame_stack_size, self.frame_stack_step
        )
        # moel_stack_size = 3
        self.model_stack_size = self.model.params["nn_module"][1]["stack_size"]

        self._frame_index2frame: dict[int, torch.Tensor] = dict()
        self._stack_indexes2features: dict[tuple[int], torch.Tensor] = dict()
        # 15フレームない最初のフレームはつかわない(確定14)
        self._predict_offset: int = self.indexes_generator.make_stack_indexes(0)[-1]

    def reset_buffers(self):
        self._frame_index2frame = dict()
        self._stack_indexes2features = dict()

    def _clear_old(self, minimum_index: int):
        for index in list(self._frame_index2frame.keys()):
            if index < minimum_index:
                del self._frame_index2frame[index]
        for stack_indexes in list(self._stack_indexes2features.keys()):
            if any([i < minimum_index for i in stack_indexes]):
                del self._stack_indexes2features[stack_indexes]

    def fetch_frames_for_indexes(self, frame_fetcher, indexes):
        for index in indexes:
            frame_fetcher._current_index = index
            frame = frame_fetcher.fetch_frame()
            # print(f"frame: {frame}")
            frame = frame.to(device=self.model.device)
            # self._frame_index2frame[index] = frame.to(dtype=torch.float32, device=self.model.device)
            self._frame_index2frame[index] = self.frames_processor(
                frame[None, None, ...]
            )[0, 0]

    # 一つのフレームとそのフレーム番号を入力
    @torch.no_grad()
    def predict(
        self, frame: torch.Tensor, index: int
    ) -> tuple[Optional[torch.Tensor], int]:
        frame = frame.to(device=self.model.device)

        # 今までのフレーム番号+今のフレーム番号を保存
        self._frame_index2frame[index] = self.frames_processor(frame[None, None, ...])[
            0, 0
        ]
        # 無視すべきフレームをなくした後のインデックスを取得(最終出力のキーフレーム番号)
        # これが結局、真ん中のキーフレームになっていたので大丈夫
        predict_index = index - self._predict_offset

        # そのフレーム番号を中心とした周囲30フレームから1つ飛ばしで15フレーム番号を取得
        predict_indexes = self.indexes_generator.make_stack_indexes(predict_index)
        # self._clear_old(predict_indexes[0])は、最初のフレーム番号を取得して、
        # それより前のフレーム番号を削除することで、メモリを節約している
        self._clear_old(predict_indexes[0])

        # print(f"set(predict_indexes): {set(predict_indexes)}")
        # print(
        #     f"set(self._frame_index2frame.keys()): {set(self._frame_index2frame.keys())}"
        # )
        # print(f"predict_index: {predict_index}")

        if set(predict_indexes) <= set(self._frame_index2frame.keys()):

            stacks_indexes = list(batched(predict_indexes, self.model_stack_size))

            # stacks_indexesは、15フレームのインデックスを3つずつまとめたリスト
            # stack_indexesは、3つのインデックスをまとめたタプル
            # ここで気になるのが、15フレーム(0.6秒)で必ずアクションを推定するのかである。
            # 実際はここの流れをlocation, teamも並行して行うようにする
            for stack_indexes in stacks_indexes:
                if stack_indexes not in self._stack_indexes2features:
                    frames = torch.stack(
                        [self._frame_index2frame[i] for i in stack_indexes], dim=0
                    )

                    if self.tta:
                        frames = torch.stack([frames, hflip(frames)], dim=0)
                    else:
                        frames = frames.unsqueeze(0)

                    # 画像の特徴量を取得(フレームの各要素は少数になっていることを確認)
                    # print(f"frames.shape: {frames.shape}")
                    # print(f"frames.shape: {frames[0][:10]}")
                    # print(f"frames2.shape: {frames[1][:10]}")
                    features = self.model.nn_module.forward_2d(frames)
                    self._stack_indexes2features[stack_indexes] = features

            features = torch.cat(
                [self._stack_indexes2features[s] for s in stacks_indexes], dim=1
            )

            features = self.model.nn_module.forward_3d(features)
            prediction = self.model.nn_module.forward_head(features)
            # print(f"prediction: {prediction}")
            prediction = self.model.prediction_transform(prediction)
            # print(f"prediction: {prediction}")
            prediction = torch.mean(prediction, dim=0)
            return prediction, predict_index
        else:
            # print("skip")
            return None, predict_index

    @torch.no_grad()
    def target_predict(
        self, frame_fetcher: NvDecFrameFetcher, key_index: int
    ) -> tuple[Optional[torch.Tensor], int]:
        # 無視すべきフレームをなくした後のインデックスを取得(最終出力のキーフレーム番号)
        predict_index = key_index
        # そのフレーム番号を中心とした周囲30フレームから1つ飛ばしで15フレーム番号を取得
        predict_indexes = self.indexes_generator.make_stack_indexes(predict_index)
        # self._index2frame[predict_indexes]にフレームを追加
        frame_fetcher.num_frames = max(predict_indexes)
        # ここで引っかかっている
        frames = frame_fetcher.fetch_frames(predict_indexes)

        for index, frame in zip(predict_indexes, frames):
            frame = frame.to(device=self.model.device)
            self._frame_index2frame[index] = self.frames_processor(
                frame[None, None, ...]
            )[0, 0]
        # self.fetch_frames_for_indexes(frame_fetcher, predict_indexes)

        # self._clear_old(predict_indexes[0])は、最初のフレーム番号を取得して、
        # それより前のフレーム番号を削除することで、メモリを節約している
        self._clear_old(predict_indexes[0])
        # print(f"set(predict_indexes): {set(predict_indexes)}")
        # print(f"predict_index: {predict_index}")
        if set(predict_indexes) <= set(self._frame_index2frame.keys()):
            stacks_indexes = list(batched(predict_indexes, self.model_stack_size))
            # stacks_indexesは、15フレームのインデックスを3つずつまとめたリスト
            # stack_indexesは、3つのインデックスをまとめたタプル
            # ここで気になるのが、15フレーム(0.6秒)で必ずアクションを推定するのかである。
            # 実際はここの流れをlocation, teamも並行して行うようにする
            # print(f"stacks_indexes: {stacks_indexes}")
            for stack_indexes in stacks_indexes:
                if stack_indexes not in self._stack_indexes2features:
                    # print(
                    #     f"self._frame_index2frame: {self._frame_index2frame[stack_indexes[0]]}"
                    # )
                    frames = torch.stack(
                        [self._frame_index2frame[i] for i in stack_indexes], dim=0
                    )

                    # print(f"frames0.shape: {frames.shape}")
                    # ttaがTrueの場合、水平反転した画像も追加(Locationのときなし)
                    if self.tta:
                        frames = torch.stack([frames, hflip(frames)], dim=0)
                    else:
                        frames = frames.unsqueeze(0)

                    # 画像の特徴量を取得
                    # print(f"frames.shape: {frames[0][:10]}")
                    # print(f"frames2.shape: {frames[1][:10]}")
                    features = self.model.nn_module.forward_2d(frames)
                    self._stack_indexes2features[stack_indexes] = features

            features = torch.cat(
                [self._stack_indexes2features[s] for s in stacks_indexes], dim=1
            )
            # print(f"features.shape: {features.shape}")
            features = self.model.nn_module.forward_3d(features)
            prediction = self.model.nn_module.forward_head(features)
            # self.model_transform = nn.Sigmoid
            prediction = self.model.prediction_transform(prediction)
            # print(f"prediction after sigmoid: {prediction}")
            prediction = torch.mean(prediction, dim=0)
            return prediction
        else:
            # print("skip")
            return None
