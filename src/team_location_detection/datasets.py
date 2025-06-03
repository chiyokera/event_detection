import abc
import random
from typing import Callable, Optional, Type

import numpy as np
import torch
from team_location_detection.frame_fetchers import (
    AbstractFrameFetcher,
    NvDecFrameFetcher,
)
from team_location_detection.indexes import FrameIndexShaker, StackIndexesGenerator
from team_location_detection.target import VideoTarget
from team_location_detection.utils import set_random_seed


class ActionDataset(metaclass=abc.ABCMeta):
    def __init__(
        self,
        videos_data: list[dict],
        classes: list[str],
        indexes_generator: StackIndexesGenerator,
        target_process_fn: Callable[[np.ndarray], torch.Tensor],
        frames_process_fn: Callable[[torch.Tensor], torch.Tensor],
    ):
        self.indexes_generator = indexes_generator
        self.frames_process_fn = frames_process_fn
        self.target_process_fn = target_process_fn

        self.videos_data = videos_data
        self.num_videos = len(self.videos_data)
        self.num_videos_actions = [
            len(v["frame_index2action"]) for v in self.videos_data
        ]
        self.num_actions = sum(self.num_videos_actions)

        # self.videos_target[video_index].targets(frame_indexes)でアノテーションを取得
        # フレーム番号を入力すると、そのフレームのアクションに対応するインデックスのみ1.0のベクトルを返す
        # すなわちこれがアノテーションラベルとなりうる
        self.videos_target = [VideoTarget(data, classes) for data in self.videos_data]

    # イベントの長さを返す
    def __len__(self) -> int:
        return self.num_actions

    # イベントのインデックスを入力すると、そのイベントの試合番号と
    # そのイベントに関するフレーム番号を返す
    @abc.abstractmethod
    def get_video_frame_indexes(self, index: int) -> tuple[int, list[int]]:
        pass

    # 特定の試合ビデオのインデックスと、フレームの全番号を入力し、その試合のアノテーションリストを取得
    def get_targets(self, video_index: int, frame_indexes: list[int]):
        # フレーム番号は1ずつ増えるので大丈夫、フレーム番号の入力でアノテーションを取得
        target_indexes = list(range(min(frame_indexes), max(frame_indexes) + 1))
        targets = self.videos_target[video_index].targets(target_indexes)
        return targets  # shape: (len(frame_indexes), num_classes)

    def get_frames_targets(
        self,
        video_index: int,
        frame_indexes: list[int],
        frame_fetcher: AbstractFrameFetcher,
    ) -> tuple[torch.Tensor, np.ndarray]:
        frames = frame_fetcher.fetch_frames(frame_indexes)
        targets = self.get_targets(video_index, frame_indexes)
        return frames, targets

    def get_frame_fetcher(
        self,
        video_index: int,
        frame_fetcher_class: Type[AbstractFrameFetcher],
        gpu_id: int = 0,
    ):
        video_data = self.videos_data[video_index]
        frame_fetcher = frame_fetcher_class(video_data["video_path"], gpu_id=gpu_id)
        frame_fetcher.num_frames = video_data["frame_count"]
        return frame_fetcher

    def process_frames_targets(self, frames: torch.Tensor, targets: np.ndarray):
        input_tensor = self.frames_process_fn(frames)
        target_tensor = self.target_process_fn(targets)
        return input_tensor, target_tensor

    # DataLoaderの__iter__メソッドで呼び出される(random_seek.py)
    def get(
        self,
        index: int,
        frame_fetcher_class: Type[AbstractFrameFetcher] = NvDecFrameFetcher,
        gpu_id: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        # indexは、__len__に対応しているから訓練データのアクションの数分、ちょうどいい
        # 訓練の場合は、ランダムに選ばれた試合の、ランダムに選ばれたフレームの周囲15フレーム(1個飛ばしで1.2秒分)
        # 検証の場合は、1試合ぶんの映像データから、actionがついたフレームを上から順に取得し、
        # 周囲フレームとラベルを出力するようにする
        video_index, frame_indexes = self.get_video_frame_indexes(index)
        # 試合のビデオ情報を獲得し、いつでもフレームのマップを取得できるようにする
        frame_fetcher = self.get_frame_fetcher(video_index, frame_fetcher_class, gpu_id)
        # frameマップと、各フレームのアノテーションリスト(もちろん全要素0もある)をまとめたリストを取得
        frames, targets = self.get_frames_targets(
            video_index, frame_indexes, frame_fetcher
        )
        # 取得したframeマップとtargetsリストの処理を行う
        # frameマップは正規化、targetsは1.0がある行をtensorで取得
        return self.process_frames_targets(
            frames, targets
        )  # shape: (15, 3, 736, 1280), (num_classes)


class TrainActionDataset(ActionDataset):
    def __init__(
        self,
        videos_data: list[dict],
        classes: list[str],
        indexes_generator: StackIndexesGenerator,
        epoch_size: int,
        videos_sampling_weights: list[np.ndarray],
        target_process_fn: Callable[[np.ndarray], torch.Tensor],
        frames_process_fn: Callable[[torch.Tensor], torch.Tensor],
        frame_index_shaker: Optional[FrameIndexShaker] = None,
    ):
        super().__init__(
            videos_data=videos_data,
            classes=classes,
            indexes_generator=indexes_generator,
            target_process_fn=target_process_fn,
            frames_process_fn=frames_process_fn,
        )
        self.num_labels = sum([len(v["frame_index2action"]) for v in videos_data])
        self.frame_index_shaker = frame_index_shaker

        self.videos_sampling_weights = videos_sampling_weights

        # vは特定の試合のデータ
        # 各試合ごとのフレーム数のリストを作成
        self.videos_frame_indexes = [np.arange(v["frame_count"]) for v in videos_data]

    # epoch単位でデータセットの長さを返す
    def __len__(self) -> int:
        return self.num_labels

    # ある整数を入力する、これはrandom.seedの値になるため、これを変えれば、randomが被らない
    # ランダムに試合番号(VideoIndex)を出力し、
    # さらにその試合において、アクションがある確率リスト(videos_sampling_weights)から1つのフレームをランダムに選ぶ
    # その試合のイベント番号の1つ飛ばし周囲15フレーム(1.2秒分)を取ってくる
    def get_video_frame_indexes(self, index) -> tuple[int, list[int]]:
        set_random_seed(index)
        # random.randrange(0, self.num_videos)で0からself.num_videos-1までの整数をランダムに取得
        video_index = random.randrange(0, self.num_videos)

        # 特定の試合のフレーム数のリストの中から、重み付けされた確率で1つのフレームを選択
        # もちろん、アノテーションにちかいフレームが選択されやすい周囲9フレーム内とか
        frame_index = np.random.choice(
            self.videos_frame_indexes[video_index],
            p=self.videos_sampling_weights[video_index],
        )

        save_zone = 1
        # Noneじゃないなら、シェイクの最大値を保存領域に追加
        if self.frame_index_shaker is not None:
            # shifts: [-1, 0, 1]より、abs(sh)の最大値は1、よってsave_zoneは2
            save_zone += max(abs(sh) for sh in self.frame_index_shaker.shifts)

        # frame_indexがフレーム数を超えないようにクリップ
        # 例えば選んだフレームがたまたまはじめ、もしくは後ろの方にある場合、
        # 周囲15フレームを取得することができない
        # ぎりぎり収まる範囲内でframe_indexを調整
        frame_index = self.indexes_generator.clip_index(
            frame_index, self.videos_data[video_index]["frame_count"], save_zone
        )
        # frame_indexを基準に周囲のフレームを取得,前後7フレームずつ1フレーム飛ばし
        frame_indexes = self.indexes_generator.make_stack_indexes(frame_index)

        # shakerは25%の確率でフレーム１つずつ1つ前、1つ後ろ、そのままに変換
        # 1こ飛ばしのため、フレームが右に行って減ったりしない
        # 例えば、frame_indexes=[2, 4, 6, 8, 10]のとき可能性として[1, 4, 6, 8, 11]とかがあり得る
        if self.frame_index_shaker is not None:
            frame_indexes = self.frame_index_shaker(frame_indexes)
        return video_index, frame_indexes


class ValActionDataset(ActionDataset):
    def get_video_frame_indexes(self, index: int) -> tuple[int, list[int]]:
        assert 0 <= index < self.__len__()
        action_index = index
        # Validationデータセットは1試合のみなので、video_indexは0
        video_index = 0
        for video_index, num_video_actions in enumerate(self.num_videos_actions):
            if action_index >= num_video_actions:
                action_index -= num_video_actions
            else:
                break

        # video_targetはVideoTargetクラスのインスタンス
        video_target = self.videos_target[video_index]
        video_data = self.videos_data[video_index]
        # 上からaction_index番目のイベントのフレーム番号1つを取得
        frame_index = video_target.get_frame_index_by_action_index(action_index)
        # frame_indexが最初か最後のフレームにならないようにクリップ
        frame_index = self.indexes_generator.clip_index(
            frame_index, video_data["frame_count"], 1
        )
        # frame_indexを基準に周囲15フレームを取得
        frame_indexes = self.indexes_generator.make_stack_indexes(frame_index)
        return video_index, frame_indexes


### Target LabelなしのChallenge用データセット
### このデータセットは、アノテーションがないため、アノテーションを取得するメソッドがない
### ActionDatasetと同じように、フレームの取得と処理を行う
### key_frame_indexesをvideo_dataから取得するの代わりに入力する
### ActionDatasetを継承しない
class ChallengeActionDataset:
    def __init__(
        self,
        key_frame_indexes: list,
        video_info: dict,
        frame_fetcher: AbstractFrameFetcher,
        indexes_generator: StackIndexesGenerator,
        frames_process_fn: Callable[[torch.Tensor], torch.Tensor],
    ):
        self.frame_fetcher = frame_fetcher
        self.indexes_generator = indexes_generator
        self.frames_process_fn = frames_process_fn

        self.key_frame_indexes = key_frame_indexes
        self.num_clips = len(self.key_frame_indexes)
        self.video_info = video_info

    def __len__(self) -> int:
        return self.num_clips

    def get_video_frame_indexes(self, index: int) -> tuple[int, list[int]]:
        frame_index = self.key_frame_indexes[index]
        frame_index = self.indexes_generator.clip_index(
            frame_index, self.video_info["num_frames"], 1
        )
        frame_indexes = self.indexes_generator.make_stack_indexes(frame_index)
        return frame_indexes

    def get_frames(self, frame_indexes: list[int]):
        frames = self.frame_fetcher.fetch_frames(frame_indexes)
        return frames

    def process_frames(self, frames: torch.Tensor):
        input_tensor = self.frames_process_fn(frames)
        return input_tensor

    def get(self, index: int) -> torch.Tensor:
        frame_indexes = self.get_video_frame_indexes(index)
        frames = self.get_frames(frame_indexes)
        return self.process_frames(frames)
