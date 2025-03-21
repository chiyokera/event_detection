import abc
from collections import defaultdict

import numpy as np
import torch


class VideoTarget:
    def __init__(self, video_data: dict, classes: list[str]):
        self.classes = classes
        self.num_classes = len(classes)
        self.class2target = {cls: trg for trg, cls in enumerate(classes)}

        # defaultdictは、キーが存在しない場合に自動的にデフォルト値(0.0)を返す辞書
        self.frame_index2class_target: dict[str, defaultdict] = {
            cls: defaultdict(float) for cls in classes
        }
        # action_indexはイベントが上から何個目かを数えたただの番号、上から〇番目のイベントのフレーム番号を返す
        self.action_index2frame_index: dict[int, int] = dict()

        # フレームインデックスでソートされた（フレームインデックス, アクション）のペアのリストが格納
        actions_sorted_by_frame_index = sorted(
            video_data["frame_index2action"].items(), key=lambda x: x[0]
        )
        for action_index, (frame_index, action) in enumerate(
            actions_sorted_by_frame_index
        ):
            self.action_index2frame_index[action_index] = frame_index
            if action in classes:
                self.frame_index2class_target[action][frame_index] = 1.0

    def target(self, frame_index: int) -> np.ndarray:
        target = np.zeros(self.num_classes, dtype=np.float32)
        # フレーム番号を入力すると、そのフレームのアクションに対応するインデックスのみ1.0のベクトルを返す
        for cls in self.classes:
            # もしframe_indexがアクションのフレームに含まれていない場合、0.0を出力
            target[self.class2target[cls]] = self.frame_index2class_target[cls][
                frame_index
            ]
        return target

    # frame_indexesはアクションがある可能性が高いフレームの周囲15フレームのリスト
    # 実際のラベルと対応されたフレームと照らし合わせながら、そのフレームのアノテーションを取得
    # 例えば、frame_indexes=[2, 4, 6, 8, 10]のとき、
    # 最初のフレーム2がなにもアクションラベルがないならtargets[0]=[0,..,0] (長さはクラスの数分)
    def targets(self, frame_indexes: list[int]) -> np.ndarray:
        targets = [self.target(idx) for idx in frame_indexes]
        return np.stack(targets, axis=0)

    def get_frame_index_by_action_index(self, action_index: int) -> int:
        return self.action_index2frame_index[action_index]

    def num_actions(self) -> int:
        return len(self.action_index2frame_index)


# targetsは16フレームでぎり(たぶんそうなってる)、その中央の15フレーム分だけを取り出す
def center_crop_targets(targets: np.ndarray, crop_size: int) -> np.ndarray:
    num_crop_targets = targets.shape[0] - crop_size
    left = num_crop_targets // 2
    right = num_crop_targets - left
    return targets[left:-right]


# 抽象基底クラス＝このクラスを継承したクラスは、__call__メソッドを実装しなければならない
class TargetsToTensorProcessor(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __call__(self, targets: np.ndarray) -> torch.Tensor:
        pass


class MaxWindowTargetsProcessor(TargetsToTensorProcessor):
    def __init__(self, window_size):
        self.window_size = window_size  # 15

    def __call__(self, targets: np.ndarray) -> torch.Tensor:
        targets = targets.astype(np.float32, copy=False)
        targets = center_crop_targets(targets, self.window_size)
        # 1.0がある行を取得
        target = np.amax(targets, axis=0)
        target_tensor = torch.from_numpy(target)
        return target_tensor
