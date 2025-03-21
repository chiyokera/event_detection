from typing import Optional

import numpy as np


class StackIndexesGenerator:
    def __init__(self, size: int, step: int):
        # size: 15, step: 2
        self.size = size
        self.step = step
        # self.behind: 7, self.ahead: 7
        self.behind = self.size // 2
        self.ahead = self.size - self.behind - 1
        # sellf.behind: 14, self.ahead: 14
        self.behind *= self.step
        self.ahead *= self.step

    # 周囲30フレームから1つ飛ばしで15フレーム番号を取得
    def make_stack_indexes(self, frame_index: int) -> list[int]:
        return list(
            range(
                frame_index - self.behind,
                frame_index + self.ahead + 1,
                self.step,
            )
        )

    def clip_index(self, index: int, frame_count: int, save_zone: int = 0) -> int:
        behind_frames = self.behind + save_zone
        ahead_frames = self.ahead + save_zone
        if index < behind_frames:
            index = behind_frames
        elif index >= frame_count - ahead_frames:
            index = frame_count - ahead_frames - 1
        return index


class FrameIndexShaker:
    def __init__(
        self,
        shifts: list[int],
        weights: Optional[list[float]] = None,
        prob: float = 1.0,
    ):
        self.shifts = shifts
        self.weights = weights
        self.prob = prob

    # 25%の確率でフレームをシャッフルする
    # 例えば、frame_indexes=[2, 4, 6, 8, 10]のとき、25％
    def __call__(self, frame_indexes: list[int]) -> list[int]:
        if np.random.random() < self.prob:
            # shifts: [-1, 0, 1], weights: [0.2, 0.6, 0.2]より、
            # 0.2の確率で-1, 0.6の確率で0, 0.2の確率で1を要素として持つリストをframe_indexesの長さで作成
            # ランダムで5回、-1,0,1のいずれかをそれぞれ0.2, 0.6, 0.2の確率で選択
            shifts = np.random.choice(self.shifts, len(frame_indexes), p=self.weights)
            shaken_indexes = list()
            for index, shift in zip(frame_indexes, shifts):
                shaken_indexes.append(int(index + shift))
            return shaken_indexes
        else:
            return frame_indexes
