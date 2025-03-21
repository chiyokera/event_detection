from queue import Empty, Queue
from typing import Optional

import torch

# from rosny import ProcessStream
from rosny import ProcessNode
from team_location_detection.data_loaders.base_data_loader import BaseDataLoader
from team_location_detection.datasets import ActionDataset
from team_location_detection.frame_fetchers import NvDecFrameFetcher


class SequentialWorkerStream(ProcessNode):  # ProcessStream
    def __init__(
        self,
        dataset: ActionDataset,
        index_queue: Queue,
        result_queue: Queue,
        frame_buffer_size: int,
        gpu_id: int = 0,
        timeout: float = 1.0,
    ):
        super().__init__()
        self._dataset = dataset
        self._index_queue = index_queue
        self._result_queue = result_queue
        self._frame_buffer_size = frame_buffer_size
        self._gpu_id = gpu_id
        self._timeout = timeout

        self._video_index = -1
        self._frame_fetcher: Optional[NvDecFrameFetcher] = None
        self._frame_index2frame: dict[int, torch.Tensor] = dict()
        self._last_frame_index = 0

    def reset(self, video_index: int = -1):
        if video_index == -1:
            self._video_index = -1
            self._frame_fetcher = None
            self._last_frame_index = 0
        # nvdecFrameFetcherはGPUにフレームとってきますよってやつ
        # 最初のイベントのみここを通る。
        else:
            self._video_index = video_index
            self._frame_fetcher = self._dataset.get_frame_fetcher(
                video_index, NvDecFrameFetcher, self._gpu_id
            )
            self._last_frame_index = 0
        self._frame_index2frame = dict()

    def read_until_last(self, last_frame_index: int):
        # 基本これはないはず
        if self._last_frame_index >= last_frame_index:
            return

        while True:
            # 実際にとってきたフレーム
            frame = self._frame_fetcher.fetch_frame()
            # 実際にとってきたフレームのフレーム番号
            frame_index = self._frame_fetcher.current_index
            self._frame_index2frame[frame_index] = frame
            self._last_frame_index = frame_index
            # self._buffer_sizeは30, つまり30こ前のフレームは削除
            del_frame_index = frame_index - self._frame_buffer_size
            if del_frame_index in self._frame_index2frame:
                del self._frame_index2frame[del_frame_index]
            if frame_index == last_frame_index:
                break

    def get_sample(self, index):
        # イベント番号indexに対応するフレーム番号の周囲15フレームのフレーム番号と、
        # ビデオ番号を取得(といっても１つだけだが)
        video_index, frame_indexes = self._dataset.get_video_frame_indexes(index)

        last_frame_index = max(frame_indexes)
        if video_index != self._video_index:
            # self._video_indexが-1から0になる
            self.reset(video_index)

        # 基本、ここは通らない(前回の最終フレーム番号が今回の最終フレーム番号より小さい場合はないだろ？)
        elif last_frame_index < self._last_frame_index:
            self.reset(video_index)

        # 最終フレーム番号までフレームを取得
        self.read_until_last(last_frame_index)

        frames = torch.stack([self._frame_index2frame[i] for i in frame_indexes], dim=0)
        targets = self._dataset.get_targets(video_index, frame_indexes)
        return self._dataset.process_frames_targets(frames, targets)

    def work(self):
        try:
            index = self._index_queue.get(timeout=self._timeout)
        except Empty:
            return
        sample = self.get_sample(index)
        self._result_queue.put(sample)


class SequentialDataLoader(BaseDataLoader):
    def __init__(
        self,
        dataset: ActionDataset,
        batch_size: int,
        frame_buffer_size: int,
        gpu_id: int = 0,
    ):
        self.frame_buffer_size = frame_buffer_size
        super().__init__(dataset=dataset, batch_size=batch_size, gpu_id=gpu_id)

    def init_workers_stream(self) -> SequentialWorkerStream:
        return SequentialWorkerStream(
            self.dataset,
            self._index_queue,
            self._result_queue,
            self.frame_buffer_size,
            gpu_id=self.gpu_id,
        )

    def clear_queues(self):
        super().clear_queues()
        self._workers_stream.reset()
