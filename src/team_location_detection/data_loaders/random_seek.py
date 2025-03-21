import queue
from multiprocessing import Queue
from typing import Type

# from rosny import ProcessStream, ComposeStream
from rosny import ComposeNode, ProcessNode
from team_location_detection.data_loaders.base_data_loader import BaseDataLoader
from team_location_detection.datasets import ActionDataset
from team_location_detection.frame_fetchers import (
    AbstractFrameFetcher,
    NvDecFrameFetcher,
    OpencvFrameFetcher,
)


class RandomSeekWorkerStream(ProcessNode):
    def __init__(
        self,
        dataset: ActionDataset,
        index_queue: Queue,
        result_queue: Queue,
        frame_fetcher_class: Type[AbstractFrameFetcher],
        gpu_id: int = 0,
        timeout: float = 1.0,
    ):
        super().__init__()
        self._dataset = dataset
        self._index_queue = index_queue
        self._result_queue = result_queue
        self._frame_fetcher_class = frame_fetcher_class
        self._gpu_id = gpu_id
        self._timeout = timeout

    # インデックスキューからインデックスを取得し、そのインデックスに対応するデータを取得
    def work(self):
        try:
            # Queue.get()メソッドは、キューの先頭からデータを取り出す
            # timeoutを指定することで、キューが空の場合、指定した秒数待機する
            # indexはイベントのインデックス
            index = self._index_queue.get(timeout=self._timeout)
        except queue.Empty:
            return
        # dataset.pyのActionDatasetクラスのgetメソッドを実行
        # sampleは、input_tensor: shape = (15frames, 3channel, 736, 1280), output_tensor: shape = (num_classes)のタプル
        sample = self._dataset.get(index, self._frame_fetcher_class, self._gpu_id)

        # あるイベントのインデックスに対応するデータを取得し、そのデータをresult_queueに入れる
        self._result_queue.put(sample)


# rosnyのComposeNodeとは、複数のProcessNodeをまとめて、一つのProcessNodeとして扱うためのクラス
# ProcessNodeとは、個別のワーカーを表すクラス
class RandomSeekWorkersStream(ComposeNode):
    def __init__(self, streams: list[ProcessNode]):
        super().__init__()
        for index, stream in enumerate(streams):
            # __setattr__メソッドは、インスタンス変数を設定するメソッド
            # 第一引数に変数名、第二引数にstreamを指定する
            # 第二引数のstreamが実行される
            self.__setattr__(f"random_seek_{index}", stream)


class RandomSeekDataLoader(BaseDataLoader):
    def __init__(
        self,
        dataset: ActionDataset,
        batch_size: int,
        num_nvdec_workers: int = 1,
        num_opencv_workers: int = 0,
        gpu_id: int = 0,
    ):
        self.num_nvdec_workers = num_nvdec_workers
        self.num_opencv_workers = num_opencv_workers

        # ここでBaseDataLoaderの__init__メソッドを実行、__iter__メソッドとかが実行
        super().__init__(dataset=dataset, batch_size=batch_size, gpu_id=gpu_id)

    def init_workers_stream(self) -> RandomSeekWorkersStream:
        # 実際はnum_nvdecworkers=3, num_opencv_workers=1
        nvdec_streams = [
            RandomSeekWorkerStream(
                self.dataset,
                self._index_queue,
                self._result_queue,
                NvDecFrameFetcher,
                self.gpu_id,
            )
            for _ in range(self.num_nvdec_workers)
        ]
        opencv_streams = [
            RandomSeekWorkerStream(
                self.dataset,
                self._index_queue,
                self._result_queue,
                OpencvFrameFetcher,
                self.gpu_id,
            )
            for _ in range(self.num_opencv_workers)
        ]
        return RandomSeekWorkersStream(nvdec_streams + opencv_streams)
