import abc

# from rosny.loop import LoopStream
from rosny.loop import LoopNode
from torch.multiprocessing import Queue
from torch.utils.data._utils.collate import default_collate


class BaseDataLoader(metaclass=abc.ABCMeta):
    def __init__(self, dataset, batch_size: int, gpu_id: int = 0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.gpu_id = gpu_id

        # __len__メソッドをDatasetクラスに実装することで、len()関数でインスタンスの長さを取得できる
        # ここでは、データセットの全試合のアクションの総数を取得
        self._index_queue = Queue(maxsize=len(self.dataset))
        self._result_queue = Queue(maxsize=self.batch_size)

        self._num_samples_left = 0

        # self._workers_streamは、データセットのインデックスを取得し、そのインデックスに対応するデータを取得する
        # ここでセッティング、まだwork関数は実行されていない
        self._workers_stream = self.init_workers_stream()
        # ここで__iter__を先に実行し、_index.queueにindexをputする。そのあと、workを実行しているらしい
        self.start_workers()

    @abc.abstractmethod
    def init_workers_stream(self) -> LoopNode:  # LoopStream
        pass

    def start_workers(self):
        # ここでwork関数が実行される
        self._workers_stream.start()

    def stop_workers(self):
        if not self._workers_stream.stopped():
            self._workers_stream.stop()
        if not self._workers_stream.joined():
            self._workers_stream.join()

    def clear_queues(self):
        while not self._index_queue.empty():
            self._index_queue.get()
        while not self._result_queue.empty():
            self._result_queue.get()

    # このクラスを反復可能にする
    # そのためには、__iter__メソッドと__next__メソッドを実装する必要がある
    # __iter__メソッドが初めに一度だけ呼び出され、__next__メソッドは反復ごとに呼び出される
    # return selfを返すことで、__next__メソッドが呼び出される
    def __iter__(self):
        self._num_samples_left = len(self.dataset)
        self.clear_queues()
        for index in range(len(self.dataset)):
            # 全試合分のイベントインデックスをキューに入れる
            self._index_queue.put(index)
        return self

    def __next__(self):
        batch_list = []
        while self._num_samples_left:
            sample = self._result_queue.get()
            batch_list.append(sample)
            self._num_samples_left -= 1
            if len(batch_list) == self.batch_size:
                # これがデータローダーの出力
                return default_collate(batch_list)
        if batch_list:
            return default_collate(batch_list)
        self.clear_queues()
        raise StopIteration

    # もしこれがないと、データローダーを反復可能にするときにエラーが発生する
    # リソース解放の役割
    def __del__(self):
        self.stop_workers()
