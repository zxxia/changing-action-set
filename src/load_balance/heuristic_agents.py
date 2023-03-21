from typing import List, Optional
import numpy as np
from load_balance.job import Job
from load_balance.worker import Worker

class LeastWorkAgent(object):
    def __init__(self):
        pass

    def get_action(self, workers: List[Worker], job: Job,
                   mask: Optional[np.ndarray] = None):
        if mask is None:
            mask = np.ones(len(workers))
        assert mask is not None and not np.all(mask == 0), "action mask must allow at least one action."
        min_work_idx = None
        min_work = np.inf

        for i in range(len(workers)):
            if mask[i] == 0:
                continue
            worker = workers[i]
            work = np.sum([j.size for j in worker.queue])
            if work < min_work:
                min_work_idx = i
                min_work = work
        assert min_work_idx is not None

        return min_work_idx

class ShortestProcessingTimeAgent(object):
    def __init__(self):
        pass

    def get_action(self, workers: List[Worker], job: Job,
                   mask: Optional[np.ndarray] = None):
        if mask is None:
            mask = np.ones(len(workers))
        assert mask is not None and not np.all(mask == 0), "action mask must allow at least one action."
        min_time_idx = None
        min_time = np.inf

        for i in range(len(workers)):
            if mask[i] == 0:
                continue
            worker = workers[i]
            work = np.sum([j.size for j in worker.queue])
            remain_time = work / worker.service_rate
            if remain_time < min_time:
                min_time_idx = i
                min_time = remain_time

        return min_time_idx

class UniformRandomAgent(object):
    def __init__(self) -> None:
        pass

    def get_action(self, workers: List[Worker], job: Job,
                   mask: Optional[np.ndarray] = None):
        if mask is None:
            mask = np.ones(len(workers))
        assert mask is not None and not np.all(mask == 0), "action mask must allow at least one action."
        prob = mask / np.sum(mask)
        idx = np.random.choice(len(workers), 1, p=prob)[0]
        return idx

class RoundRobinAgent(object):
    def __init__(self) -> None:
        self.idx = -1
        pass

    def get_action(self, workers: List[Worker], job: Job,
                   mask: Optional[np.ndarray] = None):
        if mask is None:
            mask = np.ones(len(workers))
        assert mask is not None and not np.all(mask == 0), "action mask must allow at least one action."
        n_workers = len(workers)
        self.idx = (self.idx + 1) % n_workers
        while mask[self.idx] == 0:
            self.idx = (self.idx + 1) % n_workers
        return self.idx
