import numpy as np
from typing import List, Tuple

class Job(object):
    def __init__(self, size, arrival_time):
        self.size = size
        self.arrival_time = arrival_time
        self.worker = None
        self.start_time = None
        self.finish_time = None


class JobGenerator:
    def __init__(self, num_stream_jobs: int = 2000,
                 job_distribution: str = "uniform",
                 job_size_min: float = 10.0, job_size_max: float = 10000.0,
                 job_size_pareto_shape: float = 2.0,
                 job_size_pareto_scale: float = 100.0,
                 job_interval: int = 100):
        self.num_stream_jobs = num_stream_jobs
        self.job_distribution = job_distribution
        self.job_size_min = job_size_min
        self.job_size_max = job_size_max
        self.job_size_pareto_shape = job_size_pareto_shape
        self.job_size_pareto_scale = job_size_pareto_scale
        self.job_interval = job_interval

    def generate_jobs(self) -> Tuple[List[float], List[float]]:

        # time and job size
        all_t = []
        all_size = []

        # generate streaming sequence
        t = 0
        for _ in range(self.num_stream_jobs):
            if self.job_distribution == 'uniform':
                size = int(np.random.uniform(self.job_size_min, self.job_size_max))
            elif self.job_distribution == 'pareto':
                size = int((np.random.pareto(
                    self.job_size_pareto_shape) + 1) * self.job_size_pareto_scale)
            else:
                raise ValueError('Job distribution ' + self.job_distribution + ' does not exist')

            # if args.cap_job_size:
            #     size = min(int(args.job_size_max), size)

            t += int(np.random.exponential(self.job_interval))

            all_t.append(t)
            all_size.append(size)

        return all_t, all_size
