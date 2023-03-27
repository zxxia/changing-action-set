import numpy as np
from typing import List

from load_balance.utils import generate_coin_flips
from load_balance.worker import Worker
from load_balance.job import Job, JobGenerator
from load_balance.timeline import Timeline
from load_balance.wall_time import WallTime


class Environment(object):
    def __init__(self, job_generator: JobGenerator, num_workers: int = 3,
                 service_rates: List[float] = [0.5, 1.0, 2.0],
                 service_rate_min: float = 1.0, service_rate_max: float = 10.0,
                 queue_shuffle_prob: float = 0.5):
        # global timer
        self.wall_time = WallTime()
        # uses priority queue
        self.timeline = Timeline()
        self.num_workers = num_workers
        # total number of streaming jobs (can be very large)
        self.service_rates = service_rates
        self.service_rate_min = service_rate_min
        self.service_rate_max = service_rate_max
        self.queue_shuffle_prob = queue_shuffle_prob

        self.job_generator = job_generator

        # workers
        self.workers = self.initialize_workers()
        # episode retry probability
        self.reset_prob = 0
        # current incoming job to schedule
        self.incoming_job = None
        # finished jobs
        self.finished_jobs = []

        self.worker_avail = np.ones(self.num_workers)

    def generate_jobs(self):
        all_t, all_size = self.job_generator.generate_jobs()
        for t, size in zip(all_t, all_size):
            self.timeline.push(t, size)

    def initialize(self):
        assert self.wall_time.curr_time == 0
        new_time, obj = self.timeline.pop()
        self.wall_time.update(new_time)
        assert isinstance(obj, int)  # a job arrival event
        size = obj
        self.incoming_job = Job(size, self.wall_time.curr_time)

    def initialize_workers(self):
        workers = []

        for worker_id in range(self.num_workers):
            if self.service_rates is None:
                service_rate = np.random.uniform(
                    self.service_rate_min, self.service_rate_max)
            else:
                service_rate = self.service_rates[worker_id]
            worker = Worker(worker_id, service_rate, self.wall_time,
                            self.queue_shuffle_prob)
            workers.append(worker)

        return workers

    def observe(self):
        return self.workers, self.incoming_job, self.wall_time.curr_time, self.worker_avail

    def reset(self):
        for worker in self.workers:
            worker.reset()
        self.wall_time.reset()
        self.timeline.reset()
        self.generate_jobs()
        self.max_time = generate_coin_flips(self.reset_prob)
        self.incoming_job = None
        self.finished_jobs = []
        # initialize environment (jump to first job arrival event)
        self.initialize()

    def set_random_seed(self, seed):
        np.random.seed(seed)

    def step(self, action):

        # schedule job to worker
        self.workers[action].schedule(self.incoming_job)
        running_job = self.workers[action].process()
        if running_job is not None:
            self.timeline.push(running_job.finish_time, running_job)

        # erase incoming job
        self.incoming_job = None

        # set to compute reward from this time point
        reward = 0

        while len(self.timeline) > 0:

            new_time, obj = self.timeline.pop()

            # update reward
            num_active_jobs = sum(len(w.queue) for w in self.workers)
            for worker in self.workers:
                if worker.curr_job is not None:
                    assert worker.curr_job.finish_time >= \
                           self.wall_time.curr_time  # curr job should be valid
                    num_active_jobs += 1
            if new_time is not None:
                reward -= (new_time - self.wall_time.curr_time) * num_active_jobs

            # tick time
            self.wall_time.update(new_time)

            if new_time is not None and new_time >= self.max_time:
                break

            if isinstance(obj, int):  # new job arrives
                size = obj
                self.incoming_job = Job(size, self.wall_time.curr_time)
                # break to consult agent
                break

            elif isinstance(obj, Job):  # job completion on worker
                job = obj
                self.finished_jobs.append(job)
                if job.worker and job.worker.curr_job == job:
                    # worker's current job is done
                    job.worker.curr_job = None
                running_job = job.worker.process() if job.worker else None
                if running_job is not None:
                    self.timeline.push(running_job.finish_time, running_job)

            else:
                print("illegal event type")
                exit(1)

        done = ((len(self.timeline) == 0) and \
               self.incoming_job is None) or \
               (self.wall_time.curr_time >= self.max_time)

        return self.observe(), reward, done

    def get_job_completion_time(self):
        return [j.finish_time - j.start_time for j in self.finished_jobs]

    def change_action_availability(self):
        # TODO: hard coded action change prob.
        prob = 0.9
        self.worker_avail = np.array(np.random.rand(self.num_workers) <= prob, dtype=int)
        # Make sure that there is at least one available action always.
        while not self.worker_avail.any():
            self.worker_avail = np.array(np.random.rand(self.num_workers) <= prob, dtype=int)
