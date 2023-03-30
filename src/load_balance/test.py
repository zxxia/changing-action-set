import numpy as np
from load_balance.environment import Environment
from load_balance.job import JobGenerator


def test(agent, args, num_exp=100):
    # set up environment
    job_generator = JobGenerator(
        args.num_stream_jobs, args.job_distribution, args.job_size_min,
        args.job_size_max, args.job_size_pareto_shape,
        args.job_size_pareto_scale, args.job_interval)

    env = Environment(job_generator, args.num_workers, args.service_rates,
                      args.service_rate_min, args.service_rate_max,
                      args.queue_shuffle_prob, args.action_mask)

    all_total_reward = []
    all_avg_jct = []
    all_percent_finished_jobs = []

    # run experiment
    for i in range(num_exp):
        env.set_random_seed(100000000 + i)
        env.reset()

        total_reward = 0

        state = env.observe()
        workers, job, _, mask = state
        done = False

        while not done:
            act = agent.get_action(workers, job, mask)
            state, reward, done = env.step(act)
            workers, job, _, mask = state
            total_reward += reward
        all_total_reward.append(total_reward / args.reward_scale)
        all_avg_jct.append(np.mean(env.get_job_completion_time()))
        all_percent_finished_jobs.append(
            len(env.finished_jobs) / job_generator.num_stream_jobs)

    return all_total_reward, all_avg_jct, all_percent_finished_jobs
