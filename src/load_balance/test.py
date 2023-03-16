from load_balance.environment import Environment
from load_balance.job import JobGenerator


def test(agent, args):
    # set up environment
    job_generator = JobGenerator(
        args.num_stream_jobs, args.job_distribution, args.job_size_min,
        args.job_size_max, args.job_size_pareto_shape,
        args.job_size_pareto_scale, args.job_interval)

    env = Environment(job_generator, args.num_workers, args.service_rates,
                      args.service_rate_min, args.service_rate_max,
                      args.queue_shuffle_prob)

    # run experiment
    env.set_random_seed(args.seed)
    env.reset()

    total_reward = 0

    state = env.observe()
    workers, job, _ = state
    done = False

    while not done:
        act = agent.get_action(workers, job)
        state, reward, done = env.step(act)
        workers, job, _ = state
        total_reward += reward

    return total_reward