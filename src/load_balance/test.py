from load_balance.environment import Environment
from load_balance.job import JobGenerator


def test(agent, seed):
    # set up environment
    job_generator = JobGenerator()

    env = Environment(job_generator)

    # run experiment
    env.set_random_seed(seed)
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
