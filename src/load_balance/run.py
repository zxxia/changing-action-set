import os
import numpy as np
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from load_balance.heuristic_agents import (
    LeastWorkAgent, RoundRobinAgent, ShortestProcessingTimeAgent,
    UniformRandomAgent)
from load_balance.parser import parse_args
from load_balance.input_driven_rl.train import train
from load_balance.test import test
from load_balance.input_driven_rl.actor_agent import ActorAgent
from common.utils import compute_std_of_mean, save_args


def main():
    args = parse_args()

    if args.command == 'train':
        # train an RL agent
        save_args(args, args.model_folder)
        train(args)
    elif args.command == 'test':
        # test
        for agent_name in args.agent:
            if agent_name == 'LeastWork':
                agent = LeastWorkAgent()
            elif agent_name == 'ShortestProcessingTime':
                agent = ShortestProcessingTimeAgent()
            elif agent_name == 'UniformRandom':
                agent = UniformRandomAgent()
            elif agent_name == 'RoundRobin':
                agent = RoundRobinAgent()
            elif agent_name == 'rl':
                sess = tf.compat.v1.Session()
                tf.compat.v1.set_random_seed(args.seed)
                agent = ActorAgent(sess, args.num_workers, args.job_size_norm_factor)

                # initialize parameters
                sess.run(tf.global_variables_initializer())
                saver = tf.compat.v1.train.Saver()

                # load trained model
                if args.pretrained_model is not None:
                    saver.restore(sess, args.pretrained_model)
            else:
                raise ValueError('Unsupported agent {}'.format(agent_name))
            total_reward, avg_jct, percent_finished_jobs = test(agent, args)
            print(agent_name)
            print('total_reward = ', np.mean(total_reward), compute_std_of_mean(total_reward))
            print('avg_jct = ', np.mean(avg_jct), compute_std_of_mean(avg_jct))
            print('percent of finished jobs:', np.mean(percent_finished_jobs),
                  compute_std_of_mean(percent_finished_jobs))
    else:
        raise ValueError('Unsupported command {}'.format(args.command))


if __name__ == '__main__':
    main()
