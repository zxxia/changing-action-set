import os
import numpy as np
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from load_balance.heuristic_agents import LeastWorkAgent, ShortestProcessingTimeAgent, UniformRandomAgent
from load_balance.parser import parse_args
from load_balance.input_driven_rl.train import train
from load_balance.test import test, test_unseen
from load_balance.input_driven_rl.actor_agent import ActorAgent
from common.utils import compute_std_of_mean


def main():
    args = parse_args()

    if args.command == 'train':
        if args.agent == 'rl':
            # train
            train(args)
        else:
            raise ValueError('Cannot train on {} agent'.format(args.agent))
    elif args.command == 'test':
        # test
        if args.agent == 'LeastWork':
            agent = LeastWorkAgent()
        elif args.agent == 'ShortestProcessingTime':
            agent = ShortestProcessingTimeAgent()
        elif args.agent == 'UniformRandom':
            agent = UniformRandomAgent()
        elif args.agent == 'rl':
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
            raise ValueError('Unsupported agent {}'.format(args.agent))
        total_reward = test_unseen(agent, args)
        print('total_reward = ', np.mean(total_reward), compute_std_of_mean(total_reward))
    else:
        raise ValueError('Unsupported command {}'.format(args.command))


if __name__ == '__main__':
    main()
