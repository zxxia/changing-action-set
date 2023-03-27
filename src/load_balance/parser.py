import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description='Parameters',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--result-folder', type=str, default='./results/',
                        help='Result folder path')
    parser.add_argument('--pretrained-model', type=str, default=None,
                       help='path to the saved model')

    # training
    subparsers = parser.add_subparsers(dest='command')
    train = subparsers.add_parser('train', help='train RL agent.')

    train.add_argument('--model-folder', type=str, default='./models/',
                        help='Model folder path')
    train.add_argument('--eps', type=float, default=1e-6,
                       help='epsilon')
    train.add_argument('--hid-dims', type=int, default=[200, 128], nargs='+',
                       help='hidden dimensions')
    train.add_argument('--num-agents', type=int, default=16,
                       help='number of training agents')
    train.add_argument('--num-saved-models', type=int, default=1000,
                       help='Number of models to keep')
    train.add_argument('--model-save-interval', type=int, default=100,
                       help='Interval for saving Tensorflow model')
    train.add_argument('--num-models', type=int, default=10,
                       help='Number of models for value network')
    train.add_argument('--num-ep', type=int, default=10000,
                       help='Number of training epochs')
    train.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    train.add_argument('--gamma', type=float, default=1,
                       help='discount factor')
    train.add_argument('--diff-reward', type=int, default=0,
                       help='Differential reward mode')
    train.add_argument('--average-reward-storage', type=int, default=100000,
                       help='Storage size for average reward')
    train.add_argument('--entropy-weight-init', type=float, default=1,
                       help='Initial exploration entropy weight')
    train.add_argument('--entropy-weight-min', type=float, default=0.0001,
                       help='Final minimum entropy weight')
    train.add_argument('--entropy-weight-decay', type=float, default=1e-3,
                       help='Entropy weight decay rate')
    train.add_argument('--reset-prob', type=float, default=1e-5,
                       help='Probability for episode to reset')

    # testing
    test = subparsers.add_parser('test', help='test a load balance agent.')
    test.add_argument('--agent', type=str, default=['rl'], nargs='+',
                        choices=('rl', 'LeastWork', 'ShortestProcessingTime', 'UniformRandom', 'RoundRobin'),
                        help='agent type')

    # load balance environment configurations
    parser.add_argument('--num-workers', type=int, default=3,
                        help='number of workers')
    parser.add_argument('--num-stream-jobs', type=int, default=1000,
                        help='number of streaming jobs')
    parser.add_argument('--service-rates', type=float, default=[0.5, 1.0, 2.0],
                        nargs='+', help='workers service rates')
    parser.add_argument('--service-rate-min', type=float, default=1.0,
                        help='minimum service rate')
    parser.add_argument('--service-rate-max', type=float, default=4.0,
                        help='maximum service rate')
    parser.add_argument('--job-distribution', type=str, default='uniform',
                        help='Job size distribution')
    parser.add_argument('--job-size-min', type=float, default=100.0,
                        help='minimum job size')
    parser.add_argument('--job-size-max', type=float, default=1000.0,
                        help='maximum job size')
    parser.add_argument('--job-size-norm-factor', type=float, default=1000.0,
                        help='normalize job size in the feature')
    parser.add_argument('--job-size-pareto-shape', type=float, default=2.0,
                        help='pareto job size distribution shape')
    parser.add_argument('--job-size-pareto-scale', type=float, default=100.0,
                        help='pareto job size distribution scale')
    parser.add_argument('--job-interval', type=int, default=100,
                        help='job arrival interval')
    # parser.add_argument('--cap_job_size', type=int, default=0,
    #                     help='cap job size below max')
    parser.add_argument('--queue-shuffle-prob', type=float, default=0.5,
                        help='queue shuffle prob')
    parser.add_argument('--reward-scale', type=float, default=1e4,
                       help='reward scale in training')
    return parser.parse_args()
