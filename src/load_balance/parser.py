import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Parameters')

    parser.add_argument('--seed', type=int, default=42,
                        help='random seed (default: 42)')
    parser.add_argument('--result-folder', type=str, default='./results/',
                        help='Result folder path (default: ./results)')
    parser.add_argument('--agent', type=str, default='rl',
                        choices=('rl', 'LeastWork', 'ShortestProcessTime'),
                        help='agent type (rl, LeastWork, ShortestProcessTime)')

    # training
    subparsers = parser.add_subparsers(dest='command')
    train = subparsers.add_parser('train', help='train RL agent.')

    parser.add_argument('--model-folder', type=str, default='./models/',
                        help='Model folder path (default: ./models)')
    train.add_argument('--eps', type=float, default=1e-6,
                       help='epsilon (default: 1e-6)')
    train.add_argument('--hid-dims', type=int, default=[200, 128], nargs='+',
                       help='hidden dimensions (default: [200, 128])')
    train.add_argument('--num-agents', type=int, default=10,
                       help='number of training agents (default: 10)')
    train.add_argument('--pretrained-model', type=str, default=None,
                       help='path to the saved model (default: None)')
    train.add_argument('--num-saved-models', type=int, default=1000,
                       help='Number of models to keep (default: 1000)')
    train.add_argument('--model-save-interval', type=int, default=100,
                       help='Interval for saving Tensorflow model (default: 100)')
    train.add_argument('--num-models', type=int, default=10,
                       help='Number of models for value network (default: 10)')
    train.add_argument('--num-ep', type=int, default=10000,
                       help='Number of training epochs (default: 10000)')
    train.add_argument('--lr', type=float, default=1e-3,
                       help='learning rate (default: 1e-3)')
    train.add_argument('--gamma', type=float, default=1,
                       help='discount factor (default: 1)')
    train.add_argument('--reward-scale', type=float, default=1e4,
                       help='reward scale in training (default: 1e4)')
    train.add_argument('--diff-reward', type=int, default=0,
                       help='Differential reward mode (default: 0)')
    train.add_argument('--average-reward-storage', type=int, default=100000,
                       help='Storage size for average reward (default: 100000)')
    train.add_argument('--entropy-weight-init', type=float, default=1,
                       help='Initial exploration entropy weight (default: 1)')
    train.add_argument('--entropy-weight-min', type=float, default=0.0001,
                       help='Final minimum entropy weight (default: 0.0001)')
    train.add_argument('--entropy-weight-decay', type=float, default=1e-3,
                       help='Entropy weight decay rate (default: 1e-3)')
    train.add_argument('--reset-prob', type=float, default=1e-5,
                       help='Probability for episode to reset (default: 1e-5)')

    # load balance environment configurations
    parser.add_argument('--num-workers', type=int, default=3,
                        help='number of workers (default: 3)')
    parser.add_argument('--num-stream-jobs', type=int, default=2000,
                        help='number of streaming jobs (default: 2000)')
    parser.add_argument('--service-rates', type=float, default=[0.5, 1.0, 2.0],
                        nargs='+',
                        help='workers service rates (default: [0.5, 1.0, 2.0])')
    parser.add_argument('--service-rate-min', type=float, default=1.0,
                        help='minimum service rate (default: 1.0)')
    parser.add_argument('--service-rate-max', type=float, default=10.0,
                        help='maximum service rate (default: 4.0)')
    parser.add_argument('--job-distribution', type=str, default='uniform',
                        help='Job size distribution (default: uniform)')
    parser.add_argument('--job-size-min', type=float, default=10.0,
                        help='minimum job size (default: 100.0)')
    parser.add_argument('--job-size-max', type=float, default=10000.0,
                        help='maximum job size (default: 10000.0)')
    parser.add_argument('--job-size-norm-factor', type=float, default=1000.0,
                        help='normalize job size in the feature (default: 1000.0)')
    parser.add_argument('--job-size-pareto-shape', type=float, default=2.0,
                        help='pareto job size distribution shape (default: 2.0)')
    parser.add_argument('--job-size-pareto-scale', type=float, default=100.0,
                        help='pareto job size distribution scale (default: 100.0)')
    parser.add_argument('--job-interval', type=int, default=100,
                        help='job arrival interval (default: 100)')
    # parser.add_argument('--cap_job_size', type=int, default=0,
    #                     help='cap job size below max (default: 0)')
    parser.add_argument('--queue-shuffle-prob', type=float, default=0.5,
                        help='queue shuffle prob (default: 0.5)')
    return parser.parse_args()
