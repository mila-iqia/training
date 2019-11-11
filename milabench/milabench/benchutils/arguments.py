import argparse
import os


def add_bench_args(parser):
    """ add the arguments directly to a given parser """
    parser.add_argument('--repeat', type=int, default=100, help='number of observation timed')
    parser.add_argument('--number', type=int, default=10, help='number of time a task is done in between timer')
    parser.add_argument('--report', type=str, default=None, help='file to store the benchmark result in')
    # parser.add_argument('--sync', action='store_true', default=True, help='sync cuda streams for correct timings')
    return parser


def make_bench_args_parser(parser=None, subparser=False):
    """ create a an argument parser for the bench arguments"""
    if parser is None:
        parser = argparse.ArgumentParser('Bench util argument parser')

    if parser is not None and subparser:
        add_bench_subparser(parser)
    else:
        add_bench_args(parser)

    return parser


def add_bench_subparser(parser):
    """ add the bench argument as a subparser """

    p = parser.add_subparsers(dest='bench')
    p = p.add_parser('bench')
    add_bench_args(p)
    return parser


def get_arguments(parser=None, subparser=False, show=False):
    parser = make_bench_args_parser(parser, subparser)
    args = parser.parse_args()

    try:
        import torch
        if not torch.cuda.is_available():
            args.cuda = False
    except:
        pass

    if args.report is None:
        args.report = os.environ.get('REPORT_PATH')

    if show:
        print('-' * 80)
        for key, val in args.__dict__.items():
            print('{:>30}: {}'.format(key, val))
        print('-' * 80)
    return args


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='DeepSpeech training')
    parser.add_argument('--checkpoint', dest='checkpoint', action='store_true', help='Enables checkpoints')
    parser.add_argument('--save_folder', default=None, help='Location to save epoch models')
    parser.add_argument('--model_path', default=None, help='Location to save best validation model')
    parser.add_argument('--continue_from', default='', help='Continue from checkpoint model')
    parser.add_argument('--labels_path', default='', help='path to labels.json')
    parser.add_argument('--seed', default=0xdeadbeef, type=int, help='Random Seed')
    parser.add_argument('--acc', default=23.0, type=float, help='Target WER')
    parser.add_argument('--cuda', default=False, action='store_true', help='use cuda?')
    parser.add_argument('--start_epoch', default=-1, type=int, help='Number of epochs at which to start from')

    parser = add_bench_subparser(parser)

    args = parser.parse_args('--checkpoint --seed 0 --cuda bench --repeat 9 --number 6 --report test.csv'.split(' '))

    print(args)
