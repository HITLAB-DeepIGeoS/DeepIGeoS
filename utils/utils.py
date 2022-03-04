import argparse


def get_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        '-c', '--config',
        default='None',
        help='The Configuration file')
    args = argparser.parse_args()
    return args
