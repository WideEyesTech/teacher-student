"""
This purpose of this file is to manage the teachers and
make inference with different teachers from a global file, without  
going deeper into repository files.

This file is only used as a shortcut file, configuration of each teacher 
should be done for each teacher.
"""
import argparse
from os import system
from subprocess import run


def handleinput(args):
    """Handle user input
    """
    if args.teacher == "centernet":
        system("CUDA_VISIBLE_DEVICES={} nice -n1 python3 teachers/gen_weak_labels.py".format(args.gpu_number))

    if args.teacher == "mmdetection":
        system("chmod 777 ./mmdetection.sh")
        run("./mmdetection.sh", check=True)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(
        description='Get inferences from specific teacher')

    PARSER.add_argument('teacher')
    PARSER.add_argument('gpu_number', default=0)

    ARGS = PARSER.parse_args()
    handleinput(ARGS)
