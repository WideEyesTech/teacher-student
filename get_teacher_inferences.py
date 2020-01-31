"""
This purpose of this file is to manage the teachers and
make inference with different teachers from a global file, without  
going deeper into repository files.

This file is only used as a shortcut file, configuration of each teacher 
should be done for each teacher.
"""
import argparse

from teachers.CenterNet.gen_weak_labels import GenerateWeakLabels as with_centernet

def handleinput(args):
    """Handle user input
    """
    if args.teacher == "centernet":
        with_centernet.generate()


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(
        description='Get inferences from specific teacher')
    PARSER.add_argument('teacher')

    ARGS = PARSER.parse_args()
    handleinput(ARGS)