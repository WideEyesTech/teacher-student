"""
This purpose of this file is to manage the teachers and
make inference with different teachers from a global file, without  
going deeper into repository files.

This file is only used as a shortcut file, configuration of each teacher 
should be done for each teacher.
"""
import argparse
from os import environ
from os import system
from subprocess import run


def handleinput(args):
    """Handle user input
    """
    if args.teacher == "centernet":
        system("CUDA_VISIBLE_DEVICES={} nice -n1 python3 teachers/gen_weak_labels.py".format(args.gpu_number))

    if args.teacher == "mmdetection":
        system("chmod 777 ./mmdetection.sh")
        
        # Set env variables in order to speed up process
        environ["GPU_NUMBER"] = "0"
        environ["DATA"] = "/home/toni/datasets/openimages"
        environ["FILENAMES"] = "/home/toni/datasets/sorted_4_oi_names.txt"
        environ["RESULTS"] = "/opt/results/GCNET"
        environ["CONFIG"] = "/home/toni/Desktop/teacher-student/teachers/mmdetection/configs/gcnet/mask_rcnn_r4_gcb_c3-c5_r50_fpn_1x.py"
        environ["CHECKPOINT"] = "/home/toni/Desktop/teacher-student/teachers/model_zoo/mask_rcnn_r4_gcb_c3-c5_r50_fpn_1x_20190602-18ae2dfd.pth"

        run("./mmdetection.sh", check=True)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(
        description='Get inferences from specific teacher')

    PARSER.add_argument('teacher')
    PARSER.add_argument('gpu_number', default=0)

    ARGS = PARSER.parse_args()
    handleinput(ARGS)
