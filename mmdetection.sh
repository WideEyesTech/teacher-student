#!/bin/bash
echo "Tested config: NVCC 9.0, NVIDIA DRIVERS 440.33.01, GCC 5.5, CUDA 9.0"

read -p "Are you using conda? y/n " is_using_conda
if ( [[ "$is_using_conda" != "y" ]] && [[ "$is_using_conda" != "n" ]] ); then
    echo  "Incorrect answer"
    exit
fi

if [[ "$is_using_conda" == "y" ]]; then
    conda create -n open-mmlab python=3.7 -y
    conda activate open-mmlab
    
    conda install -c pytorch pytorch torchvision -y
    conda install cython -y
    git clone https://github.com/open-mmlab/mmdetection.git ./teachers/mmdetection
    cd ./teachers/mmdetection
    python3 -m pip install -r requirements/build.txt --user
    python3 -m pip install -v -e . --user
    cd ../../
else
    git clone https://github.com/open-mmlab/mmdetection.git ./teachers/mmdetection
    cd ./teachers/mmdetection
    python3 -m pip install -r requirements/build.txt --user
    python3 -m pip install -v -e . --user
    cd ../../
fi

cp ./teachers/teacher.py ./teachers/mmdetection/teacher.py

echo "Choose a teacher: ATSS, GCNET"
read teacher
if [[ "$teacher" != "ATSS" ]] && [[ "$teacher" != "GCNET" ]]; then
    echo -n "Incorrect answer"
    exit
fi

mkdir ./teachers/model_zoo
if [[ "$teacher" == "ATSS" ]]; then
    export CONFIG_FILE = "atss/atss_r50_fpn_1x.py"
    export CHECKPOINT_FILE = "atss_r50_fpn_1x_20200113-a7aa251e.pth"
    wget "https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/models/atss/atss_r50_fpn_1x_20200113-a7aa251e.pth" -P ./teachers/model_zoo
    elif [[ "$teacher" == "GCNET" ]]; then
    export CONFIG_FILE = "gcnet/mask_rcnn_r16_gcb_c3-c5_r50_fpn_1x.py"
    export CHECKPOINT_FILE = "mask_rcnn_r4_gcb_c3-c5_r50_fpn_1x_20190602-18ae2dfd.pth"
    wget "https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/models/gcnet/mask_rcnn_r4_gcb_c3-c5_r50_fpn_1x_20190602-18ae2dfd.pth" -P ./teachers/model_zoo
fi

echo "Choose a GPU number, OPTIONS:"
nvidia-smi
read gpu_number

echo "Choose a filenames dir path"
read data

echo "Choose a filenames path"
read filenames

echo "Choose a results dir path"
read results


CUDA_VISIBLE_DEVICES=$gpu_number nice -n1 python3 ./teachers/mmdetection/teacher.py $data $filenames $results ./teachers/mmdetection/configs/gcnet/$CONFIG_FILE ../teachers/model_zoo/$CHECKPOINT_FILE
