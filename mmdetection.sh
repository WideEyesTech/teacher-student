#!/bin/bash
echo "Tested config: NVCC 9.0, NVIDIA DRIVERS 440.33.01, GCC 5.5, CUDA 9.0"

if [ ! -d ./models/mmdetection ]; then
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
        git clone https://github.com/open-mmlab/mmdetection.git ./models/mmdetection
        cd ./models/mmdetection
        python3 -m pip install -r requirements/build.txt --user
        python3 -m pip install -v -e . --user
        cd ../../
    else
        git clone https://github.com/open-mmlab/mmdetection.git ./models/mmdetection
        cd ./models/mmdetection
        python3 -m pip install -r requirements/build.txt --user
        python3 -m pip install -v -e . --user
        cd ../../
    fi
    
    cp ./models/teacher.py ./models/mmdetection/teacher.py
    
    echo "Choose a teacher: ATSS, GCNET"
    read teacher
    if [[ "$teacher" != "ATSS" ]] && [[ "$teacher" != "GCNET" ]]; then
        echo -n "Incorrect answer"
        exit
    fi
    
    if [ ! -d ./models/model_zoo ]; then
        mkdir ./models/model_zoo
        if [[ "$teacher" == "ATSS" ]]; then
            export CONFIG="./models/mmdetection/configs/atss/atss_r50_fpn_1x.py"
            export CHECKPOINT="atss_r50_fpn_1x_20200113-a7aa251e.pth"
            wget "https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/models/atss/atss_r50_fpn_1x_20200113-a7aa251e.pth" -P ./models/model_zoo
            elif [[ "$teacher" == "GCNET" ]]; then
            export CONFIG="./models/mmdetection/configs/gcnet/mask_rcnn_r16_gcb_c3-c5_r50_fpn_1x.py"
            export CHECKPOINT="mask_rcnn_r4_gcb_c3-c5_r50_fpn_1x_20190602-18ae2dfd.pth"
            wget "https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/models/gcnet/mask_rcnn_r4_gcb_c3-c5_r50_fpn_1x_20190602-18ae2dfd.pth" -P ./models/model_zoo
        fi
    fi
fi

if [ ! $GPU_NUMBER ]; then
    echo "Choose a GPU number, OPTIONS:"
    nvidia-smi
    read gpu_number
    export GPU_NUMBER=$config
fi

if [ ! $DATA ]; then
    echo "Choose a filenames dir path"
    read -e data
    export DATA=$data
fi

if [ ! $FILENAMES ]; then
    echo "Choose a filenames path"
    read -e filenames
    export FILENAMES=$filenames
fi

if [ ! $RESULTS ]; then
    echo "Choose a results dir path"
    read -e results
    export RESULTS=$results
fi

if [ ! $CONFIG ]; then
    echo "Choose a config file"
    read -e config
    export CONFIG=$config
fi

if [ ! $CHECKPOINT ]; then
    echo "Choose a checkpoint file"
    read -e checkpoint
    export CHECKPOINT=$checkpoint
fi

CUDA_VISIBLE_DEVICES=$GPU_NUMBER nice -n1 python3 ./models/mmdetection/teacher.py $DATA $FILENAMES $RESULTS $CONFIG $CHECKPOINT
