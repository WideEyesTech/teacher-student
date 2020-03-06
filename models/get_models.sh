#!/bin/bash

wget "https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/models/atss/atss_r50_fpn_1x_20200113-a7aa251e.pth" -P ./model_zoo     
wget "https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/models/gcnet/mask_rcnn_r4_gcb_c3-c5_r50_fpn_1x_20190602-18ae2dfd.pth" -P ./model_zoo
wget "http://downloads.zjulearning.org.cn/ttfnet/ttfnet18_1x-fe6884.pth" -P ./model_zoo