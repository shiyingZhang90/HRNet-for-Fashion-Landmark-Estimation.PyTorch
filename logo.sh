# sh logo.sh LOGO_ROOT MODEL_ROOT WHITE/BLACK
# sh logo.sh data/deepfashion2/test/image/data_logo/logo data/deepfashion2/test/image/data_logo/model white
# sh logo.sh data/deepfashion2/test/image/data_logo_black/logo data/deepfashion2/test/image/data_logo_black/model black
LOGO_DIR=$1
MODEL_DIR=$2
BL=$3

#download zip fill from google storage bucket
#gsutil cp gs://results_small_size/${ROOT_DIR}.zip .
#create results directory
#mkdir ${ROOT_DIR}

#unzip ${ROOT_DIR}.zip -d ${ROOT_DIR}

#mv ${ROOT_DIR}/${ROOT_DIR} ${ROOT_DIR}/source

# There are some file end with JPG, rename all of them to jpg
#rename  's/(.*)\.JPG/$1\.jpg/' ${ROOT_DIR}/source/*

#Some images are much larger than 900*1200, we will resize all of them to width 900, reserving aspect ratio
#This step need imagemagick to be installed
#convert ${ROOT_DIR}/source/*.jpg[900x] -set filename:base %[basename] ${ROOT_DIR}/source/%[filename:base].jpg
rm -rf results_expanded.zip
rm -rf visualize_landmark.zip
rm -rf segmentation.zip
rm -rf results
rm -rf results_expanded
rm -rf visualize_landmark
rm -rf segmentation

rm -rf *.jpg

python ./Self-Correction-Human-Parsing/evaluate_atr.py \
--root ${LOGO_DIR} --logo \
--output '/home/ella/ZMO/hanyang/logo_on_shirts/hrnet/segmentation/segmentation_logo/gray_atr' --output_vis '/home/ella/ZMO/hanyang/logo_on_shirts/hrnet/segmentation/segmentation_logo/vis_atr' \
--restore-weight '/home/ella/ZMO/ella/FACE_SWAP_ALGORITHM/Self-Correction-Human-Parsing/exp-schp-201908301523-atr.pth'

#### generate a rough bbox
python generate_bboxfile.py --generate_big_bbox --segroot /home/ella/ZMO/hanyang/logo_on_shirts/hrnet/segmentation/segmentation_logo/gray_atr \
--logo --dataroot ${LOGO_DIR}
python tools/test.py \
    --cfg experiments/deepfashion2/hrnet/w48_384x288_adam_lr1e-3.yaml \
    TEST.MODEL_FILE models/pose_hrnet-w48_384x288-deepfashion2_mAP_0.7017.pth \
    DATASET.ROOT ${LOGO_DIR} TEST.DEEPFASHION2_BBOX_FILE data/bbox_result_test.json DATASET.TEST_SET test DATASET.USE_DEEPFASHION1 True TAG logo



python ./Self-Correction-Human-Parsing/evaluate_atr.py \
--root ${MODEL_DIR} \
--output '/home/ella/ZMO/hanyang/logo_on_shirts/hrnet/segmentation/segmentation_model/gray_atr' --output_vis '/home/ella/ZMO/hanyang/logo_on_shirts/hrnet/segmentation/segmentation_model/vis_atr' \
--restore-weight '/home/ella/ZMO/ella/FACE_SWAP_ALGORITHM/Self-Correction-Human-Parsing/exp-schp-201908301523-atr.pth'

python generate_bboxfile.py --generate_big_bbox --segroot /home/ella/ZMO/hanyang/logo_on_shirts/hrnet/segmentation/segmentation_model/gray_atr \
--dataroot ${MODEL_DIR}


#### generate a rough landmark



#### generate an accurate landmark
python tools/test.py \
    --cfg experiments/deepfashion2/hrnet/w48_384x288_adam_lr1e-3.yaml \
    TEST.MODEL_FILE models/pose_hrnet-w48_384x288-deepfashion2_mAP_0.7017.pth \
    DATASET.ROOT ${MODEL_DIR} TEST.DEEPFASHION2_BBOX_FILE data/bbox_result_test.json DATASET.TEST_SET test DATASET.USE_DEEPFASHION1 True TAG model



#### tps
if [ $BL = "black" ]; then
    python tps_black.py --logo_root ${LOGO_DIR} --model_root ${MODEL_DIR} --output_root ./results
elif [ $BL = "white" ]; then
    python tps_white.py --logo_root ${LOGO_DIR} --model_root ${MODEL_DIR} --output_root ./results
elif [ $BL = "other" ]; then
    python tps_other.py --logo_root ${LOGO_DIR} --model_root ${MODEL_DIR} --output_root ./results
elif [ $BL = "green" ]; then
    python tps_green.py --logo_root ${LOGO_DIR} --model_root ${MODEL_DIR} --output_root ./results
elif [ $BL = "whole" ]; then
    python tps_whole.py --logo_root ${LOGO_DIR} --model_root ${MODEL_DIR} --output_root ./results

else
    echo "error: no color specified."
fi

mkdir results_expanded
python mov.py
rm -rf results_expanded/*warp*
zip results_expanded.zip results_expanded/ -r
zip visualize_landmark.zip visualize_landmark -r
zip segmentation.zip segmentation -r