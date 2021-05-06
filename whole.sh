LOGO_DIR=$1
MODEL_DIR=$2
SEG_DIR=$3
TEMPLATE_DIR=$4
SHIFT_H=$5
SHIFT_W=$6
LAMBDA=$7
ALPHA=$8

rm -rf *.jpg
rm -rf visualize_landmark*
rm -rf results*
python ./Self-Correction-Human-Parsing/evaluate_atr.py \
--root ./template2 \
--output '/home/ella/ZMO/hanyang/logo_on_shirts/ZMO_logo_change/hrnet/segmentation/segmentation_logo/gray_atr' --output_vis '/home/ella/ZMO/hanyang/logo_on_shirts/ZMO_logo_change/hrnet/segmentation/segmentation_logo/vis_atr' \
--restore-weight '/home/ella/ZMO/ella/FACE_SWAP_ALGORITHM/Self-Correction-Human-Parsing/exp-schp-201908301523-atr.pth'


python generate_bboxfile.py --generate_big_bbox --segroot /home/ella/ZMO/hanyang/logo_on_shirts/ZMO_logo_change/hrnet/segmentation/segmentation_logo/gray_atr \
--dataroot ./template2

python tools/test.py \
    --cfg experiments/deepfashion2/hrnet/w48_384x288_adam_lr1e-3.yaml \
    TEST.MODEL_FILE models/pose_hrnet-w48_384x288-deepfashion2_mAP_0.7017.pth \
    DATASET.ROOT ./template TEST.DEEPFASHION2_BBOX_FILE data/bbox_result_test.json DATASET.TEST_SET test DATASET.USE_DEEPFASHION1 True TAG logo

python ./Self-Correction-Human-Parsing/evaluate_atr.py \
--root ${MODEL_DIR} \
--output '/home/ella/ZMO/hanyang/ZMO_logo_change/logo_on_shirts/hrnet/segmentation/segmentation_model/gray_atr' --output_vis '/home/ella/ZMO/hanyang/logo_on_shirts/ZMO_logo_change/hrnet/segmentation/segmentation_model/vis_atr' \
--restore-weight '/home/ella/ZMO/ella/FACE_SWAP_ALGORITHM/Self-Correction-Human-Parsing/exp-schp-201908301523-atr.pth'

python generate_bboxfile.py --generate_big_bbox --segroot /home/ella/ZMO/hanyang/ZMO_logo_change/logo_on_shirts/hrnet/segmentation/segmentation_model/gray_atr \
--dataroot ${MODEL_DIR}

python tools/test.py \
    --cfg experiments/deepfashion2/hrnet/w48_384x288_adam_lr1e-3.yaml \
    TEST.MODEL_FILE models/pose_hrnet-w48_384x288-deepfashion2_mAP_0.7017.pth \
    DATASET.ROOT ${MODEL_DIR} TEST.DEEPFASHION2_BBOX_FILE data/bbox_result_test.json DATASET.TEST_SET test DATASET.USE_DEEPFASHION1 True TAG model

python tps_whole_slant.py --logo_root ${LOGO_DIR} --model_root ${MODEL_DIR} --output_root ./results --seg_root ${SEG_DIR} --shift_h ${SHIFT_H} --shift_w ${SHIFT_W} --_lambda ${LAMBDA} --template_path ${TEMPLATE_DIR} --alpha ${ALPHA}

mkdir results_expanded
python mov.py
zip results_expanded.zip results_expanded/ -r
zip visualize_landmark.zip visualize_landmark -r
zip segmentation.zip segmentation -r
