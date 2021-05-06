
MODEL_DIR=data/deepfashion2/test/image/data_logo_blue/model_mask

python ./Self-Correction-Human-Parsing/evaluate_atr.py \
--root ${MODEL_DIR} \
--output '/home/ella/ZMO/hanyang/logo_on_shirts/hrnet/segmentation/segmentation_model/gray_atr' --output_vis '/home/ella/ZMO/hanyang/logo_on_shirts/hrnet/segmentation/segmentation_model/vis_atr' \
--restore-weight '/home/ella/ZMO/ella/FACE_SWAP_ALGORITHM/Self-Correction-Human-Parsing/exp-schp-201908301523-atr.pth'

python generate_bboxfile.py --generate_big_bbox --segroot /home/ella/ZMO/hanyang/logo_on_shirts/hrnet/segmentation/segmentation_model/gray_atr \
--dataroot ${MODEL_DIR}



#### generate an accurate landmark
python tools/test.py \
    --cfg experiments/deepfashion2/hrnet/w48_384x288_adam_lr1e-3.yaml \
    TEST.MODEL_FILE models/pose_hrnet-w48_384x288-deepfashion2_mAP_0.7017.pth \
    DATASET.ROOT ${MODEL_DIR} TEST.DEEPFASHION2_BBOX_FILE data/bbox_result_test.json DATASET.TEST_SET test DATASET.USE_DEEPFASHION1 True TAG model

