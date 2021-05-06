#rm -rf results_expanded.zip
rm -rf results_mask
rm -rf results_expanded_mask

sh body_landmark_mask.sh

python tps_whole_mask.py --logo_root data/deepfashion2/test/image/data_logo_blue/logo_whole --model_root data/deepfashion2/test/image/data_logo_blue/model_mask --output_root ./results_mask

mkdir results_expanded_mask
python mov_mask.py

zip results_expanded_mask.zip results_expanded_mask/ -r
