rm -rf results_expanded.zip
rm -rf results
rm -rf results_expanded

sh body_landmark.sh

python tps_whole_han.py --logo_root data/deepfashion2/test/image/data_logo_blue/logo_whole --model_root data/deepfashion2/test/image/data_logo_blue/model --output_root ./results

mkdir results_expanded
python mov.py

zip results_expanded.zip results_expanded/ -r