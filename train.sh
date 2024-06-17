cd src
echo "Training the baseline with 4M samples"
python train.py -c baseline_77MRT_4M.yaml

echo "Training the baseline with 5M samples"
python train.py -c baseline_77MRT_5M.yaml