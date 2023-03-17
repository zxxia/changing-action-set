# python src/load_balance/run.py --result-folder ./results/debug_train \
#     train --num-ep 500 --model-folder ./models/debug_train \
#     --model-save-interval 10

python3 src/load_balance/run.py --num-workers 10 \
    --service-rates 0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95 1.05 \
    --result-folder ./results/regular_value_network/ \
    train \
    --model-folder ./results/parameters/regular_value_network/
