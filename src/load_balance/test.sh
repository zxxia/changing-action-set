# no action mask
python3 src/load_balance/run.py --num-workers 10 \
    --pretrained-model ./results/parameters/regular_value_network/model_ep_09900.ckpt \
    --service-rates 0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95 1.05 \
    --result-folder ./results/regular_value_network/ \
    test --agent LeastWork RoundRobin ShortestProcessingTime UniformRandom rl
# ./results/parameters/regular_value_network_masked_every_step/model_ep_09900.ckpt \
    # --pretrained-model ./results/parameters/regular_value_network_masked_every_step_0.7/model_ep_09900.ckpt \

# with action mask
python3 src/load_balance/run.py --num-workers 10 \
    --action-mask 1 1 1 1 1 1 1 0 0 0 \
    --pretrained-model ./results/parameters/regular_value_network/model_ep_09900.ckpt \
    --service-rates 0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95 1.05 \
    --result-folder ./results/regular_value_network/ \
    test --agent LeastWork RoundRobin ShortestProcessingTime UniformRandom rl
