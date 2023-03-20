for agent in rl LeastWork ShortestProcessingTime UniformRandom; do
    echo $agent
    python3 src/load_balance/run.py --num-workers 10 \
        --pretrained-model ./results/parameters/regular_value_network/model_ep_09900.ckpt \
        --service-rates 0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95 1.05 \
        --result-folder ./results/regular_value_network/ \
        test --agent $agent
done
