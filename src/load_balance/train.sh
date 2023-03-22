# command to reproduce Hongzi's results at https://github.com/hongzimao/input_driven_rl_example/blob/master/figures/regular_value_network_testing.png
# python3 src/load_balance/run.py --num-workers 10 \
#     --service-rates 0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95 1.05 \
#     --result-folder ./results/regular_value_network/ \
#     train \
#     --model-folder ./results/parameters/regular_value_network/

git_summary () {
    printf "branch: " > $1
    git rev-parse --abbrev-ref HEAD >> $1
    printf "commit: " >> $1
    git rev-parse HEAD >> $1
}

git_summary ./results/parameters/regular_value_network_change_act_avail/git_summary.txt
python3 src/load_balance/run.py --num-workers 10 \
    --service-rates 0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95 1.05 \
    --result-folder ./results/regular_value_network_change_act_avail/ \
    train \
    --model-folder ./results/parameters/regular_value_network_change_act_avail/
