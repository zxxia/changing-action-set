# command to reproduce Hongzi's results at https://github.com/hongzimao/input_driven_rl_example/blob/master/figures/regular_value_network_testing.png
# python3 src/load_balance/run.py --num-workers 10 \
#     --service-rates 0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95 1.05 \
#     --result-folder ./results/regular_value_network/ \
#     train \
#     --model-folder ./results/parameters/regular_value_network/

git_summary () {
    git_summary=$1/git_summary.txt
    patch=$1/uncommited_changes.patch
    date > $git_summary
    printf "branch: " >> $git_summary
    git rev-parse --abbrev-ref HEAD >> $git_summary
    printf "commit: " >> $git_summary
    git rev-parse HEAD >> $git_summary
    git diff HEAD > $patch
}

exp_name=regular_value_network_masked
mkdir -p ./results/parameters/${exp_name}
git_summary ./results/parameters/${exp_name}/
python3 src/load_balance/run.py --num-workers 10 \
    --service-rates 0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95 1.05 \
    --result-folder ./results/${exp_name}/ \
    train \
    --model-folder ./results/parameters/${exp_name}/
