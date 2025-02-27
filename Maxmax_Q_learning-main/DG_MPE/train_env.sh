seed_set=(0 1 2 3 4 5 6 7)


################################################# Differential Game ##################################################
# for seed in "${seed_set[@]}"; do
#    python main.py --m 0.13 --update_rule "MMQ_Quantile" --sub_reward 0.15 --seed ${seed} --epsilon 0.1 --eval_episodes 100 --start_timesteps 20000 --start_rf_timesteps 20000 --start_fm_timesteps 20000 --max_timesteps 500000 --use_wandb  --n_ensemble 1 --p_sample_n 15 --random_batch --IterOnlyCritic_n 10 --max_ep_len 25  --learn_reward --reward_dependence "ns" --record_one --shift_reward --negative_const 2 --device "cuda"&
# done
# 
# wait


################################################# MPE ##################################################
# for seed in "${seed_set[@]}"; do
#    python main.py --update_rule "MMQ_Quantile"  --scenario "simple_spread_RO1"   --n_ensemble 1  --p_sample_n 15  --env "MPE"  --seed ${seed} --epsilon 0.1 --eval_episodes 100  --max_timesteps 500000  --use_wandb --max_ep_len 25  --record_one  --start_timesteps 20000 --device "cuda" --start_rf_timesteps 20000 --start_fm_timesteps 20000  --learn_reward --reward_dependence "ns" --random_batch  --IterOnlyCritic_n 10&
# done



