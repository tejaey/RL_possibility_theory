#!/bin/bash
seed_set=(0 1 2 3 4 5 6 7)

# --scenario "HalfCheetah-v2"  --env_conf "2x4"
# --scenario "HalfCheetah-v2"  --env_conf "4|2"
# --scenario "Ant-v2" --env_conf "2x4" 
# --scenario "Ant-v2" --env_conf "4x2" 
##################################################### Half-Cheetah ##################################################### 

for seed in "${seed_set[@]}"; do
   python main.py --update_rule MMQ_Quantile --n_ensemble 1  --p_sample_n 15  --env "MAmujoco"  --seed ${seed}  --epsilon 0.1 --eval_episodes 100  --random_batch  --max_timesteps 500000  --max_ep_len 25 --record_one  --start_timesteps 20000   --start_rf_timesteps 20000  --start_fm_timesteps 20000  --scenario "HalfCheetah-v2"  --env_conf "2x3"    --learn_reward  --reward_dependence sans  --device "cuda"  --IterOnlyCritic_n 10  --use_wandb --RO_reward  --RO_p 2  --negative_const 2    --d_threshold 0.04  --RO_threshold 0.035&
done

wait


for seed in "${seed_set[@]}"; do
    python main.py --update_rule HyDDPG  --env "MAmujoco" --seed ${seed}  --epsilon 0.1  --eval_episodes 100  --max_timesteps 500000  --max_ep_len 25  --record_one  --start_timesteps 20000  --n_iter_train 1  --scenario "HalfCheetah-v2"  --env_conf "2x3"  --IterOnlyCritic_n 10  --device "cuda"  --use_wandb  --RO_reward  --RO_p 2  --negative_const 2  --d_threshold 0.04  --RO_threshold 0.035&
done

wait

for seed in "${seed_set[@]}"; do
    python main.py --update_rule IDDPG  --env "MAmujoco" --seed ${seed}  --epsilon 0.1  --eval_episodes 100  --max_timesteps 500000  --max_ep_len 25  --record_one  --start_timesteps 20000  --n_iter_train 1  --scenario "HalfCheetah-v2"  --env_conf "2x3"  --IterOnlyCritic_n 10  --device "cuda"  --use_wandb  --RO_reward  --RO_p 2  --negative_const 2  --d_threshold 0.04  --RO_threshold 0.035&
done

wait

for seed in "${seed_set[@]}"; do
   python main.py --update_rule I2Q  --update_choice first  --env "MAmujoco"  --seed ${seed}  --epsilon 0.1  --eval_episodes 100  --max_timesteps 500000  --max_ep_len 25  --record_one  --start_timesteps 20000   --lambda_ 0.01    --scenario "HalfCheetah-v2"  --env_conf "2x3"  --device "cuda"  --IterOnlyCritic_n 10  --n_iter_train 1  --use_wandb  --RO_reward  --RO_p 2  --negative_const 2  --d_threshold 0.04  --RO_threshold 0.035&
done

wait

for seed in "${seed_set[@]}"; do
   python main.py --update_rule I2Q  --update_choice first  --env "MAmujoco"  --seed ${seed}  --epsilon 0.1  --eval_episodes 100  --max_timesteps 500000  --max_ep_len 25  --record_one  --start_timesteps 20000   --lambda_ 0.01    --scenario "HalfCheetah-v2"  --env_conf "2x3"  --device "cuda"  --IterOnlyCritic_n 1  --n_iter_train 10  --use_wandb  --RO_reward  --RO_p 2  --negative_const 2  --d_threshold 0.04  --RO_threshold 0.035&
done

wait


python main.py --update_rule IDDPG  --env "MAmujoco" --seed 0 --epsilon 0.1  --eval_episodes 100  --max_timesteps 500000  --max_ep_len 25  --record_one  --start_timesteps 20000  --n_iter_train 1  --scenario "HalfCheetah-v2"  --env_conf "2x3"  --IterOnlyCritic_n 10  --device "cuda"  --use_wandb  --RO_reward  --RO_p 2  --negative_const 2  --d_threshold 0.04  --RO_threshold 0.035
