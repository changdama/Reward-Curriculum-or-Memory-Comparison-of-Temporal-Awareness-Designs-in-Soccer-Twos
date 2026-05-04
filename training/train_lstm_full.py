import sys, os, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ray
from ray import tune
from soccer_twos import EnvType
from training.env_factories import create_env_lstm_full
from training.callbacks import TemporalCurriculumCallback

NUM_ENVS_PER_WORKER = 1
BASE_PORT = 22000 + random.randint(0, 1000)

if __name__ == "__main__":
    ray.init(include_dashboard=False)
    tune.registry.register_env("Soccer", create_env_lstm_full)

    temp_env = create_env_lstm_full({
        "variation": EnvType.multiagent_player,
        "base_port": BASE_PORT,
    })
    obs_space = temp_env.observation_space
    act_space = temp_env.action_space
    temp_env.close()

    analysis = tune.run(
        "PPO", name="exp5_lstm_adaptive_ttg_curriculum",
        config={
            "num_gpus": 0, "num_workers": 2,
            "num_envs_per_worker": NUM_ENVS_PER_WORKER,
            "log_level": "WARN", "framework": "torch",
            "callbacks": TemporalCurriculumCallback,
            "multiagent": {
                "policies": {"default": (None, obs_space, act_space, {})},
                "policy_mapping_fn": tune.function(lambda _: "default"),
                "policies_to_train": ["default"],
            },
            "env": "Soccer",
            "env_config": {
                "num_envs_per_worker": NUM_ENVS_PER_WORKER,
                "variation": EnvType.multiagent_player,
                "base_port": BASE_PORT + 10,
            },
            "model": {
                "use_lstm": True, "lstm_cell_size": 256,
                "max_seq_len": 32, "lstm_use_prev_action": True,
                "lstm_use_prev_reward": True, "vf_share_layers": True,
                "fcnet_hiddens": [256], "fcnet_activation": "relu",
            },
        },
        stop={"timesteps_total": 15000000},
        checkpoint_freq=100, checkpoint_at_end=True,
        local_dir="./ray_results",
        restore="./ray_results/exp5_lstm_adaptive_ttg_curriculum/PPO_Soccer_d7c88_00000_0_2026-04-23_11-48-16/checkpoint_003400/checkpoint-3400",
    )
    best_trial = analysis.get_best_trial("episode_reward_mean", mode="max")
    best_checkpoint = analysis.get_best_checkpoint(trial=best_trial, metric="episode_reward_mean", mode="max")
    print(best_trial); print(best_checkpoint); print("Done training")
