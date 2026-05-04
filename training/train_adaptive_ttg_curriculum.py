
import ray
from ray import tune
from soccer_twos import EnvType
from env_factories import create_env_adaptive_ttg_curriculum
from callbacks import TemporalCurriculumCallback

NUM_ENVS_PER_WORKER = 3

if __name__ == "__main__":
    ray.init()
    tune.registry.register_env("Soccer", create_env_adaptive_ttg_curriculum)

    temp_env = create_env_adaptive_ttg_curriculum({"variation": EnvType.multiagent_player})
    obs_space = temp_env.observation_space
    act_space = temp_env.action_space
    temp_env.close()

    analysis = tune.run(
        "PPO",
        name="exp4_adaptive_ttg_curriculum",
        config={
            "num_gpus": 0,
            "num_workers": 4,
            "num_envs_per_worker": NUM_ENVS_PER_WORKER,
            "log_level": "INFO",
            "framework": "torch",
            "callbacks": TemporalCurriculumCallback,
            "multiagent": {
                "policies": {
                    "default": (None, obs_space, act_space, {}),
                },
                "policy_mapping_fn": lambda agent_id, **kwargs: "default",
                "policies_to_train": ["default"],
            },
            "env": "Soccer",
            "env_config": {
                "num_envs_per_worker": NUM_ENVS_PER_WORKER,
                "variation": EnvType.multiagent_player,
            },
            "model": {
                "vf_share_layers": True,
                "fcnet_hiddens": [512, 512],
                "fcnet_activation": "relu",
            },
            "lr": 3e-4, "lambda": 0.95, "gamma": 0.99,
            "entropy_coeff": 0.01, "clip_param": 0.2,
            "num_sgd_iter": 10, "sgd_minibatch_size": 2048,
            "rollout_fragment_length": 1000, "train_batch_size": 12000,
            "batch_mode": "truncate_episodes",
        },
        stop={"timesteps_total": 15_000_000, "time_total_s": 7200},
        checkpoint_freq=50, checkpoint_at_end=True,
        local_dir="./ray_results",
    )
    best_trial = analysis.get_best_trial("episode_reward_mean", mode="max")
    best_checkpoint = analysis.get_best_checkpoint(trial=best_trial, metric="episode_reward_mean", mode="max")
    print(f"Best checkpoint: {best_checkpoint}")
