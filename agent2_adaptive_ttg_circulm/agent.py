
import os
import pickle
from typing import Dict

import gym
import numpy as np
import ray
from ray import tune
from ray.rllib.env.base_env import BaseEnv
from ray.tune.registry import get_trainable_cls
from soccer_twos import AgentInterface


ALGORITHM = "PPO"
CHECKPOINT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "ray_results/checkpoint-3750",
)
POLICY_NAME = "default"


class AdaptiveTTGAgent(AgentInterface):
    """PPO + Adaptive TTG agent (multiagent_player trained)."""

    def __init__(self, env: gym.Env):
        super().__init__()
        self._step_counts = {}
        self._max_steps = 1000

        ray.init(ignore_reinit_error=True)

        config_path = ""
        if CHECKPOINT_PATH:
            config_dir = os.path.dirname(CHECKPOINT_PATH)
            config_path = os.path.join(config_dir, "params.pkl")
            if not os.path.exists(config_path):
                config_path = os.path.join(config_dir, "../params.pkl")

        if os.path.exists(config_path):
            with open(config_path, "rb") as f:
                config = pickle.load(f)
        else:
            raise ValueError(f"Could not find params.pkl! Looked at: {config_path}")

        config["num_workers"] = 0
        config["num_gpus"] = 0

        tune.registry.register_env("DummyEnv", lambda *_: BaseEnv())
        config["env"] = "DummyEnv"

        cls = get_trainable_cls(ALGORITHM)
        agent = cls(env=config["env"], config=config)
        agent.restore(CHECKPOINT_PATH)
        self.policy = agent.get_policy(POLICY_NAME)

    def act(self, observation: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        actions = {}
        for player_id in observation:
            augmented_obs = self._augment_obs(observation[player_id], player_id)
            action, *_ = self.policy.compute_single_action(augmented_obs)
            actions[player_id] = action
            self._step_counts[player_id] = self._step_counts.get(player_id, 0) + 1
        return actions

    def _augment_obs(self, obs: np.ndarray, player_id: int) -> np.ndarray:
        step = self._step_counts.get(player_id, 0)
        tau = max(0, 1.0 - step / self._max_steps)
        progress = 1.0 - tau
        temporal_pressure = 1.0
        time_features = np.array([tau, progress, temporal_pressure], dtype=np.float32)
        return np.concatenate([obs.astype(np.float32), time_features])