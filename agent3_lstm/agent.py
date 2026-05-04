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
    "ray_results/checkpoint-3400",
)
POLICY_NAME = "default"


class LSTMAdaptiveTTGAgent(AgentInterface):
    """PPO + LSTM agent - maintains per-player hidden states + prev_action + prev_reward."""

    def __init__(self, env: gym.Env):
        super().__init__()
        self._step_counts = {}
        self._max_steps = 1000
        self._lstm_states = {}
        self._prev_actions = {}
        self._prev_rewards = {}

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

        self._init_state = self.policy.get_initial_state()

        # MultiDiscrete action space: 3 components
        self._action_space = self.policy.action_space
        self._zero_action = np.zeros(
            self._action_space.nvec.shape if hasattr(self._action_space, 'nvec') else (3,),
            dtype=np.int64
        )

    def act(self, observation: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        actions = {}
        for player_id in observation:
            augmented_obs = self._augment_obs(observation[player_id], player_id)

            if player_id not in self._lstm_states:
                self._lstm_states[player_id] = [
                    s.copy() if hasattr(s, 'copy') else s
                    for s in self._init_state
                ]
                self._prev_actions[player_id] = self._zero_action.copy()
                self._prev_rewards[player_id] = 0.0

            state_in = self._lstm_states[player_id]
            prev_action = self._prev_actions[player_id]
            prev_reward = self._prev_rewards[player_id]

            action, state_out, _ = self.policy.compute_single_action(
                augmented_obs,
                state=state_in,
                prev_action=prev_action,
                prev_reward=prev_reward,
            )

            self._lstm_states[player_id] = state_out
            self._prev_actions[player_id] = action
            # prev_reward 在 eval 里拿不到，保持 0
            actions[player_id] = action
            self._step_counts[player_id] = self._step_counts.get(player_id, 0) + 1

        return actions

    def _augment_obs(self, obs: np.ndarray, player_id: int) -> np.ndarray:
        step = self._step_counts.get(player_id, 0)
        tau = max(0, 1.0 - step / self._max_steps)
        progress = 1.0 - tau
        time_features = np.array([tau, progress, 1.0], dtype=np.float32)
        return np.concatenate([obs.astype(np.float32), time_features])