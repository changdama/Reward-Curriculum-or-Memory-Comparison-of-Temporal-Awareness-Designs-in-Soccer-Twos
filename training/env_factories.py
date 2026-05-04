"""
env_factories.py (multiagent_player 版)

所有实验都使用 EnvType.multiagent_player:
  - obs: Dict[int, ndarray]   {0: obs0, 1: obs1, 2: obs2, 3: obs3}
  - reward: Dict[int, float]
  - done: Dict[int, bool]
  - action: Dict[int, ndarray] (MultiDiscrete)

4 个 player 共享同一个 policy ("default")，通过 multiagent config 配置。
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gym
import soccer_twos
from ray.rllib import MultiAgentEnv


class RLLibWrapper(gym.core.Wrapper, MultiAgentEnv):
    """A RLLib wrapper so our env can inherit from MultiAgentEnv."""
    pass


def _setup_worker_id(env_config):
    if hasattr(env_config, "worker_index"):
        env_config["worker_id"] = (
            env_config.worker_index * env_config.get("num_envs_per_worker", 1)
            + env_config.vector_index
        )


# ===== Experiment 1: Baseline PPO (multiagent, no modifications) =====
def create_env_baseline(env_config):
    _setup_worker_id(env_config)
    env = soccer_twos.make(**env_config)
    return RLLibWrapper(env)


# ===== Experiment 2: PPO + Constant Time Penalty =====
def create_env_constant_penalty(env_config):
    from wrappers.constant_penalty_wrapper import ConstantPenaltyWrapper
    _setup_worker_id(env_config)
    env = soccer_twos.make(**env_config)
    env = ConstantPenaltyWrapper(env, time_penalty=-0.001)
    return RLLibWrapper(env)


# ===== Experiment 3: PPO + Adaptive TTG (no curriculum) =====
def create_env_adaptive_ttg(env_config):
    from wrappers.adaptive_ttg_wrapper import AdaptiveTTGWrapper
    from wrappers.obs_augment_wrapper import ObsAugmentWrapper
    _setup_worker_id(env_config)
    env = soccer_twos.make(**env_config)
    env = AdaptiveTTGWrapper(env, temporal_pressure=1.0)  # 固定 1.0
    env = ObsAugmentWrapper(env)
    return RLLibWrapper(env)


# ===== Experiment 4: PPO + Adaptive TTG + Temporal Curriculum =====
def create_env_adaptive_ttg_curriculum(env_config):
    from wrappers.adaptive_ttg_wrapper import AdaptiveTTGWrapper
    from wrappers.obs_augment_wrapper import ObsAugmentWrapper
    _setup_worker_id(env_config)
    env = soccer_twos.make(**env_config)
    env = AdaptiveTTGWrapper(env, temporal_pressure=0.0)  # callback 控制
    env = ObsAugmentWrapper(env)
    return RLLibWrapper(env)


# ===== Experiment 5: LSTM (same wrappers as exp4) =====
def create_env_lstm_full(env_config):
    return create_env_adaptive_ttg_curriculum(env_config)


def create_env_ttg_v2(env_config):
    """exp7: AdaptiveTTG V2 + Curriculum + ObsAugment"""
    from wrappers.adaptive_ttg_wrapper_v2 import AdaptiveTTGWrapperV2
    from wrappers.obs_augment_wrapper import ObsAugmentWrapper

    _setup_worker_id(env_config)
    env = soccer_twos.make(**env_config)
    env = AdaptiveTTGWrapperV2(
        env,
        alpha_base=0.0005,
        ball_approach_coeff=0.0002,
        ball_advance_coeff=0.0003,
        goal_reward_multiplier=5.0,
        concede_penalty_multiplier=3.0,
        temporal_pressure=0.0,
    )
    env = ObsAugmentWrapper(env)
    return RLLibWrapper(env)


def create_env_potential(env_config):
    """exp8: Potential-based shaping (Ng et al. 1999)"""
    from wrappers.potential_shaping_wrapper import PotentialShapingWrapper

    _setup_worker_id(env_config)
    env = soccer_twos.make(**env_config)
    env = PotentialShapingWrapper(
        env,
        gamma=0.99,
        potential_scale=0.05,
    )
    return RLLibWrapper(env)


def create_env_ttg_v4(env_config):
    """exp11: exp9 + Team Possession Bonus + Curriculum + ObsAugment"""
    from wrappers.adaptive_ttg_wrapper_v4 import AdaptiveTTGWrapperV4
    from wrappers.obs_augment_wrapper import ObsAugmentWrapper

    _setup_worker_id(env_config)
    env = soccer_twos.make(**env_config)
    env = AdaptiveTTGWrapperV4(
        env,
        alpha_base=0.001,
        ball_approach_coeff=0.0005,
        ball_advance_coeff=0.0008,
        goal_reward_multiplier=5.0,
        concede_penalty_multiplier=1.0,
        possession_bonus=0.0015,
        attack_possession_bonus=0.001,
        temporal_pressure=0.0,
    )
    env = ObsAugmentWrapper(env)
    return RLLibWrapper(env)
