"""
debug_obs.py — 确定 multiagent_player 模式下 obs 的维度含义。

用法: python debug_obs.py

输出会显示每个 player 的 obs shape 和每维的值范围。
*** 运行后，把找到的索引填回 adaptive_ttg_wrapper.py ***
"""
import numpy as np
import soccer_twos
from soccer_twos import EnvType

print("=" * 60)
print("Soccer-Twos Observation Debug (multiagent_player)")
print("=" * 60)

env = soccer_twos.make(variation=EnvType.multiagent_player)

print(f"\nObservation Space: {env.observation_space}")
print(f"Observation Shape: {env.observation_space.shape}")
print(f"Action Space: {env.action_space}")

obs = env.reset()
print(f"\n--- Initial Observation ---")
print(f"Type: {type(obs)}")
if isinstance(obs, dict):
    for pid, o in obs.items():
        print(f"  Player {pid}: shape={np.array(o).shape}")

# 收集多步数据
all_obs = {pid: [np.array(obs[pid])] for pid in obs}

print(f"\n--- Running 20 random steps ---")
for step in range(20):
    action = {pid: env.action_space.sample() for pid in obs}
    obs, reward, done, info = env.step(action)

    for pid in obs:
        all_obs[pid].append(np.array(obs[pid]))

    print(f"\nStep {step + 1}:")
    print(f"  Rewards: { {pid: f'{r:.4f}' for pid, r in reward.items()} }")
    print(f"  Dones:   {done}")
    if isinstance(info, dict):
        for pid, inf in info.items():
            if isinstance(inf, dict):
                print(f"  Info[{pid}]: {list(inf.keys())}")

    if isinstance(done, dict) and max(done.values()):
        obs = env.reset()
        for pid in obs:
            all_obs[pid].append(np.array(obs[pid]))
        print("  [Episode reset]")

# 分析 Player 0 的 obs
print(f"\n{'=' * 60}")
print(f"--- Player 0 Observation Analysis ---")
obs_array = np.array(all_obs[0])
print(f"Shape over {len(all_obs[0])} frames: {obs_array.shape}")
print(f"\n{'Dim':>4} | {'Min':>10} | {'Max':>10} | {'Mean':>10} | {'Std':>10} | {'Const?':>6}")
print("-" * 60)
for d in range(obs_array.shape[-1]):
    col = obs_array[:, d]
    mn, mx, mean, std = col.min(), col.max(), col.mean(), col.std()
    is_const = "YES" if std < 1e-6 else ""
    print(f"{d:>4} | {mn:>10.4f} | {mx:>10.4f} | {mean:>10.4f} | {std:>10.4f} | {is_const:>6}")

print(f"\n{'=' * 60}")
print("*** 找到以下维度的索引 ***")
print("1. 球相对位置 (ball_relative_pos): 3维 → 用于 situation_multiplier, approach, advance")
print("2. 球相对速度 (ball_relative_vel): 3维")
print("3. 对方球门方向 (opponent_goal_dir): 3维")
print("提示：观察哪些维度变化最大，尤其是最后 12~16 维")
print(f"{'=' * 60}")

env.close()
