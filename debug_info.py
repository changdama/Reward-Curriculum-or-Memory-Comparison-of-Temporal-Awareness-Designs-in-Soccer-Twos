"""
debug_info.py — 查看 info['player_info'] 和 info['ball_info'] 的具体内容。
运行: python debug_info.py
"""
import numpy as np
import soccer_twos
from soccer_twos import EnvType

env = soccer_twos.make(variation=EnvType.multiagent_player)
obs = env.reset()

print("Running 5 steps to inspect info dict...\n")
for step in range(5):
    action = {pid: env.action_space.sample() for pid in obs}
    obs, reward, done, info = env.step(action)

    print(f"=== Step {step + 1} ===")
    for pid in sorted(info.keys()):
        if isinstance(info[pid], dict):
            print(f"\n  Player {pid}:")
            for key, val in info[pid].items():
                print(f"    {key}: {val}")
                if isinstance(val, dict):
                    for k2, v2 in val.items():
                        print(f"      {k2}: {v2}")
    print()

env.close()
