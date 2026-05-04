"""
extract_checkpoint.py — 训练完成后提取 checkpoint 并打包 agent。

用法:
    python extract_checkpoint.py \\
        --experiment ray_results/exp4_adaptive_ttg_curriculum \\
        --agent agents/agent_adaptive_ttg

步骤：
1. 从 ray_results 找到最新 checkpoint
2. 复制 checkpoint + params.pkl 到 agent 目录
3. 自动更新 agent.py 中的 CHECKPOINT_PATH
4. 打包为 .zip
"""
import os
import re
import shutil
import argparse


def find_best_checkpoint(experiment_dir):
    checkpoints = []
    for root, dirs, files in os.walk(experiment_dir):
        for f in files:
            match = re.match(r'^checkpoint-(\d+)$', f)
            if match:
                checkpoints.append((int(match.group(1)), os.path.join(root, f)))

    if not checkpoints:
        print(f"[ERROR] No checkpoints found in {experiment_dir}")
        return None

    checkpoints.sort(key=lambda x: x[0])
    best_num, best_path = checkpoints[-1]
    print(f"[OK] Found checkpoint-{best_num}: {best_path}")
    return best_path


def package_agent(checkpoint_path, agent_dir):
    if not os.path.exists(agent_dir):
        print(f"[ERROR] Agent directory {agent_dir} not found")
        return False

    checkpoint_dir = os.path.dirname(checkpoint_path)
    checkpoint_name = os.path.basename(checkpoint_path)

    dest_dir = os.path.join(agent_dir, "ray_results")
    os.makedirs(dest_dir, exist_ok=True)

    dest_ckpt = os.path.join(dest_dir, checkpoint_name)
    shutil.copy2(checkpoint_path, dest_ckpt)
    print(f"[OK] Copied checkpoint to {dest_ckpt}")

    for suffix in [".tune_metadata", ".is_checkpoint"]:
        src = checkpoint_path + suffix
        if os.path.exists(src):
            shutil.copy2(src, dest_ckpt + suffix)
            print(f"[OK] Copied {suffix}")

    params_found = False
    for search_dir in [checkpoint_dir, os.path.dirname(checkpoint_dir)]:
        params_src = os.path.join(search_dir, "params.pkl")
        if os.path.exists(params_src):
            shutil.copy2(params_src, os.path.join(dest_dir, "params.pkl"))
            print(f"[OK] Copied params.pkl from {search_dir}")
            params_found = True
            break

    if not params_found:
        print("[WARN] params.pkl not found! Agent may fail to load.")

    agent_py = os.path.join(agent_dir, "agent.py")
    if os.path.exists(agent_py):
        with open(agent_py, 'r') as f:
            content = f.read()
        new_path = f"ray_results/{checkpoint_name}"
        content = re.sub(
            r'CHECKPOINT_PATH\s*=\s*os\.path\.join\([^)]+\)',
            f'CHECKPOINT_PATH = os.path.join(\n'
            f'    os.path.dirname(os.path.abspath(__file__)),\n'
            f'    "{new_path}",\n'
            f')',
            content
        )
        with open(agent_py, 'w') as f:
            f.write(content)
        print(f"[OK] Updated CHECKPOINT_PATH → {new_path}")

    agent_name = os.path.basename(os.path.normpath(agent_dir))
    parent_dir = os.path.dirname(os.path.normpath(agent_dir))
    zip_path = shutil.make_archive(agent_name, 'zip', parent_dir, agent_name)
    print(f"[OK] Created {zip_path}")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--agent", required=True)
    args = parser.parse_args()

    checkpoint = find_best_checkpoint(args.experiment)
    if checkpoint:
        package_agent(checkpoint, args.agent)
        print(f"\nTest with: python -m soccer_twos.watch -m {args.agent}")
