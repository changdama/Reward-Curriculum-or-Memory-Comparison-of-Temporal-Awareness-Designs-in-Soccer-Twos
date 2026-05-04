"""
plot_agent_curves.py - Robust version that selects the most complete CSV.
"""
import os
import glob
import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXPERIMENTS = [
    {
        "name": "Agent 1: Adaptive TTG",
        "exp_dir": "ray_results/exp3_adaptive_ttg",
        "color": "#1f77b4",
    },
    {
        "name": "Agent 2: TTG + Curriculum",
        "exp_dir": "ray_results/exp4_adaptive_ttg_curriculum",
        "color": "#ff7f0e",
    },
    {
        "name": "Agent 3: LSTM + TTG + Curriculum",
        "exp_dir": "ray_results/exp5_lstm_adaptive_ttg_curriculum",
        "color": "#2ca02c",
    },
]

OUTPUT_PATH = "training_curves.png"
SMOOTHING_WINDOW = 20


def find_best_progress_csv(exp_dir):
    """Find all progress.csv under exp_dir, return the one with largest
    timesteps_total range (i.e. most complete training)."""
    if not os.path.exists(exp_dir):
        return None, None

    candidates = glob.glob(os.path.join(exp_dir, "**", "progress.csv"),
                           recursive=True)

    best_csv = None
    best_max_step = -1
    best_df = None

    for csv in candidates:
        if os.path.getsize(csv) < 1024:
            continue
        try:
            df = pd.read_csv(csv)
            if "timesteps_total" not in df.columns or len(df) == 0:
                continue
            max_step = df["timesteps_total"].max()
            if max_step > best_max_step:
                best_max_step = max_step
                best_csv = csv
                best_df = df
        except Exception as e:
            print(f"  [skip] {csv}: {e}")

    return best_csv, best_df


def main():
    print("=" * 60)
    print(f"Working dir: {os.getcwd()}")
    print("=" * 60)

    fig, ax = plt.subplots(figsize=(8, 5))
    loaded = 0
    max_step_overall = 0

    for cfg in EXPERIMENTS:
        print(f"\n--- {cfg['name']} ---")
        csv_path, df = find_best_progress_csv(cfg["exp_dir"])
        if df is None:
            print(f"  [WARN] No usable progress.csv under {cfg['exp_dir']}")
            print(f"         You may need to download it from PACE.")
            continue

        print(f"  CSV: {csv_path}")
        print(f"  Iterations: {len(df)}")
        print(f"  Step range: {df['timesteps_total'].min():,} "
              f"-> {df['timesteps_total'].max():,}")
        print(f"  Final reward: {df['episode_reward_mean'].iloc[-1]:.3f}")

        x = df["timesteps_total"]
        y = df["episode_reward_mean"]
        max_step_overall = max(max_step_overall, x.max())

        ax.plot(x, y, color=cfg["color"], alpha=0.20, linewidth=0.7)

        if len(df) > SMOOTHING_WINDOW:
            y_smooth = y.rolling(window=SMOOTHING_WINDOW,
                                 min_periods=1).mean()
            ax.plot(x, y_smooth, color=cfg["color"], linewidth=2.0,
                    label=cfg["name"])
        else:
            ax.plot(x, y, color=cfg["color"], linewidth=2.0,
                    label=cfg["name"])

        loaded += 1

    if loaded == 0:
        print("\n[ERROR] No data loaded.")
        sys.exit(1)

    ax.set_xlabel("Training Steps", fontsize=12)
    ax.set_ylabel("Mean Episode Reward", fontsize=12)
    ax.set_title(
        f"Training Curves: Three Temporal-Awareness Agents "
        f"({loaded}/{len(EXPERIMENTS)} loaded)",
        fontsize=12)
    ax.ticklabel_format(axis="x", style="sci", scilimits=(6, 6))
    ax.legend(loc="lower right", fontsize=10, framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle="--")

    # Phase markers (only meaningful if Agent 2/3 loaded)
    if max_step_overall > 2_000_000:
        for boundary, label in [(2_000_000, "P1"),
                                (5_000_000, "P2"),
                                (8_000_000, "P3")]:
            if boundary < max_step_overall:
                ax.axvline(boundary, color="gray", alpha=0.3,
                           linestyle=":", linewidth=0.8)
                ymin, ymax = ax.get_ylim()
                ax.text(boundary, ymax * 0.95, label,
                        fontsize=8, color="gray", ha="center", va="top")

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight")
    abs_path = os.path.abspath(OUTPUT_PATH)
    print(f"\n[OK] Saved to: {abs_path}")
    print(f"     ({os.path.getsize(OUTPUT_PATH) / 1024:.1f} KB)")


if __name__ == "__main__":
    main()