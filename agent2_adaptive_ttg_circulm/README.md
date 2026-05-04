# Adaptive TTG Agent with Curriculm

**Agent name:** Agent2

**Author(s):** Group12\_Changda Ma cma326@gatech.edu

## Description

PPO agent trained with **multiagent\_player** variation (4 players share one policy).
Uses **Adaptive Time-to-Go Aware Reward Shaping** and **Temporal Curriculum Learning**.

* Reward: Dynamic time penalty adapting to remaining time and game situation
* Observation: 3 temporal features appended (τ, progress, pressure)
* Curriculum: 4-phase temporal pressure schedule (0→0.3→0.6→1.0)

