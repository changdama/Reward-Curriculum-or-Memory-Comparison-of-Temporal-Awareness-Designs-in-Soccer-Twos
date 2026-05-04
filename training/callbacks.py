"""
TemporalCurriculumCallback: 根据训练进度动态调整 temporal_pressure。
Phase 0 (0~2M steps):  pressure = 0.0
Phase 1 (2M~5M):       pressure = 0.3
Phase 2 (5M~8M):       pressure = 0.6
Phase 3 (8M+):         pressure = 1.0
"""
from ray.rllib.agents.callbacks import DefaultCallbacks


class TemporalCurriculumCallback(DefaultCallbacks):
    PHASE_SCHEDULE = [
        (0,          0.0),
        (2_000_000,  0.3),
        (5_000_000,  0.6),
        (8_000_000,  1.0),
    ]

    def on_train_result(self, trainer, result, **kwargs):
        total_steps = result["timesteps_total"]
        current_pressure = 0.0
        current_phase = 0
        for i, (threshold, pressure) in enumerate(self.PHASE_SCHEDULE):
            if total_steps >= threshold:
                current_pressure = pressure
                current_phase = i

        def _update_worker(worker):
            def _update_env(env):
                self._update_env_pressure(env, current_pressure)
            worker.foreach_env(_update_env)

        trainer.workers.foreach_worker(_update_worker)

        result["custom_metrics"] = result.get("custom_metrics", {})
        result["custom_metrics"]["temporal_pressure"] = current_pressure
        result["custom_metrics"]["curriculum_phase"] = current_phase

        if current_phase > 0:
            step_in_phase = total_steps - self.PHASE_SCHEDULE[current_phase][0]
            if step_in_phase < 50000:
                print(f"[Curriculum] Phase {current_phase} (pressure={current_pressure}) at step {total_steps}")

    @staticmethod
    def _update_env_pressure(env, pressure):
        current = env
        depth = 0
        while depth < 10:
            if hasattr(current, 'set_temporal_pressure'):
                current.set_temporal_pressure(pressure)
                return
            if hasattr(current, 'env'):
                current = current.env
                depth += 1
            else:
                break
