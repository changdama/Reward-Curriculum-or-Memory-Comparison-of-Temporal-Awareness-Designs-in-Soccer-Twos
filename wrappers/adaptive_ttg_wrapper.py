
import gym
import numpy as np


class AdaptiveTTGWrapper(gym.Wrapper):

    def __init__(
        self,
        env,
        alpha_base=0.002,
        beta=2.0,
        max_episode_steps=1000,
        ball_approach_coeff=0.001,
        ball_advance_coeff=0.002,
        temporal_pressure=1.0,
    ):
        super().__init__(env)
        self.alpha_base = alpha_base
        self.beta = beta
        self.max_episode_steps = max_episode_steps
        self.ball_approach_coeff = ball_approach_coeff
        self.ball_advance_coeff = ball_advance_coeff
        self.temporal_pressure = temporal_pressure
        self._step_count = 0
        self._prev_ball_pos = None       # [x, y, z]
        self._prev_player_pos = {}       # {pid: [x, y, z]}

    def reset(self, **kwargs):
        self._step_count = 0
        self._prev_ball_pos = None
        self._prev_player_pos = {}
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._step_count += 1

        if isinstance(reward, dict) and isinstance(info, dict):
            for player_id in reward:
                if player_id in info and isinstance(info[player_id], dict):
                    shaped = self._compute_shaped_reward(player_id, info[player_id])
                    reward[player_id] += shaped

            for pid in info:
                if isinstance(info[pid], dict):
                    ball_info = info[pid].get('ball_info', {})
                    if isinstance(ball_info, dict) and 'position' in ball_info:
                        self._prev_ball_pos = list(ball_info['position'])
                        break

            for pid in info:
                if isinstance(info[pid], dict):
                    player_info = info[pid].get('player_info', {})
                    if isinstance(player_info, dict) and 'position' in player_info:
                        self._prev_player_pos[pid] = list(player_info['position'])

        return obs, reward, done, info

    def _compute_shaped_reward(self, player_id, player_info_dict):
        tau = max(0, 1.0 - self._step_count / self.max_episode_steps)
        time_progress = 1.0 - tau


        ball_info = player_info_dict.get('ball_info', {})
        player_info = player_info_dict.get('player_info', {})

        ball_pos = None
        player_pos = None

        if isinstance(ball_info, dict):
            ball_pos = ball_info.get('position')
        if isinstance(player_info, dict):
            player_pos = player_info.get('position')


        situation_mult = self._get_situation_multiplier(ball_pos)
        alpha = self.alpha_base * situation_mult * self.temporal_pressure


        time_penalty = -alpha * (time_progress ** self.beta)


        approach_bonus = 0.0
        advance_bonus = 0.0

        if self.temporal_pressure > 0 and self._prev_ball_pos is not None:
            approach_bonus = self._ball_approach_reward(
                ball_pos, player_pos, player_id
            ) * self.temporal_pressure

            advance_bonus = self._ball_advance_reward(
                ball_pos
            ) * self.temporal_pressure

            if time_progress > 0.5:
                approach_bonus *= 0.3

        return time_penalty + approach_bonus + advance_bonus

    def _get_situation_multiplier(self, ball_pos):

        if ball_pos is None:
            return 1.0
        try:
            abs_x = abs(float(ball_pos[0]))
            if abs_x > 10:
                return 0.5
            elif abs_x > 5:
                return 0.8
            else:
                return 1.0
        except (IndexError, TypeError, ValueError):
            return 1.0

    def _ball_approach_reward(self, ball_pos, player_pos, player_id):

        if ball_pos is None or player_pos is None:
            return 0.0
        prev_player = self._prev_player_pos.get(player_id)
        if prev_player is None or self._prev_ball_pos is None:
            return 0.0
        try:
            curr_dist = np.sqrt(
                (float(ball_pos[0]) - float(player_pos[0])) ** 2 +
                (float(ball_pos[1]) - float(player_pos[1])) ** 2
            )
            prev_dist = np.sqrt(
                (float(self._prev_ball_pos[0]) - float(prev_player[0])) ** 2 +
                (float(self._prev_ball_pos[1]) - float(prev_player[1])) ** 2
            )
            return (prev_dist - curr_dist) * self.ball_approach_coeff
        except (IndexError, TypeError, ValueError):
            return 0.0

    def _ball_advance_reward(self, ball_pos):
        if ball_pos is None or self._prev_ball_pos is None:
            return 0.0
        try:
            curr_abs_x = abs(float(ball_pos[0]))
            prev_abs_x = abs(float(self._prev_ball_pos[0]))
            advance = (curr_abs_x - prev_abs_x) * self.ball_advance_coeff
            return max(advance, 0)
        except (IndexError, TypeError, ValueError):
            return 0.0

    def set_temporal_pressure(self, pressure: float):
        self.temporal_pressure = float(np.clip(pressure, 0.0, 1.0))
