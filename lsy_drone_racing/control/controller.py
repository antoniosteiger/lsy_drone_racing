from __future__ import annotations

import numpy as np
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy.typing import NDArray


class Controller(ABC):
    """Base class for controller implementations with common bookkeeping and hooks."""

    def __init__(
        self,
        obs: dict[str, NDArray[np.floating]],
        info: dict,
        config: dict,
    ):
        """Initialization of the controller.

        Args:
            obs: The initial observation of the environment's state.
            info: The initial environment information from reset.
            config: Race configuration.
        """
        # Store inputs
        self.config = config
        self.initial_obs = obs
        self.initial_info = info

        # Core bookkeeping
        self._step_count = 0
        self._total_steps = 0
        self._is_finished = False
        self._has_failed = False
        self._failure_reason: str | None = None

        # Trajectory tracking
        self._trajectory_points: list[NDArray[np.floating]] = []
        self._reference_trajectory_points: list[NDArray[np.floating]] = []

        # Performance tracking
        self._last_position: NDArray[np.floating] | None = None
        self._total_distance = 0.0

        # Simulation frequency
        self._freq = self._get_env_frequency(config)

        # Hook for subclasses
        self._post_init()

    def _post_init(self):
        """Hook for derived classes to perform additional initialization."""
        pass

    @abstractmethod
    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute the next desired command."""
        ...
    

    def step_callback(
    self,
    action: NDArray[np.floating],
    obs: dict[str, NDArray[np.floating]],
    reward: float,
    terminated: bool,
    truncated: bool,
    info: dict,
) -> bool:
        self._step_count += 1
        self._total_steps += 1
        if "pos" in obs:
            pos = obs["pos"]
        if self._last_position is not None:
            self._total_distance += np.linalg.norm(pos - self._last_position)
        self._last_position = pos.copy()
        self._trajectory_points.append(pos.copy())
        if len(self._trajectory_points) > 1000:
            self._trajectory_points.pop(0)
        
        if terminated and not self._is_finished:
            self._has_failed = True
            self._failure_reason = info.get("failure_reason", "terminated")
        
        if "target_gate" in obs:
            print(f"DEBUG step {self._step_count}: target_gate = {obs['target_gate']}")
            
        if obs.get("target_gate", None) == -1:
            print(f"DEBUG: Last gate passed at step {self._step_count}, time {self.get_current_time():.2f}s")
            self._is_finished = True

            return True
        
        if self._step_callback_impl(action, obs, reward, terminated, truncated, info):
            self._is_finished = True
            
        return self._is_finished or self._has_failed


    def _step_callback_impl(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Override in subclasses for per‐step logic. Return True to force finish."""
        return False
    

    

    def reset(self):
        """Reset internal state."""
        self._step_count = 0
        self._is_finished = False
        self._has_failed = False
        self._failure_reason = None
        self._last_position = None
        self._total_distance = 0.0
        self._trajectory_points.clear()
        # Subclass reset
        self._reset_impl()

    def _reset_impl(self):
        """Override in subclasses for reset logic."""
        pass

    def episode_callback(self):
        """Called after each episode."""
        # Subclass hook
        self._episode_callback_impl()

    def _episode_callback_impl(self):
        """Override in subclasses for episode-end logic."""
        pass

    def episode_reset(self):
        """Reset at episode end."""
        self.reset()
        self._episode_reset_impl()

    def _episode_reset_impl(self):
        """Override in subclasses."""
        pass

    # Utility methods
    def get_current_time(self) -> float:
        return self._step_count / self._freq if self._freq > 0 else self._step_count * 0.02

    def is_finished(self) -> bool:
        return self._is_finished and not self._has_failed

    def has_failed(self) -> bool:
        return self._has_failed

    def get_failure_reason(self) -> str | None:
        return self._failure_reason

    def get_stats(self) -> dict:
        return {
            'step_count': self._step_count,
            'total_steps': self._total_steps,
            'distance': self._total_distance,
            'is_finished': self._is_finished,
            'has_failed': self._has_failed,
            'failure_reason': self._failure_reason,
            'current_time': self.get_current_time(),
        }

    def _get_env_frequency(self, config: dict) -> float:
        return getattr(getattr(config, 'env', None), 'freq', 50.0)
