"""GREENIT energy-management optimization with PPO.

Public entry points:

* :class:`~src.ppo_agent.ActorCritic` and :class:`~src.ppo_agent.PPOAgent`
* :func:`~src.reward_functions.compute_reward_original` and
  :func:`~src.reward_functions.compute_reward_improved`

The reward functions are importable without PyTorch installed; the PPO agent
requires PyTorch.
"""

from .reward_functions import compute_reward_improved, compute_reward_original

__all__ = [
    "compute_reward_original",
    "compute_reward_improved",
]
