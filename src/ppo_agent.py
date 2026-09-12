"""Proximal Policy Optimization (PPO) agent for discrete energy-storage actions.

The implementation follows the clipped-surrogate variant of PPO described in
Schulman et al., "Proximal Policy Optimization Algorithms" (2017).

Two objects are defined here:

* :class:`ActorCritic` -- a shared-trunk network with a policy head (actor) and
  a state-value head (critic).
* :class:`PPOAgent`    -- rollout post-processing (returns, advantages) and the
  PPO update loop.

The agent is environment-agnostic: it consumes a dictionary of collected
transitions, so any environment that produces flat float states and discrete
action indices can be plugged in.
"""

from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


# ---------------------------------------------------------------------------
# Actor-Critic network
# ---------------------------------------------------------------------------
class ActorCritic(nn.Module):
    """Shared-trunk actor-critic network for a discrete action space.

    The trunk encodes the energy state (for example state of charge, demand,
    renewable generation and price). The actor head outputs unnormalized action
    logits; the critic head outputs a scalar state-value estimate.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(action_logits, state_value)`` for the given state batch."""
        features = self.shared(state)
        logits = self.actor(features)
        value = self.critic(features)
        return logits, value

    def get_action(
        self, state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample an action from the current policy.

        Used during rollout collection. Returns the sampled action, its
        log-probability under the *current* (soon to be "old") policy, the
        policy entropy and the critic's value estimate.
        """
        logits, value = self.forward(state)
        distribution = Categorical(logits=logits)
        action = distribution.sample()
        log_prob = distribution.log_prob(action)
        entropy = distribution.entropy()
        return action, log_prob, entropy, value

    def evaluate_actions(
        self, states: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Re-score stored actions under the current policy parameters.

        This is the core of the PPO update: the same transitions are evaluated
        repeatedly as the policy changes, which yields the probability ratio
        between the new and the old policy.
        """
        logits, values = self.forward(states)
        distribution = Categorical(logits=logits)
        log_probs = distribution.log_prob(actions)
        entropy = distribution.entropy()
        return log_probs, entropy, values.squeeze(-1)


# ---------------------------------------------------------------------------
# PPO agent
# ---------------------------------------------------------------------------
class PPOAgent:
    """PPO agent with a clipped policy objective, value loss and entropy bonus.

    Args:
        state_dim: Dimensionality of the flat observation vector.
        action_dim: Number of discrete actions.
        lr: Adam learning rate shared by actor and critic.
        gamma: Discount factor used when accumulating returns.
        clip_eps: Clipping range epsilon of the PPO surrogate objective.
        value_coef: Weight of the value (critic) loss in the total loss.
        entropy_coef: Weight of the entropy bonus, which encourages exploration.
        update_epochs: Number of passes over each batch of collected rollouts.
        batch_size: Mini-batch size used inside each epoch.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        clip_eps: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        update_epochs: int = 10,
        batch_size: int = 64,
    ) -> None:
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.update_epochs = update_epochs
        self.batch_size = batch_size

        self.model = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def compute_returns(
        self,
        rewards: Sequence[float],
        dones: Sequence[bool],
        last_value: float = 0.0,
    ) -> torch.Tensor:
        """Accumulate discounted returns backwards through a rollout.

        ``last_value`` bootstraps the return of a truncated (not terminated)
        rollout. The accumulator is reset at episode boundaries so returns never
        leak across episodes.
        """
        returns: List[float] = []
        running_return = last_value

        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                running_return = 0.0
            running_return = reward + self.gamma * running_return
            returns.insert(0, running_return)

        return torch.tensor(returns, dtype=torch.float32)

    def update(self, memory: Dict[str, Any]) -> None:
        """Run the PPO update on one batch of collected transitions.

        Args:
            memory: Rollout buffer with the keys ``states``, ``actions``,
                ``log_probs`` (under the policy that generated the data),
                ``rewards`` and ``dones``.
        """
        states = torch.tensor(memory["states"], dtype=torch.float32)
        actions = torch.tensor(memory["actions"], dtype=torch.int64)
        old_log_probs = torch.tensor(memory["log_probs"], dtype=torch.float32)
        rewards = memory["rewards"]
        dones = memory["dones"]

        returns = self.compute_returns(rewards, dones)

        # Advantage estimate: return minus baseline, normalized for stable
        # gradients. Computed once, outside the epoch loop, from the old policy.
        with torch.no_grad():
            _, _, values = self.model.evaluate_actions(states, actions)
            advantages = returns - values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        num_samples = states.size(0)

        for _ in range(self.update_epochs):
            indices = torch.randperm(num_samples)

            for start in range(0, num_samples, self.batch_size):
                batch_idx = indices[start : start + self.batch_size]

                batch_states = states[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_returns = returns[batch_idx]
                batch_advantages = advantages[batch_idx]

                new_log_probs, entropy, values = self.model.evaluate_actions(
                    batch_states, batch_actions
                )

                # Probability ratio between the new and the old policy.
                ratios = torch.exp(new_log_probs - batch_old_log_probs)

                # Clipped surrogate objective: the pessimistic (minimum) of the
                # unclipped and clipped terms bounds how far the policy can move
                # in a single update.
                unclipped = ratios * batch_advantages
                clipped = (
                    torch.clamp(ratios, 1.0 - self.clip_eps, 1.0 + self.clip_eps)
                    * batch_advantages
                )

                policy_loss = -torch.min(unclipped, clipped).mean()
                value_loss = ((batch_returns - values) ** 2).mean()
                entropy_bonus = entropy.mean()

                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    - self.entropy_coef * entropy_bonus
                )

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
