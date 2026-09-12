"""Smoke tests: the modules import and one PPO update runs end to end.

These check wiring, not learning quality. Run from the repository root:

    python -m pytest tests/          # if pytest is installed
    python tests/test_smoke.py       # plain Python, no pytest needed
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.reward_functions import compute_reward_improved, compute_reward_original


def test_baseline_reward_is_negative_cost() -> None:
    assert compute_reward_original(grid_energy=2.0, price=1.5) == -3.0


def test_improved_reward_prefers_local_supply() -> None:
    """Meeting demand locally should score better than importing the same energy."""
    common = dict(
        soc=0.5,
        prev_soc=0.5,
        unserved_demand=0.0,
        curtailed=0.0,
        battery_charge=0.0,
        grid_to_battery=0.0,
        demand=10.0,
        is_peak=False,
    )
    from_solar = compute_reward_improved(
        grid_energy=0.0, price=1.0, solar_used=10.0, battery_out=0.0, **common
    )
    from_grid = compute_reward_improved(
        grid_energy=10.0, price=1.0, solar_used=0.0, battery_out=0.0, **common
    )
    assert from_solar > from_grid


def test_improved_reward_penalises_peak_hours() -> None:
    common = dict(
        grid_energy=5.0,
        price=1.0,
        soc=0.5,
        prev_soc=0.5,
        unserved_demand=0.0,
        curtailed=0.0,
        solar_used=0.0,
        battery_out=0.0,
        battery_charge=0.0,
        grid_to_battery=0.0,
        demand=5.0,
    )
    assert compute_reward_improved(is_peak=True, **common) < compute_reward_improved(
        is_peak=False, **common
    )


def test_ppo_update_runs() -> None:
    """One full PPO update on synthetic rollout data. Requires PyTorch."""
    try:
        import torch
    except ImportError:  # pragma: no cover - PyTorch is an optional test dep
        print("SKIP test_ppo_update_runs: PyTorch not installed")
        return

    from src.ppo_agent import PPOAgent

    torch.manual_seed(0)
    state_dim, action_dim, num_steps = 6, 3, 128
    agent = PPOAgent(state_dim=state_dim, action_dim=action_dim, update_epochs=2)

    states, actions, log_probs, rewards, dones = [], [], [], [], []
    for step in range(num_steps):
        state = torch.rand(state_dim)
        action, log_prob, _, _ = agent.model.get_action(state)
        states.append(state.tolist())
        actions.append(int(action.item()))
        log_probs.append(float(log_prob.item()))
        rewards.append(float(torch.rand(1).item()))
        dones.append((step + 1) % 32 == 0)

    memory = {
        "states": states,
        "actions": actions,
        "log_probs": log_probs,
        "rewards": rewards,
        "dones": dones,
    }

    returns = agent.compute_returns(rewards, dones)
    assert returns.shape == (num_steps,)

    before = [p.detach().clone() for p in agent.model.parameters()]
    agent.update(memory)
    after = list(agent.model.parameters())
    assert any(not torch.equal(b, a) for b, a in zip(before, after)), (
        "PPO update did not change any parameters"
    )


if __name__ == "__main__":
    for name, test in sorted(globals().items()):
        if name.startswith("test_") and callable(test):
            test()
            print(f"ok  {name}")
    print("all smoke tests passed")
