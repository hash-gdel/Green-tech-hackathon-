"""Baseline and improved reward functions for the energy-storage agent.

This module is the main contribution of the hackathon project: the PPO
algorithm itself is kept unchanged, and the multi-objective reward signal is
what was redesigned.

Provenance
----------
The hackathon prototype was not saved as a Python file. Both functions below
were recovered verbatim from the project artifact
``results/reward-function-baseline-vs-improved.png`` (the screenshot shown
during the presentation), so the coefficients and the term structure are the
original ones. Nothing has been added, removed or re-tuned.

Note that ``compute_reward_improved`` accepts ``prev_soc`` but does not use it;
that is how the original function was written. The signature also differs from
the ``compute_reward(...)`` call in ``incomplete/environment_step.py``, which
belongs to a different iteration of the prototype. See ``docs/reward-design.md``.
"""


def compute_reward_original(grid_energy: float, price: float) -> float:
    """Baseline reward: minimize the cost of energy imported from the grid.

    A single objective. It says nothing about renewable utilization, battery
    wear, unserved demand or peak hours, which is the limitation the improved
    reward addresses.
    """
    return -(grid_energy * price)


def compute_reward_improved(
    grid_energy: float,
    price: float,
    soc: float,
    prev_soc: float,
    unserved_demand: float,
    curtailed: float,
    solar_used: float,
    battery_out: float,
    battery_charge: float,
    grid_to_battery: float,
    demand: float,
    is_peak: bool,
) -> float:
    """Multi-objective reward combining cost, renewables, battery and peak use.

    Args:
        grid_energy: Energy imported from the grid this step.
        price: Electricity price for this step.
        soc: Battery state of charge after the action, in ``[0, 1]``.
        prev_soc: State of charge before the action (accepted, unused).
        unserved_demand: Demand that could not be met this step.
        curtailed: Renewable generation that was wasted.
        solar_used: Renewable generation consumed directly.
        battery_out: Energy discharged from the battery to the load.
        battery_charge: Energy charged into the battery.
        grid_to_battery: Portion of the charge that came from the grid.
        demand: Total load demand this step.
        is_peak: Whether the step falls inside a peak-demand hour.

    Returns:
        The scalar reward for the step.
    """
    local_supply = solar_used + battery_out
    local_supply_ratio = local_supply / (demand + 1e-8)
    battery_throughput = abs(battery_charge) + abs(battery_out)

    reward = 0.0
    reward += 3.5 * local_supply_ratio  # reward serving load locally
    reward -= 1.5 * grid_energy * price  # electricity cost
    reward -= 0.4 * curtailed  # wasted renewable generation
    reward -= 0.05 * battery_throughput  # battery cycling / wear
    reward -= 0.7 * grid_to_battery * price  # discourage charging from the grid
    reward -= 3.0 * unserved_demand  # unmet demand is the heaviest penalty

    # Keep the battery inside a safe operating band.
    if soc < 0.15 or soc > 0.95:
        reward -= 0.5

    # Extra penalty for importing during peak hours.
    if is_peak:
        reward -= 0.5 * grid_energy * price

    return reward
