# Reward design

Reference for `src/reward_functions.py`. This is where the project's contribution
lives: the PPO algorithm is standard, and the reward signal is what was changed.

## Baseline

```python
def compute_reward_original(grid_energy, price):
    return -(grid_energy * price)
```

One objective: minimize the cost of imported grid energy. An agent trained on
this signal has no reason to care whether demand is actually met, whether
renewable generation is wasted, how hard the battery is cycled, or when during
the day the energy is drawn. Cost can be reduced simply by not serving load.

## Improved reward

The improved function keeps cost as one term among several. Each term maps to an
operational objective:

| Term | Weight | Objective |
| --- | --- | --- |
| `+ local_supply_ratio` | 3.5 | Serve demand from solar and battery rather than the grid (renewable utilization) |
| `- grid_energy * price` | 1.5 | Electricity cost |
| `- curtailed` | 0.4 | Avoid wasting available renewable generation |
| `- battery_throughput` | 0.05 | Limit charge/discharge cycling (battery wear) |
| `- grid_to_battery * price` | 0.7 | Do not fill the battery from the grid |
| `- unserved_demand` | 3.0 | Reliability — the heaviest penalty |
| `soc < 0.15 or soc > 0.95` | 0.5 (flat) | Keep the battery inside a safe operating band |
| `is_peak` extra cost | 0.5 | Discourage grid import during peak hours (peak shaving) |

where

```
local_supply       = solar_used + battery_out
local_supply_ratio = local_supply / (demand + 1e-8)
battery_throughput = abs(battery_charge) + abs(battery_out)
```

The relative magnitudes encode a priority ordering: serving demand (3.0–3.5)
outweighs cost (1.5), which outweighs curtailment (0.4) and battery wear (0.05).
The `1e-8` in the denominator guards against division by zero at zero demand.

The weights are the ones used during the hackathon. They were hand-tuned under
time pressure, not swept or optimized — treating them as a hyperparameter search
space is listed as future work in the README.

## Provenance and known inconsistency

Both functions were recovered verbatim from
[`results/reward-function-baseline-vs-improved.png`](../results/reward-function-baseline-vs-improved.png),
the screenshot presented at the hackathon; the prototype was never saved as a
`.py` file. Two honest caveats:

1. `compute_reward_improved` accepts `prev_soc` but never uses it. Presumably it
   was intended for a state-of-charge *change* penalty (`abs(soc - prev_soc)`),
   which the `battery_throughput` term ends up covering instead. The unused
   parameter is preserved rather than removed, to keep the function faithful to
   the original.
2. The `compute_reward(...)` call in
   [`incomplete/environment_step.py`](../incomplete/environment_step.py) has a
   different signature — it passes `stored_from_renewables`,
   `discharged_to_load`, `thermal_avoided` and `battery_power`, which neither
   function accepts. The two files therefore come from different iterations of
   the design. The missing variant has **not** been reconstructed.

## Reported effect

The evaluation output captured in
[`results/training-and-evaluation-output.png`](../results/training-and-evaluation-output.png)
compares the two reward formulations over a 100-episode run. Those numbers are
reproduced in the README; they come from the hackathon environment, which is not
part of this repository, so they cannot currently be re-run from this code.
