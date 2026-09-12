# Incomplete components

This directory holds code from the hackathon prototype that **cannot run** and is
kept for reference only. It is separated from `src/` so that the PPO
implementation and the reward functions can be inspected without ambiguity about
what works and what does not.

| File | Status | What is missing |
| --- | --- | --- |
| `environment_step.py` | Fragment | The environment class that owned this `step()` method, its `__init__`, the helpers `_apply_action`, `_update_energy_flows`, `_get_state`, `is_peak_hour`, and the `compute_reward(...)` variant it calls. |

Nothing here has been reconstructed or guessed. The original simulation
environment (demand profile, solar generation, price signal, battery model) was
not saved after the hackathon, so it is not part of this repository.
