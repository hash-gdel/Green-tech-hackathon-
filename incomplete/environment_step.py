"""INCOMPLETE: the environment step function from the hackathon prototype.

This file is preserved for reference only and is deliberately kept outside
``src/`` because it cannot run. It is the ``step()`` method of the energy
environment class from the hackathon prototype; the class itself, its
``__init__`` and its helper methods were not saved.

It is reproduced unchanged (only reformatted) rather than reconstructed, so the
structure of the original environment loop stays visible: apply action, update
energy flows, compute the reward from the updated quantities, advance time,
build the next observation.

Missing dependencies, none of which exist anywhere in this repository:

* ``compute_reward(...)`` -- the reward function this iteration called. Its
  signature does not match either function in ``src/reward_functions.py``
  (it expects ``stored_from_renewables``, ``discharged_to_load`` and
  ``thermal_avoided``), so the two are different iterations of the design.
* ``self._apply_action(action)``
* ``self._update_energy_flows()``
* ``self._get_state()``
* ``self.is_peak_hour(t)``
* the attributes ``soc``, ``grid_energy``, ``price_signal``, ``battery_power``,
  ``unserved_demand``, ``curtailed_renewable``, ``stored_from_renewables``,
  ``discharged_to_load``, ``thermal_avoided``, ``t`` and ``max_steps``.

Note also that ``step`` is defined at module level while taking ``self`` as its
first argument: it was extracted from its class and never put back.

Do not import this module expecting it to work. Rebuilding a runnable
environment is tracked as future work in the README.
"""

from typing import Any, Dict, Tuple


def step(self, action: Any) -> Tuple[Any, float, bool, Dict[str, Any]]:  # noqa: N805
    """Advance the environment by one timestep. NOT RUNNABLE -- see module docstring."""
    # 1. Save previous state info
    self.prev_soc = self.soc

    # 2. Apply action to environment
    self._apply_action(action)

    # 3. Update system quantities after action
    self._update_energy_flows()

    # 4. Compute reward using updated values
    reward = compute_reward(  # noqa: F821  -- missing integration point
        grid_energy=self.grid_energy,
        price=self.price_signal[self.t],
        battery_power=self.battery_power,
        soc=self.soc,
        prev_soc=self.prev_soc,
        unserved_demand=self.unserved_demand,
        curtailed_renewable=self.curtailed_renewable,
        stored_from_renewables=self.stored_from_renewables,
        discharged_to_load=self.discharged_to_load,
        thermal_avoided=self.thermal_avoided,
        is_peak_hour=self.is_peak_hour(self.t),
    )

    # 5. Advance time
    self.t += 1

    # 6. Build next state
    next_state = self._get_state()
    done = self.t >= self.max_steps
    info: Dict[str, Any] = {}

    return next_state, reward, done, info
