# Demo

There is no runnable demo script in this repository, because the simulation
environment that the agent was trained against was not preserved after the
hackathon. See [`incomplete/README.md`](../incomplete/README.md).

What exists instead:

- [`results/`](../results/) — the reward-function comparison and the training and
  evaluation output captured during the hackathon.
- [`tests/test_smoke.py`](../tests/test_smoke.py) — runs one real PPO update on
  synthetic rollout data, which is the closest thing to a working demo of the
  agent right now.

Screen recordings of the original working prototype exist locally but are ~77 MB
in total, so they are deliberately not committed (they are excluded by
`.gitignore`). If a demo video is ever needed, it should be uploaded to an
external host and linked from here rather than added to the repository.
