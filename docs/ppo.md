# PPO implementation notes

Reference for `src/ppo_agent.py`. The implementation follows the clipped-surrogate
variant of PPO from Schulman et al., *Proximal Policy Optimization Algorithms*
(2017), which is the paper cited in the hackathon presentation.

## Network

`ActorCritic` is a shared-trunk network:

```
state -> Linear(state_dim, 64) -> Tanh -> Linear(64, 64) -> Tanh
                                                    |-> Linear(64, action_dim)  (actor logits)
                                                    |-> Linear(64, 1)           (critic value)
```

The action space is **discrete**: the actor emits logits for a `Categorical`
distribution, so actions are indices (for example charge / hold / discharge
levels) rather than continuous power set-points.

Two methods separate the two phases of PPO:

- `get_action(state)` samples during rollout collection and returns the action,
  its log-probability under the behavior policy, the entropy and the value
  estimate. The log-probability must be stored, because it becomes
  `log π_old` in the update.
- `evaluate_actions(states, actions)` re-scores already-collected actions under
  the current parameters. Calling it repeatedly as the parameters change is what
  produces the probability ratio.

## Returns and advantages

`compute_returns` walks the rollout backwards and accumulates the discounted
return, resetting the accumulator at episode boundaries so returns do not leak
across episodes. `last_value` bootstraps a rollout that was truncated rather
than terminated.

Advantages are the simple difference `A_t = G_t - V(s_t)`, then normalized to
zero mean and unit variance. They are computed once per batch under
`torch.no_grad()`, before the epoch loop, so all epochs use the advantages of
the old policy.

This is **not** GAE(λ); adding generalized advantage estimation is listed as
future work.

## Update

For `update_epochs` passes over the batch, in mini-batches of `batch_size`:

```
ratio      = exp(log π_new(a|s) - log π_old(a|s))
L_clip     = min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)
policy_loss  = -mean(L_clip)
value_loss   =  mean((G - V(s))^2)
entropy_bonus = mean(entropy)

loss = policy_loss + value_coef * value_loss - entropy_coef * entropy_bonus
```

Taking the minimum of the unclipped and clipped terms is the pessimistic bound
that keeps the policy from moving too far in a single update. The entropy term
is *subtracted* from the loss, so maximizing entropy is rewarded, which keeps the
agent exploring charge/discharge options instead of collapsing early onto one
behavior.

Defaults: `lr=3e-4`, `gamma=0.99`, `clip_eps=0.2`, `value_coef=0.5`,
`entropy_coef=0.01`, `update_epochs=10`, `batch_size=64`.

Actor and critic share one Adam optimizer over all parameters, which is why the
two losses are combined with `value_coef` rather than stepped separately.

## Original pseudocode

The pseudocode below was written during the hackathon (it was the file `old_ago`
in the original project folder) and is preserved verbatim, since it documents the
intended training loop that surrounds `PPOAgent.update`:

```
Initialize policy network πθ
Initialize value network Vφ

for iteration = 1 to max_iterations do

    Collect trajectories using current policy πθ:
        for t = 1 to T do
            observe state s_t
            sample action a_t ~ πθ(. | s_t)
            execute action a_t
            receive reward r_t and next state s_{t+1}
            store (s_t, a_t, r_t, s_{t+1}, logπ_old(a_t|s_t))
        end for

    Compute returns G_t
    Compute advantages A_t
        A_t = G_t - Vφ(s_t)
        optionally normalize advantages

    for epoch = 1 to K do
        divide collected data into mini-batches

        for each mini-batch do
            Compute new log probabilities logπθ(a_t|s_t)
            Compute ratio:
                r_t(θ) = exp(logπθ(a_t|s_t) - logπ_old(a_t|s_t))

            Compute clipped surrogate objective:
                L_clip = min(r_t * A_t, clip(r_t, 1-ε, 1+ε) * A_t)

            Policy loss:
                L_policy = -mean(L_clip)

            Value loss:
                L_value = mean((Vφ(s_t) - G_t)^2)

            Entropy bonus:
                L_entropy = mean(entropy of πθ(.|s_t))

            Total loss:
                L = L_policy + c1 * L_value - c2 * L_entropy

            Update θ and φ using gradient descent
        end for
    end for

end for
```

`PPOAgent` implements everything from "Compute returns" onwards. The trajectory
collection loop needs an environment, which is the part of the prototype that was
not preserved — see [`incomplete/README.md`](../incomplete/README.md).

## Expected rollout buffer

`update()` takes a plain dictionary:

```python
memory = {
    "states": [...],     # sequence of flat float state vectors
    "actions": [...],    # discrete action indices
    "log_probs": [...],  # log π_old(a|s), from get_action at collection time
    "rewards": [...],    # scalar rewards
    "dones": [...],      # episode-termination flags
}
```
