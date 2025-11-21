from __future__ import annotations

from typing import Sequence


def _validate_common_inputs(
    *,
    rewards: Sequence[float],
    dones: Sequence[bool],
    next_values: Sequence[float],
    gamma: float,
) -> int:
    if len(rewards) == 0:
        raise ValueError("rewards must not be empty.")
    if len(rewards) != len(dones) or len(rewards) != len(next_values):
        raise ValueError("rewards, dones, and next_values must have the same length.")
    if gamma < 0.0 or gamma > 1.0:
        raise ValueError("gamma must be in [0, 1].")
    return len(rewards)


def n_step_bootstrap_targets(
    *,
    rewards: Sequence[float],
    dones: Sequence[bool],
    next_values: Sequence[float],
    gamma: float,
    n_steps: int,
) -> list[float]:
    horizon = _validate_common_inputs(
        rewards=rewards,
        dones=dones,
        next_values=next_values,
        gamma=gamma,
    )
    if n_steps <= 0:
        raise ValueError("n_steps must be > 0.")

    targets: list[float] = [0.0] * horizon
    for start in range(horizon):
        target = 0.0
        discount = 1.0
        terminated = False
        used_steps = 0

        for offset in range(n_steps):
            index = start + offset
            if index >= horizon:
                break
            target += discount * float(rewards[index])
            used_steps += 1
            if bool(dones[index]):
                terminated = True
                break
            discount *= gamma

        if not terminated and used_steps > 0:
            bootstrap_transition = min(horizon - 1, start + used_steps - 1)
            target += discount * float(next_values[bootstrap_transition])

        targets[start] = target

    return targets


def td_lambda_targets(
    *,
    rewards: Sequence[float],
    dones: Sequence[bool],
    next_values: Sequence[float],
    gamma: float,
    lambda_: float,
) -> list[float]:
    horizon = _validate_common_inputs(
        rewards=rewards,
        dones=dones,
        next_values=next_values,
        gamma=gamma,
    )
    if lambda_ < 0.0 or lambda_ > 1.0:
        raise ValueError("lambda_ must be in [0, 1].")

    targets: list[float] = [0.0] * horizon
    for index in range(horizon - 1, -1, -1):
        reward = float(rewards[index])
        done = bool(dones[index])
        if done:
            targets[index] = reward
            continue

        if index + 1 < horizon:
            lambda_future = ((1.0 - lambda_) * float(next_values[index])) + (
                lambda_ * float(targets[index + 1])
            )
        else:
            lambda_future = float(next_values[index])
        targets[index] = reward + (gamma * lambda_future)

    return targets
