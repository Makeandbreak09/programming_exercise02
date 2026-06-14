from typing import Callable

import numpy as np
from torchrl.envs.utils import ExplorationType, set_exploration_type


def evaluate(agent, createfn: Callable, nepisodes: int = 10):

    total_ep_rew = []

    with set_exploration_type(ExplorationType.DETERMINISTIC):

        for idx in range(nepisodes):
            data = createfn().rollout(
                10_000_000,
                policy=agent,
                break_when_any_done=True,
            )

            total_ep_rew.append(data["next"]["reward"].sum().item())

    return {
        "avg_eval_ep_rew": np.mean(total_ep_rew),
        "std_eval_ep_rew": np.std(total_ep_rew),
    }
