from typing import List

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from tensordict import TensorDictBase
from torchrl.envs.utils import exploration_type, ExplorationType


class REINFORCE(torch.nn.Module):
    def __init__(self, cfg: DictConfig, action_space_spec, observation_space_spec):
        super().__init__()

        self.cfg = cfg
        self.action_space_spec = action_space_spec
        self.observation_space_spec = observation_space_spec

        self.net_output_dim = self.action_space_spec.space.n
        self.net_input_dim = self.observation_space_spec["observation"].shape[0]

        # ------------------------------------------------------------ #
        # - TODO: Create policy network and optimizer
        # ------------------------------------------------------------ #

    def forward(self, tensordict: TensorDictBase):

        mode = exploration_type()
        obs = tensordict["observation"]

        # ------------------------------------------------------------ #
        # - TODO: Implement action selection based on learned Q Values
        # - TODO: Implement epsilon-greedy action selection for agent
        # ------------------------------------------------------------ #

        tensordict["action"] = self.action_space_spec.sample()

        return tensordict

    def update(self, episodes: List, steps: int):
        metrics, losses = {}, []

        for niter in range(self.cfg.loss.num_update_iters):
            loss = 0.0

            for episode in episodes:
                obs = episode["observation"]
                actions = episode["action"]
                rewards = episode["next"]["reward"]
                nxtobs = episode["next"]["observation"]
                dones = episode["next"]["done"].float()

                # ------------------------------------------------------------#
                # ---------- Compute Log. Prob. of Selected Action ---------- #
                # ------------------------------------------------------------#

                # ...


                # ----------------------------------------------------------#
                # ---------- Compute Importance Sampling Weights ---------- #
                # ----------------------------------------------------------#

                # ...


                # -----------------------------------------------------------#
                # ---------- Compute Monte-Carlo Return Estimates ---------- #
                # -----------------------------------------------------------#

                # ...

                # ----------------------------------------------------------#
                # ---------- Compute and Subtract Baseline Value ---------- #
                # ----------------------------------------------------------#

                # ...

                # ---------------------------------------------------#
                # ---------- Compute Policy Gradient Loss ---------- #
                # ---------------------------------------------------#

                # ...

            # ------------------------------------------------------#
            # ---------- Accumulate Loss Across Episodes ---------- #
            # ------------------------------------------------------#

            # ...

        return metrics
