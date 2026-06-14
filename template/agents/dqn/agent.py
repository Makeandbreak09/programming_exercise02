from copy import deepcopy

import torch
import numpy as np
import torch.nn.functional as F
from omegaconf import DictConfig
from tensordict import TensorDictBase
from torchrl.envs.utils import exploration_type, ExplorationType


class DeepQNetwork(torch.nn.Module):
    def __init__(self, cfg: DictConfig, action_space_spec, observation_space_spec):
        super().__init__()

        self.cfg = cfg
        self.action_space_spec = action_space_spec
        self.observation_space_spec = observation_space_spec

        self.net_output_dim = self.action_space_spec.space.n
        self.net_input_dim = self.observation_space_spec["observation"].shape[0]

        # ------------------------------------------------------------ #
        # - TODO: Create Q-network and target network
        # - TODO: Create optimizer for neural networks
        # ------------------------------------------------------------ #

        self._eps = self.cfg.exploration.eps_start_value

    def forward(self, tensordict: TensorDictBase):

        mode = exploration_type()
        obs = tensordict["observation"]

        # ------------------------------------------------------------ #
        # - TODO: Implement action selection based on learned Q Values
        # - TODO: Implement epsilon-greedy action selection for agent
        # ------------------------------------------------------------ #

        tensordict["action"] = self.action_space_spec.sample()

        return tensordict

    def update(self, dqn_replay_buffer, steps: int):
        metrics = {}

        data = dqn_replay_buffer.sample()

        obs = data["observation"]
        actions = data["action"]
        rewards = data["next"]["reward"]
        nxtobs = data["next"]["observation"]
        dones = data["next"]["done"].float()

        # ------------------------------------------------------------ #
        # - TODO: Implement DQN temporal-difference update
        # - TODO: Implement update of target networks
        # - TODO: Implement monitoring of loss values and average q values
        # ------------------------------------------------------------ #

        metrics["epsilon"] = self._eps
        metrics["avg_dqn_loss"] = 0.0
        metrics["avg_q_values"] = 0.0

        return metrics
