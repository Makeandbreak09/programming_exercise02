from copy import deepcopy

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from omegaconf import DictConfig
from tensordict import TensorDictBase
from torchrl.envs.utils import exploration_type, ExplorationType
from torchrl.envs.common import TensorSpec
from torchrl.data import ReplayBuffer
from torchrl.data.tensor_specs import ContinuousBox


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, observation_space_spec: TensorSpec, action_space_spec: TensorSpec):
        super().__init__()

        obs_dim = observation_space_spec.shape[-1]
        act_dim = int(np.prod(action_space_spec.shape))

        # ------------------------------------------------------------ #
        # - TODO: Implement Actor-Network Architecture outputting 
        #         mean and log_std of Gaussian Policy
        # ------------------------------------------------------------ #

        # Action Rescaling
        assert isinstance(action_space_spec.space, ContinuousBox), \
            "Only continuous action spaces are supported for SAC."
        self.register_buffer(
            "action_scale",
            (action_space_spec.space.high - action_space_spec.space.low) / 2.0
        )
        self.register_buffer(
            "action_bias",
            (action_space_spec.space.high + action_space_spec.space.low) / 2.0
        )

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # ------------------------------------------------------------ #
        # - TODO: Implement Actor Forward Pass to output mean 
        #         and log_std of Gaussian Policy. For log_std, use
        #         the normalization trick that is already implemented
        # ------------------------------------------------------------ #

        raise NotImplementedError

        # From SpinUp / Denis Yarats' SAC implementation:
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  

        return mean, log_std

    def get_action(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert isinstance(self.action_scale, torch.Tensor) 
        assert isinstance(self.action_bias, torch.Tensor)

        # ------------------------------------------------------------ #
        # - TODO: Implement Action Sampling from Gaussian Policy with 
        #         Reparameterization Trick, outputting sampled action, 
        #         log-probability of the action and the mean action 
        #         (for evaluation)
        # ------------------------------------------------------------ #

        raise NotImplementedError

        return action, log_prob, mean


class SoftQNetwork(nn.Module):
    def __init__(self, observation_space_spec: TensorSpec, action_space_spec: TensorSpec):
        super().__init__()
        obs_dim = observation_space_spec.shape[-1]
        act_dim = int(np.prod(action_space_spec.shape))
        
        # ------------------------------------------------------------ #
        # - TODO: Implement Soft Q-Network Architecture
        # ------------------------------------------------------------ #

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        # ------------------------------------------------------------ #
        # - TODO: Implement forward pass of Soft Q-Network
        # ------------------------------------------------------------ #

        raise NotImplementedError


class SoftActorCritic(nn.Module):
    def __init__(
            self,
            cfg: DictConfig,
            action_space_spec: TensorSpec,
            observation_space_spec: TensorSpec
        ):
        super().__init__()

        self.cfg = cfg
        self.action_space_spec = action_space_spec
        self.observation_space_spec = observation_space_spec["observation"] # type: ignore
        act_dim = int(np.prod(action_space_spec.shape))

        # ------------------------------------------------------------ #
        # - TODO: Setup Actor and Crtic Networks
        # - TODO: Setup Optimizers for Actor and Critic Networks
        # - TODO: (Optional) Setup Learnable Temperature Parameter for 
        #         Automatic Entropy Tuning
        # ------------------------------------------------------------ #

        if cfg.alpha.autotune:
            # TODO initialize log_alpha as a learnable parameter and set up the optimizer
            pass

    def forward(self, tensordict: TensorDictBase):
        mode = exploration_type()
        obs = tensordict["observation"]

        # ------------------------------------------------------------ #
        # - TODO: Implement action selection based on Gaussian policy
        # ------------------------------------------------------------ #

        return tensordict
    
    def update(self, sac_replay_buffer: ReplayBuffer, steps: int) -> dict[str, float]:
        metrics = {}

        data = sac_replay_buffer.sample()

        obs = data["observation"]
        actions = data["action"]
        rewards = data["next"]["reward"]
        next_obs = data["next"]["observation"]
        dones = data["next"]["done"].float()

        # ------------------------------------------------------------ #
        # - TODO: Implement SAC critic update 
        # - TODO: Implement SAC actor update
        # - TODO: Implement soft update of target networks
        # - TODO: Implement monitoring of critic and actor losses
        # ------------------------------------------------------------ #

        self.maybe_update_alpha(obs, metrics)

        return metrics
    
    def maybe_update_alpha(
            self,
            obs: torch.Tensor,
            metrics: dict[str, float]
        ) -> None:
        if self.alpha_optimizer is None:
            return
        
        # ------------------------------------------------------------ #
        # - TODO: Implement automatic entropy tuning 
        # - TODO: Log alpha loss and current value of alpha in metrics 
        # ------------------------------------------------------------ #
    