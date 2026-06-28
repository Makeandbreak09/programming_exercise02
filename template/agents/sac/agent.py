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
        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, act_dim)
        self.fc_log_std = nn.Linear(256, act_dim)

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
        obs = obs.float()
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_log_std(x)

        # From SpinUp / Denis Yarats' SAC implementation:
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (torch.tanh(log_std) + 1)  

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
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)

        # Reparameterization trick (rsample)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)

        action = y_t * self.action_scale + self.action_bias

        # Log probability correction for tanh squashing and action scaling
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1.0 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        mean_action = torch.tanh(mean) * self.action_scale + self.action_bias

        return action, log_prob, mean_action


class SoftQNetwork(nn.Module):
    def __init__(self, observation_space_spec: TensorSpec, action_space_spec: TensorSpec):
        super().__init__()
        obs_dim = observation_space_spec.shape[-1]
        act_dim = int(np.prod(action_space_spec.shape))
        
        # ------------------------------------------------------------ #
        # - TODO: Implement Soft Q-Network Architecture
        # ------------------------------------------------------------ #
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        # ------------------------------------------------------------ #
        # - TODO: Implement forward pass of Soft Q-Network
        # ------------------------------------------------------------ #
        obs = obs.float()
        action = action.float()
        x = torch.cat([obs, action], dim=-1)
        return self.net(x)


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
        self.actor = Actor(self.observation_space_spec, self.action_space_spec)
        self.qf1 = SoftQNetwork(self.observation_space_spec, self.action_space_spec)
        self.qf2 = SoftQNetwork(self.observation_space_spec, self.action_space_spec)
        self.qf1_target = deepcopy(self.qf1)
        self.qf2_target = deepcopy(self.qf2)

        for p in self.qf1_target.parameters():
            p.requires_grad_(False)
        for p in self.qf2_target.parameters():
            p.requires_grad_(False)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=float(cfg.optim.actor_lr))
        self.q_optimizer = torch.optim.Adam(
            list(self.qf1.parameters()) + list(self.qf2.parameters()),
            lr=float(cfg.optim.critic_lr)
        )

        if cfg.alpha.autotune:
            # TODO initialize log_alpha as a learnable parameter and set up the optimizer
            self.log_alpha = nn.Parameter(torch.tensor(np.log(cfg.alpha.value), dtype=torch.float32))
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=float(cfg.alpha.lr))
        else:
            self.alpha = float(cfg.alpha.value)
            self.log_alpha = None
            self.alpha_optimizer = None

        self.target_entropy = -float(act_dim)

    def forward(self, tensordict: TensorDictBase):
        mode = exploration_type()
        obs = tensordict["observation"]

        # ------------------------------------------------------------ #
        # - TODO: Implement action selection based on Gaussian policy
        # ------------------------------------------------------------ #
        action, log_prob, mean_action = self.actor.get_action(obs)
        if mode == ExplorationType.DETERMINISTIC:
            tensordict["action"] = mean_action
        else:
            tensordict["action"] = action

        tensordict["log_prob"] = log_prob

        return tensordict
    
    def update(self, sac_replay_buffer: ReplayBuffer, steps: int) -> dict[str, float]:
        metrics = {}

        data = sac_replay_buffer.sample()

        obs = data["observation"]
        actions = data["action"]
        rewards = data["next"]["reward"]
        next_obs = data["next"]["observation"]
        dones = data["next"]["done"].float()

        # Enforce batch layout
        rewards = rewards.view(-1, 1)
        dones = dones.view(-1, 1)

        # ------------------------------------------------------------ #
        # - TODO: Implement SAC critic update 
        # - TODO: Implement SAC actor update
        # - TODO: Implement soft update of target networks
        # - TODO: Implement monitoring of critic and actor losses
        # ------------------------------------------------------------ #
        if self.cfg.alpha.autotune:
            alpha = self.log_alpha.exp().detach()
        else:
            alpha = torch.tensor(self.alpha, device=obs.device)

        # Critic update
        with torch.no_grad():
            next_state_actions, next_state_log_pi, _ = self.actor.get_action(next_obs)
            qf1_next_target = self.qf1_target(next_obs, next_state_actions)
            qf2_next_target = self.qf2_target(next_obs, next_state_actions)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
            next_q_value = rewards + (1.0 - dones) * self.cfg.loss.gamma * min_qf_next_target

        qf1_a_values = self.qf1(obs, actions)
        qf2_a_values = self.qf2(obs, actions)
        qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
        qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
        qf_loss = qf1_loss + qf2_loss

        self.q_optimizer.zero_grad()
        qf_loss.backward()
        self.q_optimizer.step()

        # Actor update
        pi, log_pi, _ = self.actor.get_action(obs)
        qf1_pi = self.qf1(obs, pi)
        qf2_pi = self.qf2(obs, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update of target networks
        with torch.no_grad():
            for p_online, p_target in zip(self.qf1.parameters(), self.qf1_target.parameters()):
                p_target.data.mul_(1.0 - self.cfg.optim.tau).add_(self.cfg.optim.tau * p_online.data)
            for p_online, p_target in zip(self.qf2.parameters(), self.qf2_target.parameters()):
                p_target.data.mul_(1.0 - self.cfg.optim.tau).add_(self.cfg.optim.tau * p_online.data)

        metrics["critic_loss"] = qf_loss.item()
        metrics["actor_loss"] = actor_loss.item()
        metrics["qf1_values"] = qf1_a_values.mean().item()
        metrics["qf2_values"] = qf2_a_values.mean().item()

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
        with torch.no_grad():
            _, log_pi, _ = self.actor.get_action(obs)

        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        metrics["alpha_loss"] = alpha_loss.item()
        metrics["alpha"] = self.log_alpha.exp().item()
    