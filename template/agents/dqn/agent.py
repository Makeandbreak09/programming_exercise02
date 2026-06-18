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

        # Online Q-Network
        self.net = torch.nn.Sequential(
            torch.nn.Linear(self.net_input_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, self.net_output_dim),
        )

        # Target Q-Network
        self.target_net = deepcopy(self.net)

        # Freeze target so its parameters are never updated by the optimizer
        for p in self.target_net.parameters():
            p.requires_grad_(False)


        # Optimizer
        # cfg.optim:  lr, tau
        # cfg.loss:   gamma
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=float(cfg.optim.lr))
        self.tau   = float(cfg.optim.tau)    # Polyak coefficient τ ≪ 1
        self.gamma = float(cfg.loss.gamma)   # discount factor (lives under [loss])
 
        # Double DQN flag – add   double_dqn: true   to the yaml to enable
        self.double_dqn = bool(cfg.get("double_dqn", False))
 
        # Exploration schedule: linear decay ε_start → ε_end over anneal_steps
        self._eps             = float(cfg.exploration.eps_start_value)
        self._eps_end         = float(cfg.exploration.eps_end_value)
        self._eps_anneal_steps = int(cfg.exploration.eps_anneal_steps)
        self._eps_decay       = (self._eps - self._eps_end) / max(self._eps_anneal_steps, 1)

    def forward(self, tensordict: TensorDictBase):

        mode = exploration_type()
        obs = tensordict["observation"]

        # ------------------------------------------------------------ #
        # - TODO: Implement action selection based on learned Q Values
        # - TODO: Implement epsilon-greedy action selection for agent
        # ------------------------------------------------------------ #

        # Compute greedy action (action with highest Q-value) using online network
        with torch.no_grad():
            q_values      = self.net(obs)      # (..., n_actions)
            greedy_action = q_values.argmax(dim=-1)  # (...,)

        # Epsilon-greedy action selection
        if mode == ExplorationType.RANDOM or mode is None:
            # With probability ε take a uniformly random action
            rand_action  = torch.randint(
                low=0, high=self.net_output_dim,
                size=greedy_action.shape, device=obs.device,
            )
            explore_mask = torch.rand(greedy_action.shape, device=obs.device) < self._eps
            action_indices = torch.where(explore_mask, rand_action, greedy_action)
        else:
            # Evaluation / DETERMINISTIC mode – always greedy
            action_indices = greedy_action

        # Falls das Environment One-Hot erwartet, konvertieren wir hier, 
        # um die UserWarning zu vermeiden.
        if self.action_space_spec.shape: # Prüft ob Shape z.B. (4,) statt () ist
            tensordict["action"] = F.one_hot(action_indices.long(), num_classes=self.net_output_dim).float()
        else:
            tensordict["action"] = action_indices

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

        # Sicherstellen, dass actions [Batch, 1] ist und rewards/dones [Batch] sind
        # Falls actions One-Hot sind [Batch, N], wandle sie in Indices [Batch, 1] um
        if actions.ndim > 1 and actions.shape[-1] > 1:
            actions = actions.argmax(dim=-1, keepdim=True)
            
        actions = actions.long().view(-1, 1)
        rewards = rewards.view(-1)
        dones = dones.view(-1)

        # Compute TD targets using target network
        with torch.no_grad():
            if self.double_dqn:
                # Double DQN: online net selects a*, target net evaluates it
                best_next_actions = self.net(nxtobs).argmax(dim=-1, keepdim=True)
                next_q = self.target_net(nxtobs).gather(-1, best_next_actions).view(-1)
            else:
                # Standard DQN: target net selects and evaluates
                next_q = self.target_net(nxtobs).max(dim=-1).values
 
            # Q_TD = r + γ · Q_{θ⁻}(s', a') · (1 − done)
            td_targets = rewards + self.gamma * next_q * (1.0 - dones)
 
        # Q-values for the actions that were actually taken
        current_q = self.net(obs).gather(-1, actions).view(-1)
 
        loss = F.mse_loss(current_q, td_targets)
 
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=10.0)
        self.optimizer.step()

        # Update target network with Polyak averaging
        with torch.no_grad():
            for p_online, p_target in zip(
                self.net.parameters(), self.target_net.parameters()
            ):
                p_target.data.mul_(1.0 - self.tau).add_(self.tau * p_online.data)
            
        # Decay epsilon
        self._eps = max(self._eps_end, self._eps - self._eps_decay)

        # Log metrics
        metrics["epsilon"] = self._eps
        metrics["avg_dqn_loss"] = loss.item()
        metrics["avg_q_values"] = current_q.mean().item()

        return metrics
