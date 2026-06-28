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

        self.net = torch.nn.Sequential(
            torch.nn.Linear(self.net_input_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, self.net_output_dim),
        )
        # Optimizer
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=float(cfg.optim.lr))
        self.gamma = float(cfg.gamma)
        

    def forward(self, tensordict: TensorDictBase):

        mode = exploration_type()
        obs = tensordict["observation"]

        logits = self.net(obs)
        dist = torch.distributions.Categorical(logits=logits)

        if mode == ExplorationType.DETERMINISTIC:
            action = logits.argmax(dim=-1)
        else:
            action = dist.sample()

        if self.action_space_spec.shape:
            tensordict["action"] = F.one_hot(action.long(), num_classes=self.net_output_dim).float()
        else:
            tensordict["action"] = action
            
        tensordict["log_prob"] = dist.log_prob(action)

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

                if actions.ndim > 1 and actions.shape[-1] == self.net_output_dim:
                    action_indices = actions.argmax(dim=-1)
                else:
                    action_indices = actions

                logits = self.net(obs)
                dist = torch.distributions.Categorical(logits=logits)
                log_probs = dist.log_prob(action_indices)


                # ----------------------------------------------------------#
                # ---------- Compute Importance Sampling Weights ---------- #
                # ----------------------------------------------------------#

                if self.cfg.loss.use_importance_sampling:
                    log_ratio = log_probs - episode["log_prob"]
                    log_rho = torch.cumsum(log_ratio, dim=0)
                    # Clamp log_rho to avoid exp overflow
                    log_rho = torch.clamp(log_rho, max=20.0)
                    rho = torch.exp(log_rho)
                else:
                    rho = torch.ones_like(log_probs)


                # -----------------------------------------------------------#
                # ---------- Compute Monte-Carlo Return Estimates ---------- #
                # -----------------------------------------------------------#

                returns = []
                G = 0.0
                for r, d in zip(reversed(rewards), reversed(dones)):
                    # Handle both scalar and tensor rewards safely
                    r_val = r.item() if isinstance(r, torch.Tensor) else r
                    d_val = d.item() if isinstance(d, torch.Tensor) else d
                    
                    G = r_val + self.gamma * G * (1.0 - d_val)
                    returns.insert(0, G)
                returns = torch.tensor(returns, dtype=torch.float32, device=obs.device)


                # ----------------------------------------------------------#
                # ---------- Compute and Subtract Baseline Value ---------- #
                # ----------------------------------------------------------#

                if self.cfg.loss.subtract_baseline_value:
                    baseline = returns.mean()
                    advantages = returns - baseline
                else:
                    advantages = returns

                # ---------------------------------------------------#
                # ---------- Compute Policy Gradient Loss ---------- #
                # ---------------------------------------------------#

                episode_loss = - (rho.detach() * log_probs * advantages).mean()
                loss += episode_loss

            # ------------------------------------------------------#
            # ---------- Accumulate Loss Across Episodes ---------- #
            # ------------------------------------------------------#

            loss /= len(episodes)
            losses.append(loss.item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        metrics["loss"] = sum(losses) / len(losses)
        return metrics
