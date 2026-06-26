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
        
        # Create the policy network
        self.net = torch.nn.Sequential(
            torch.nn.Linear(self.net_input_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, self.net_output_dim)
        )
        
        # Set up the Adam optimizer
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=float(cfg.optim.lr))
        
        # Read parameters from the config (baseline and importance sampling)
        self.gamma = float(cfg.gamma)
        self.use_baseline = bool(cfg.loss.get("subtract_baseline_value", False))

        self.use_is = bool(cfg.loss.get("use_is", False))
    def forward(self, tensordict: TensorDictBase):

        mode = exploration_type()
        obs = tensordict["observation"]

        # ------------------------------------------------------------ #
        # - TODO: Implement action selection based on learned Q Values
        # - TODO: Implement epsilon-greedy action selection for agent
        # (Note: the comments above are a mistake in the template, there are no Q Values or Epsilon-Greedy here!)
        # ------------------------------------------------------------ #

        # Get the raw network outputs (logits)
        logits = self.net(obs)
        
        # Create a categorical probability distribution
        dist = torch.distributions.Categorical(logits=logits)

        if mode == ExplorationType.RANDOM or mode is None:
            # Stochastic action selection (sample based on our probabilities)
            action = dist.sample()
        else:
            # For evaluation, choose the action with the highest probability
            action = logits.argmax(dim=-1)

        tensordict["action"] = action
        
        # SAVE the probability of the selected action. This is required for Importance Sampling!
        # We store the log probability specifically, because it is easier to compute with.
        tensordict["behavior_log_prob"] = dist.log_prob(action).detach()

        return tensordict

    def update(self, episodes: List, steps: int):
        metrics = {}
        total_loss_val = 0.0

        # Update iterations (for Importance Sampling num_update_iters can be > 1)
        for niter in range(self.cfg.loss.num_update_iters):
            iteration_loss = 0.0

            for episode in episodes:
                obs = episode["observation"]
                actions = episode["action"].squeeze(-1) # Ensure correct shape (1D vector)
                rewards = episode["next"]["reward"].squeeze(-1)
                
                # Old log probabilities (from when the data was collected)
                old_log_probs = episode["behavior_log_prob"].squeeze(-1)

                # ------------------------------------------------------------#
                # ---------- Compute Log. Prob. of Selected Action ---------- #
                # ------------------------------------------------------------#
                
                # Compute NEW probabilities for the same states
                logits = self.net(obs)
                dist = torch.distributions.Categorical(logits=logits)
                current_log_probs = dist.log_prob(actions)

                # ----------------------------------------------------------#
                # ---------- Compute Importance Sampling Weights ---------- #
                # ----------------------------------------------------------#
                
                if self.use_is and niter > 0:
                    # rho_t = exp( sum(current_log_prob - old_log_prob) )
                    # Use torch.cumsum for the cumulative sum of probabilities up to step t
                    log_rhos = current_log_probs - old_log_probs
                    is_weights = torch.exp(torch.cumsum(log_rhos, dim=0)).detach()
                else:
                    # If IS is disabled or it's the first iteration, weight = 1
                    is_weights = torch.ones_like(current_log_probs)

                # -----------------------------------------------------------#
                # ---------- Compute Monte-Carlo Return Estimates ---------- #
                # -----------------------------------------------------------#
                
                T = rewards.size(0)
                returns = torch.zeros(T, device=rewards.device)
                G = 0.0
                
                # Calculate G_t from the end of the episode to the beginning
                for t in reversed(range(T)):
                    G = rewards[t] + self.gamma * G
                    returns[t] = G

                # ----------------------------------------------------------#
                # ---------- Compute and Subtract Baseline Value ---------- #
                # ----------------------------------------------------------#
                
                if self.use_baseline:
                    # Subtract the episode mean to reduce variance
                    baseline = returns.mean()
                    returns = returns - baseline

                # ---------------------------------------------------#
                # ---------- Compute Policy Gradient Loss ---------- #
                # ---------------------------------------------------#
                
                # Loss = - Sum ( IS_Weight * Log_Prob * Return )
                episode_loss = -torch.sum(is_weights * current_log_probs * returns)

                # ------------------------------------------------------#
                # ---------- Accumulate Loss Across Episodes ---------- #
                # ------------------------------------------------------#
                
                iteration_loss += episode_loss

            # Average loss across all episodes in the current batch
            iteration_loss = iteration_loss / len(episodes)

            # Update network weights
            self.optimizer.zero_grad()
            iteration_loss.backward()
            self.optimizer.step()
            
            total_loss_val += iteration_loss.item()

        # Log metrics (take the average over all iterations)
        metrics["avg_policy_loss"] = total_loss_val / self.cfg.loss.num_update_iters

        return metrics
