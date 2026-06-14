from collections import defaultdict

import wandb
import hydra
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
from torchrl.envs import GymEnv

from template.utils import evaluate
from template.agents.reinforce.agent import REINFORCE

MAX_CARTPOLE_EP_LENGTH = 500


@hydra.main(config_path="../../../configs", config_name="reinforce", version_base="1.1")
def train(cfg: DictConfig):

    wandb.init(project=cfg.logging.project, config=OmegaConf.to_container(cfg))

    def createfn():
        return GymEnv(cfg.env.name).to(cfg.device)

    # ---------------------------------------------#
    # ---------- Create REINFORCE Agent ---------- #
    # ---------------------------------------------#

    agent = REINFORCE(
        cfg=cfg,
        action_space_spec=createfn().action_spec_unbatched,
        observation_space_spec=createfn().observation_spec_unbatched,
    )

    # -------------------------------------------------------#
    # ---------- Starting REINFORCE Training Loop ---------- #
    # -------------------------------------------------------#

    status, reinf_train_metrics, reinf_eval_metrics = defaultdict(float), {}, {}

    with tqdm(total=cfg.collector.total_num_episodes, desc="REINFORCE", leave=True) as pbar:
        while pbar.n < cfg.collector.total_num_episodes:
            # --------------------------------------------------#
            # ---------- Collect Episodes with Agent ---------- #
            # --------------------------------------------------#

            episodes = [
                createfn().rollout(max_steps=MAX_CARTPOLE_EP_LENGTH, policy=agent)
                for epidx in range(cfg.collector.num_episodes_per_iter)
            ]

            status["iter_num_episodes"] = len(episodes)
            status["total_num_episodes"] = status["total_num_episodes"] + len(episodes)
            status["avg_episode_len"] = sum(len(ep["next"]["reward"]) for ep in episodes) / len(episodes)
            status["avg_episode_rew"] = sum(ep["next"]["reward"].sum().cpu().item() for ep in episodes)
            status["avg_episode_rew"] = status["avg_episode_rew"] / len(episodes)

            # -------------------------------------------------#
            # ---------- Update Agent with Episodes ---------- #
            # -------------------------------------------------#

            reinf_train_metrics = agent.update(episodes, steps=int(status["total_num_episodes"]))

            # ----------------------------------------------------#
            # ---------- Evaluate Performance of Agent ---------- #
            # ----------------------------------------------------#

            if (
                pbar.n // cfg.evaluation.eval_ep_freq
                > (pbar.n - cfg.collector.num_episodes_per_iter) // cfg.evaluation.eval_ep_freq
            ):
                reinf_eval_metrics = evaluate(agent, createfn, nepisodes=cfg.evaluation.episodes)

            status = {**status, **reinf_train_metrics, **reinf_eval_metrics}

            wandb.log(status, step=int(status["total_num_episodes"]))
            pbar.set_postfix({key: "{:.5f}".format(val) for key, val in status.items()})

            pbar.update(cfg.collector.num_episodes_per_iter)

    print("Training Completed!")


if __name__ == "__main__":
    train()
