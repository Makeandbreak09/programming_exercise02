from collections import defaultdict

import wandb
import hydra
import torchrl
import gymnasium as gym
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
from torchrl.envs import GymEnv
from torchrl.collectors import Collector
from torchrl.data.replay_buffers import samplers
from torchrl.data import LazyTensorStorage, ReplayBuffer

from template.utils import evaluate
from template.agents.sac.agent import SoftActorCritic
import numpy as np


@hydra.main(config_path="../../../configs", config_name="sac", version_base="1.1")
def train(cfg: DictConfig):

    wandb.init(project=cfg.logging.project, config=OmegaConf.to_container(cfg))

    def createfn():
        return GymEnv(cfg.env.name)

    # -----------------------------------------------------#
    # ---------- Create Soft Actor-Critic Agent ---------- #
    # -----------------------------------------------------#

    agent = SoftActorCritic(
        cfg=cfg,
        action_space_spec=createfn().action_spec_unbatched,
        observation_space_spec=createfn().observation_spec_unbatched,
    )

    # ------------------------------------------------------------------#
    # ---------- Create Experience Replay Buffers & Storages ---------- #
    # ------------------------------------------------------------------#

    sac_replay_storage = LazyTensorStorage(
        max_size=cfg.buffer.buffer_size,
        device=cfg.device,
    )

    sac_replay_buffer = ReplayBuffer(
        storage=sac_replay_storage,
        batch_size=cfg.buffer.batch_size,
        sampler=samplers.RandomSampler(),
    )

    # -------------------------------------------------------------#
    # ---------- Create Data Collectors for Environment ---------- #
    # -------------------------------------------------------------#

    sac_data_collector = Collector(
        policy=agent,
        create_env_fn=createfn,
        total_frames=cfg.collector.total_frames,
        frames_per_batch=cfg.collector.frames_per_batch,
        init_random_frames=cfg.collector.init_random_frames,
        device=cfg.device,
    )

    # ---------------------------------------------------------------#
    # ---------- Starting Soft Actor-Critic Training Loop ---------- #
    # ---------------------------------------------------------------#

    status, sac_train_metrics, sac_eval_metrics = defaultdict(float), {}, {}

    with tqdm(total=cfg.collector.total_frames, desc="SAC", leave=True) as pbar:

        for outidx, rollout in tqdm(enumerate(sac_data_collector)):

            status["num_env_steps"] = status["num_env_steps"] + len(rollout)
            status["avg_step_rew"] = rollout["next"]["reward"].mean().cpu().item()
            status["std_step_rew"] = rollout["next"]["reward"].std().cpu().item()

            sac_replay_buffer.extend(rollout)

            if len(sac_replay_buffer) < cfg.collector.init_random_frames:
                continue
            
            metrics = []
            for idx in range(cfg.loss.num_updates * cfg.collector.frames_per_batch):
                sac_train_metrics = agent.update(
                    sac_replay_buffer, steps=int(status["num_env_steps"])
                )
                metrics += [sac_train_metrics]
            sac_train_metrics = {
                k: np.mean([metric[k] for metric in metrics]) for k in metrics[0].keys()
            }

            if outidx % (cfg.evaluation.eval_step_freq // cfg.collector.frames_per_batch) == 0:
                sac_eval_metrics = evaluate(agent, createfn, nepisodes=cfg.evaluation.episodes)

            status = {**status, **sac_train_metrics, **sac_eval_metrics}

            wandb.log(status, step=int(status["num_env_steps"]))
            pbar.set_postfix({key: "{:.5f}".format(val) for key, val in status.items()})

            pbar.update(len(rollout))

    sac_data_collector.shutdown()


if __name__ == "__main__":
    train()
