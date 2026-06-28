import os
import sys
import argparse
import imageio
import numpy as np
import torch
import gymnasium as gym
from hydra import initialize_config_dir, compose
from torchrl.envs import GymEnv
from torchrl.envs.utils import set_exploration_type, ExplorationType

# Add template directory to path to ensure imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(r"c:\Users\simon\OneDrive\Dokumente\Uni\26SS\DeepRL\programming_exercise02")

from template.agents.dqn.agent import DeepQNetwork
from template.agents.reinforce.agent import REINFORCE
from template.agents.sac.agent import SoftActorCritic

def record_gif_dqn(artifact_dir):
    print("Initializing DQN config (original settings)...")
    with initialize_config_dir(version_base="1.1", config_dir=r"c:\Users\simon\OneDrive\Dokumente\Uni\26SS\DeepRL\programming_exercise02\configs"):
        cfg = compose(config_name="dqn", overrides=[
            "logging.project=dqn-vis"
        ])

    def createfn():
        return GymEnv(cfg.env.name)

    from torchrl.collectors import Collector
    from torchrl.data import LazyTensorStorage, ReplayBuffer
    from torchrl.data.replay_buffers import samplers

    print("Creating DQN Agent...")
    env_sample = createfn()
    agent = DeepQNetwork(
        cfg=cfg,
        action_space_spec=env_sample.action_spec_unbatched,
        observation_space_spec=env_sample.observation_spec_unbatched,
    )

    dqn_replay_storage = LazyTensorStorage(max_size=cfg.buffer.buffer_size, device=cfg.device)
    dqn_replay_buffer = ReplayBuffer(storage=dqn_replay_storage, batch_size=cfg.buffer.batch_size, sampler=samplers.RandomSampler())
    
    dqn_data_collector = Collector(
        policy=agent,
        create_env_fn=createfn,
        total_frames=cfg.collector.total_frames,
        frames_per_batch=cfg.collector.frames_per_batch,
        init_random_frames=cfg.collector.init_random_frames,
        device=cfg.device,
    )

    print(f"Training DQN agent for {cfg.collector.total_frames} steps...")
    num_env_steps = 0
    for rollout in dqn_data_collector:
        num_env_steps += len(rollout)
        dqn_replay_buffer.extend(rollout)
        if len(dqn_replay_buffer) < cfg.collector.init_random_frames:
            continue
        
        for _ in range(cfg.loss.num_updates):
            agent.update(dqn_replay_buffer, steps=num_env_steps)
        print(f"Steps: {num_env_steps}/{cfg.collector.total_frames}", end="\r")

    dqn_data_collector.shutdown()
    print("\nTraining complete. Recording rollout GIF...")

    gym_env = gym.make(cfg.env.name, render_mode="rgb_array")
    obs, info = gym_env.reset()
    frames = []
    
    with set_exploration_type(ExplorationType.DETERMINISTIC):
        for _ in range(500):
            frames.append(gym_env.render())
            obs_tensor = torch.tensor(obs, dtype=torch.float32)
            with torch.no_grad():
                q_values = agent.net(obs_tensor)
                action = q_values.argmax(dim=-1).item()
            obs, reward, terminated, truncated, info = gym_env.step(action)
            if terminated or truncated:
                break
                
    gym_env.close()
    
    gif_path = os.path.join(artifact_dir, "lunarlander_play.gif")
    imageio.mimsave(gif_path, frames, fps=30)
    print(f"Saved LunarLander visualization to {gif_path}")

def record_gif_reinforce(artifact_dir):
    print("Initializing REINFORCE config (original settings)...")
    with initialize_config_dir(version_base="1.1", config_dir=r"c:\Users\simon\OneDrive\Dokumente\Uni\26SS\DeepRL\programming_exercise02\configs"):
        cfg = compose(config_name="reinforce", overrides=[
            "collector.total_num_episodes=2500",
            "loss.subtract_baseline_value=true",
            "loss.use_importance_sampling=true",
            "logging.project=reinforce-vis"
        ])

    def createfn():
        return GymEnv(cfg.env.name)

    print("Creating REINFORCE Agent...")
    env_sample = createfn()
    agent = REINFORCE(
        cfg=cfg,
        action_space_spec=env_sample.action_spec_unbatched,
        observation_space_spec=env_sample.observation_spec_unbatched,
    )

    print(f"Training REINFORCE agent for {cfg.collector.total_num_episodes} episodes...")
    total_episodes = 0
    while total_episodes < cfg.collector.total_num_episodes:
        episodes = [
            createfn().rollout(max_steps=500, policy=agent)
            for _ in range(cfg.collector.num_episodes_per_iter)
        ]
        agent.update(episodes, steps=total_episodes)
        total_episodes += len(episodes)
        print(f"Episodes: {total_episodes}/{cfg.collector.total_num_episodes}", end="\r")

    print("\nTraining complete. Recording rollout GIF...")
    
    gym_env = gym.make("CartPole-v1", render_mode="rgb_array")
    obs, info = gym_env.reset()
    frames = []
    
    with set_exploration_type(ExplorationType.DETERMINISTIC):
        for _ in range(500):
            frames.append(gym_env.render())
            obs_tensor = torch.tensor(obs, dtype=torch.float32)
            logits = agent.net(obs_tensor)
            action = logits.argmax(dim=-1).item()
            obs, reward, terminated, truncated, info = gym_env.step(action)
            if terminated or truncated:
                break
                
    gym_env.close()
    
    gif_path = os.path.join(artifact_dir, "cartpole_play.gif")
    imageio.mimsave(gif_path, frames, fps=30)
    print(f"Saved CartPole visualization to {gif_path}")

def record_gif_sac(artifact_dir):
    print("Initializing SAC config (original settings)...")
    with initialize_config_dir(version_base="1.1", config_dir=r"c:\Users\simon\OneDrive\Dokumente\Uni\26SS\DeepRL\programming_exercise02\configs"):
        cfg = compose(config_name="sac", overrides=[
            "logging.project=sac-vis"
        ])

    def createfn():
        return GymEnv(cfg.env.name)

    from torchrl.collectors import Collector
    from torchrl.data import LazyTensorStorage, ReplayBuffer
    from torchrl.data.replay_buffers import samplers

    print("Creating SAC Agent...")
    env_sample = createfn()
    agent = SoftActorCritic(
        cfg=cfg,
        action_space_spec=env_sample.action_spec_unbatched,
        observation_space_spec=env_sample.observation_spec_unbatched,
    )

    sac_replay_storage = LazyTensorStorage(max_size=cfg.buffer.buffer_size, device=cfg.device)
    sac_replay_buffer = ReplayBuffer(storage=sac_replay_storage, batch_size=cfg.buffer.batch_size, sampler=samplers.RandomSampler())
    
    sac_data_collector = Collector(
        policy=agent,
        create_env_fn=createfn,
        total_frames=cfg.collector.total_frames,
        frames_per_batch=cfg.collector.frames_per_batch,
        init_random_frames=cfg.collector.init_random_frames,
        device=cfg.device,
    )

    print(f"Training SAC agent for {cfg.collector.total_frames} steps...")
    num_env_steps = 0
    for rollout in sac_data_collector:
        num_env_steps += len(rollout)
        sac_replay_buffer.extend(rollout)
        if len(sac_replay_buffer) < cfg.collector.init_random_frames:
            continue
        
        for _ in range(cfg.loss.num_updates * cfg.collector.frames_per_batch):
            agent.update(sac_replay_buffer, steps=num_env_steps)
        print(f"Steps: {num_env_steps}/{cfg.collector.total_frames}", end="\r")

    sac_data_collector.shutdown()
    print("\nTraining complete. Recording rollout GIF...")

    gym_env = gym.make("Pendulum-v1", render_mode="rgb_array")
    obs, info = gym_env.reset()
    frames = []
    
    with set_exploration_type(ExplorationType.DETERMINISTIC):
        for _ in range(200):
            frames.append(gym_env.render())
            obs_tensor = torch.tensor(obs, dtype=torch.float32)
            _, _, mean_action = agent.actor.get_action(obs_tensor)
            action = mean_action.detach().cpu().numpy()
            obs, reward, terminated, truncated, info = gym_env.step(action)
            if terminated or truncated:
                break
                
    gym_env.close()
    
    gif_path = os.path.join(artifact_dir, "pendulum_play.gif")
    imageio.mimsave(gif_path, frames, fps=30)
    print(f"Saved Pendulum visualization to {gif_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, required=True, choices=["dqn", "reinforce", "sac"])
    parser.add_argument("--outdir", type=str, required=True)
    args = parser.parse_args()
    
    if args.env == "dqn":
        record_gif_dqn(args.outdir)
    elif args.env == "reinforce":
        record_gif_reinforce(args.outdir)
    elif args.env == "sac":
        record_gif_sac(args.outdir)