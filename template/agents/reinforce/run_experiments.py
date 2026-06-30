import subprocess
import os
import json
import matplotlib.pyplot as plt
import numpy as np

def main():
    # Set wandb to offline mode to run without credentials or blocking
    os.environ["WANDB_MODE"] = "offline"

    experiments = {
        "basic": {
            "name": "Basic REINFORCE (No Baseline, No IS)",
            "dir": "outputs/reinforce/reinforce_basic",
            "overrides": [
                "loss.num_update_iters=1",
                "loss.subtract_baseline_value=false",
                "loss.use_importance_sampling=false",
                "collector.total_num_episodes=1000",
                "evaluation.eval_ep_freq=50",
                "hydra.run.dir=outputs/reinforce/reinforce_basic"
            ]
        },
        "baseline": {
            "name": "REINFORCE with Baseline (No IS)",
            "dir": "outputs/reinforce/reinforce_baseline",
            "overrides": [
                "loss.num_update_iters=1",
                "loss.subtract_baseline_value=true",
                "loss.use_importance_sampling=false",
                "collector.total_num_episodes=1000",
                "evaluation.eval_ep_freq=50",
                "hydra.run.dir=outputs/reinforce/reinforce_baseline"
            ]
        },
        "is_5": {
            "name": "REINFORCE with Baseline & IS (Num Updates = 5)",
            "dir": "outputs/reinforce/reinforce_is_5",
            "overrides": [
                "loss.num_update_iters=5",
                "loss.subtract_baseline_value=true",
                "loss.use_importance_sampling=true",
                "collector.total_num_episodes=1000",
                "evaluation.eval_ep_freq=50",
                "hydra.run.dir=outputs/reinforce/reinforce_is_5"
            ]
        },
        "is_10": {
            "name": "REINFORCE with Baseline & IS (Num Updates = 10)",
            "dir": "outputs/reinforce/reinforce_is_10",
            "overrides": [
                "loss.num_update_iters=10",
                "loss.subtract_baseline_value=true",
                "loss.use_importance_sampling=true",
                "collector.total_num_episodes=1000",
                "evaluation.eval_ep_freq=50",
                "hydra.run.dir=outputs/reinforce/reinforce_is_10"
            ]
        },
        "is_50": {
            "name": "REINFORCE with Baseline & IS (Num Updates = 50)",
            "dir": "outputs/reinforce/reinforce_is_50",
            "overrides": [
                "loss.num_update_iters=50",
                "loss.subtract_baseline_value=true",
                "loss.use_importance_sampling=true",
                "collector.total_num_episodes=1000",
                "evaluation.eval_ep_freq=50",
                "hydra.run.dir=outputs/reinforce/reinforce_is_50"
            ]
        }
    }

    # Run each experiment if not already done
    for key, exp in experiments.items():
        history_path = os.path.join(exp["dir"], "history.json")
        if os.path.exists(history_path):
            print(f"Skipping experiment {exp['name']} as it has already finished.")
            continue

        print(f"\n==========================================")
        print(f"Starting experiment: {exp['name']}")
        print(f"==========================================")
        cmd = ["uv", "run", "template/agents/reinforce/train.py"] + exp["overrides"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error running {exp['name']}:")
            print(result.stderr)
            raise RuntimeError(f"Experiment {key} failed")
        print(f"Finished experiment: {exp['name']}")

    # Load data and plot
    data = {}
    for key, exp in experiments.items():
        history_path = os.path.join(exp["dir"], "history.json")
        if not os.path.exists(history_path):
            raise FileNotFoundError(f"Could not find history file at {history_path}")
        with open(history_path, "r") as f:
            data[key] = json.load(f)

    # Ensure output directory for plots exists
    os.makedirs("plots", exist_ok=True)

    # Plot Basic REINFORCE
    plt.figure(figsize=(8, 5))
    episodes = [h["total_num_episodes"] for h in data["basic"]]
    returns = [h["avg_eval_ep_rew"] for h in data["basic"]]
    plt.plot(episodes, returns, label="Basic REINFORCE", color="#E66101")
    plt.xlabel("Number of Episodes")
    plt.ylabel("Evaluation Return")
    plt.title("Basic REINFORCE on CartPole-v1")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/reinforce_basic.png", dpi=300)
    plt.close()

    # Plot REINFORCE with Baseline
    plt.figure(figsize=(8, 5))
    episodes = [h["total_num_episodes"] for h in data["baseline"]]
    returns = [h["avg_eval_ep_rew"] for h in data["baseline"]]
    plt.plot(episodes, returns, label="REINFORCE + Baseline", color="#5E3C99")
    plt.xlabel("Number of Episodes")
    plt.ylabel("Evaluation Return")
    plt.title("REINFORCE with Baseline on CartPole-v1")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/reinforce_baseline.png", dpi=300)
    plt.close()

    # Plot Importance Sampling ablation (5, 10, 50 update iterations) vs basic/baseline
    plt.figure(figsize=(10, 6))
    colors = {
        "basic": "#E66101",
        "baseline": "#5E3C99",
        "is_5": "#2C7BB6",
        "is_10": "#ABD9E9",
        "is_50": "#FDAE61"
    }
    
    # Plot all for comparison
    for key, exp in experiments.items():
        ep = [h["total_num_episodes"] for h in data[key]]
        ret = [h["avg_eval_ep_rew"] for h in data[key]]
        plt.plot(ep, ret, label=exp["name"], color=colors[key], linewidth=1.8)

    plt.xlabel("Number of Episodes")
    plt.ylabel("Evaluation Return")
    plt.title("Ablation Study: Importance Sampling & Baseline Subtraction")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/reinforce_importance_sampling.png", dpi=300)
    plt.close()

    # Generate a combined plot with just basic vs baseline
    plt.figure(figsize=(9, 5.5))
    plt.plot([h["total_num_episodes"] for h in data["basic"]], [h["avg_eval_ep_rew"] for h in data["basic"]], label="Basic REINFORCE", color="#E66101", linewidth=2)
    plt.plot([h["total_num_episodes"] for h in data["baseline"]], [h["avg_eval_ep_rew"] for h in data["baseline"]], label="REINFORCE with Baseline", color="#5E3C99", linewidth=2)
    plt.xlabel("Number of Episodes")
    plt.ylabel("Evaluation Return")
    plt.title("Comparison: Basic REINFORCE vs. REINFORCE with Baseline")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/reinforce_comparison.png", dpi=300)
    plt.close()

    print("\nAll experiments complete! Plots saved in the 'plots/' directory.")

if __name__ == "__main__":
    main()
