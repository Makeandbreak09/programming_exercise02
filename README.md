# Deep Q-Learning and Actor-Critic Methods

This repository contains practical exercises for implementing and evaluating Reinforcement Learning agents, with a specific focus on Deep Q-Networks (DQN) and Actor-Critic architectures. 

## Environment Setup and Installation

This project relies on several external Python dependencies. To manage these efficiently and ensure reproducible builds, we utilize `uv`, a fast, Rust-based Python package manager. 

**Step-by-Step Installation:**

1. **Install the package manager:** If you do not already have `uv` installed on your system, please follow the official setup instructions provided on the [`uv` documentation website](https://docs.astral.sh/uv/getting-started/installation/).
2. **Navigate to the project directory:** Open your terminal and change your current working directory to the root folder of this specific exercise (the `hw05` directory).
3. **Install dependencies in editable mode:** Execute the command below to install the project and its required dependencies. 

```bash
uv pip install -e .
```


## Setting up Weights & Biases (W&B)

This exercise utilizes Weights & Biases (`wandb`) for comprehensive experiment tracking, allowing you to monitor your agent's training metrics, such as episode rewards and loss values, in real-time. To leverage this functionality, you must configure a personal W&B account and authenticate your local development environment before initiating training.

**Step-by-Step Configuration:**

1. **Create an account:** Navigate to the [Weights & Biases registration page](https://wandb.ai) and create a free account. You can sign up using a standard email and password combination, or utilize single sign-on through platforms like GitHub or Google.
2. **Locate your API key:** Once your account is created and you are logged in, navigate to your user settings or directly access the [authorization page](https://wandb.ai/authorize). Here, you will find a unique, alphanumeric API key required to link your local training scripts to your cloud account. Copy this key to your clipboard.
3. **Authenticate your environment:** Return to your terminal and ensure you are within the project's root directory. Since we are managing our environment with `uv`, execute the following command to initiate the local login process:

```bash
uv run wandb login
```


## Getting Started

Once your environment is set up and the packages are installed, you can get started with this exercise. Run the following command from the root folder of this exercise:

````bash
uv run template/agents/dqn/train.py 
````

````bash
uv run template/agents/reinforce/train.py 
````

````bash
uv run template/agents/sac/train.py 
````