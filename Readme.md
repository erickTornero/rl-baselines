# Reinforcement Learning Baselines

The goal of this repository is to learn a little more about reinforcement learning algorithms.

<p align="center">
    <img src="./assets/reinforce_out_example.gif" />
    <img src="./assets/dqn_breakout.gif" />
    <img src="./assets/dqn_pong.gif" />
</p>

### Stack:
- Pytorch
- Pytorch Lightning
- TensorDict

## Getting Started

### Install

Please [install uv](https://docs.astral.sh/uv/getting-started/installation/#standalone-installer)

Then install the project
```
uv sync
```

### Training

Example: To train the Reinforce algorithm, use the following command:
```
uv run rl-runner --train --config configs/reinforce.yaml
```
You can customize the training parameters, such as the number of episodes. For example, to set the maximum number of training episodes to 500:

```
uv run rl-runner --train --config configs/reinforce.yaml trainer.max_episodes=500
```

### Evaluation

To evaluate your models, use the same launch.py script but with the --test flag. You'll also need to specify the configuration file and the checkpoint from the outputs folder. Here’s an example:

```
uv run rl-runner --test --config outputs/reinforce-discrete/../parsed.yaml --resume=outputs/reinforce-discrete/.../checkpoint.ckpt
```

By default, this command will print the cumulative reward for each episode. If you'd like to render the environment and save a video, add the following options:

```
uv run rl-runner --test --config outputs/reinforce-discrete/../parsed.yaml --resume=outputs/reinforce-discrete/.../checkpoint.ckpt system.environment.render=True --save-video
```

## Contributions
We welcome contributions! If you'd like to add new features, improve documentation, or fix bugs, please create a pull request.
