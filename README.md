![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Python](https://img.shields.io/badge/PyThon-3670A0.svg?style=for-the-badge&logo=Python&logoColor=ffdd54)
![Jupyter](https://img.shields.io/badge/Jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)
![PyCharm](https://img.shields.io/badge/pycharm-143?style=for-the-badge&logo=pycharm&logoColor=black&color=black&labelColor=green)
![WandB](https://img.shields.io/badge/Weights_&_Biases-FFBE00?style=for-the-badge&logo=WeightsAndBiases&logoColor=white)

<p align="center">
    <h3 align="center">Deep Q-Networks: Pong game - Activity</h3>
    <p align="center">
        Reinforcement Learning (RL) activity using DQN for solving the Pong game.
    </p>
</p>

> [!IMPORTANT]
> The best model weights are available in Google Drive from the [following link](https://drive.google.com/file/d/11nMg-szJpWDAXoiFbTZf7xWZdo5VBVJF/view?usp=sharing).

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#introduction">Introduction</a></li>
    <li><a href="#installation">Installation</a></li>
    <li><a href="#license">License</a></li>
  </ol>
</details>

## Introduction

This is the official repository of the DQN experimentation optional exercise for solving the Pong game in the Master in Computer Vision in Barcelona.

### Folder Structure

```
├── config/
│   ├── default.yaml                   # Configuration YAML files for model experiments
│   └── ...
│
├── src/
│   ├── train.py                       # Script to train the DQN model on Pong
│   └── test.py                        # Script to test the trained model
│
├── third_party/
│   └── stable_baselines3/            # Third-party resources such as Stable Baselines3
│
├── videos/
│   ├── best_model.gif                # GIF showing the best-performing trained agent
│   └── worst_model.gif               # GIF showing the worst-performing trained agent
```

## Installation

### Prerequisites

- Python 3.9+ (PyTorch 2.3+ requirement)
- For Windows users, Anaconda is recommended for easier installation of packages

### Setting Up Your Environment

1. **Create a virtual environment** (recommended):
   ```bash
   # Using venv
   python -m venv sb3_env
   source sb3_env/bin/activate  # On Windows: sb3_env\Scripts\activate
   
   # OR using Anaconda
   conda create -n sb3_env python=3.9
   conda activate sb3_env
   ```

2. **Install Stable Baselines3 with extras**:
   ```bash
   # Standard shells
   pip install stable-baselines3[extra]
   
   # Zsh or shells requiring quotes around brackets
   pip install 'stable-baselines3[extra]'
   ```

   This includes optional dependencies like Tensorboard, OpenCV or ale-py to train on Atari games.

3. **Install Atari dependencies**:
   ```bash
   pip install gymnasium ale-py
   ```

### Troubleshooting

- If you encounter vague errors related to missing DLL files and modules when creating Atari environments, this is an issue with the atari-py package. See the Stable Baselines3 documentation for more information.

- If you need to work with OpenCV on a machine without a X-server (for instance inside a docker image), you will need to install `opencv-python-headless`.

- If you encounter installation issues with newer Python versions, try creating an environment with Python 3.9 as it's known to be compatible.

### Alternative Installation Methods

- **Provided development version**:
  ```bash
  git submodule update --init --recursive
  cd third_party/rl-baselines3-zoo
  pip install -e .[extra]
  ```

## License

The project is licensed using MIT License (see the license [here](LICENSE)).
