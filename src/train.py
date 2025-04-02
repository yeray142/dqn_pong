import sys
import os

from rl_zoo3.train import train

# Set wandb directory
os.environ["WANDB_DIR"] = "/data/cvcqml/common/ycordero/dqn-pong-logs/wandb"

# Set the training arguments
sys.argv = ["python", "--algo", "dqn", "--env", "PongNoFrameskip-v4", "--track", "--wandb-project-name", "dqn_pong", "--uuid", 
            "--gym-packages", "ale_py", "--tensorboard-log", "/data/cvcqml/common/ycordero/dqn-pong-logs/tensorboard", "-P", 
            "--optimization-log-path", "/data/cvcqml/common/ycordero/dqn-pong-logs/optim", "-f", "/data/cvcqml/common/ycordero/dqn-pong-logs/logs"]
train()