import sys
import os
import argparse
from rl_zoo3.train import train

# Local imports
from layers.qcnn import HybridCnnPolicy


def main():
    # Create argument parser for this script
    parser = argparse.ArgumentParser(description='Train DQN on Pong')
    parser.add_argument('--conf', type=str, help='Path to the configuration file')
    script_args, _ = parser.parse_known_args()
    
    # Extract config file path if provided
    config_path = script_args.conf
    
    # Set wandb directory
    os.environ["WANDB_DIR"] = "/data/cvcqml/common/ycordero/dqn-pong-logs/wandb"
    
    # Set the base training arguments
    train_args = ["python", "--algo", "dqn", "--env", "PongNoFrameskip-v4", "--track", 
                  "--wandb-project-name", "dqn_pong", "--uuid", "--gym-packages", "ale_py", 
                  "--tensorboard-log", "/data/cvcqml/common/ycordero/dqn-pong-logs/tensorboard", 
                  "-P", "--optimization-log-path", "/data/cvcqml/common/ycordero/dqn-pong-logs/optim", 
                  "-f", "/data/cvcqml/common/ycordero/dqn-pong-logs/logs"]
    
    # Add config file argument if provided
    if config_path:
        print(f"Configuration YAML file path given with path: {config_path}")
        train_args.extend(["-conf", config_path])
    
    # Set sys.argv and call train
    sys.argv = train_args
    train()

if __name__ == "__main__":
    main()