"""Get and process config, create agent, run agent."""

# pylint: disable=no-member
import os
import argparse
from utils.config import process_config
from agents.dqnagent import *


def main(test=False):
    # Get the config file from the run arguments
    argparser = argparse.ArgumentParser(description="Load a config file, train or test")
    argparser.add_argument(
        "config",
        metavar="config_json_file",
        default="None",
        help="The Configuration file in json format",
    )
    args = argparser.parse_args()
    config = process_config("configs" + "/" + args.config)

    # make directory for model checkpoints
    if not test:
        config.model_dir = config.checkpoint_dir + f"{config.exp_name}"
        config.model_dir = f"results/models/{config.exp_name}"
        os.makedirs(config.model_dir, exist_ok=True)

    if config.test:
        # create checkpoint path
        config.checkpoint_file = (
            config.checkpoint_dir + config.exp_name + "/" + config.checkpoint
        )

    agent_class = globals()[config.agent]
    agent = agent_class(config, config.test)
    # agent = DQNAgent(config)
    # agent.load_checkpoint(config.checkpoint_file)
    if not config.test:
        wandb.init(project="drl2023-dqn", name=config.exp_name, config=config)
        agent.prefill_replay_buffer()
    agent.train()


if __name__ == "__main__":
    main(test=False)
