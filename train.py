import torch
from src.config import Config
from src.engine import Trainer

CONFIG_NAME = "default"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    config = Config.from_toml(f"configs/{CONFIG_NAME}.toml")
    trainer = Trainer(config, device)
    trainer.run_loop()


if __name__ == "__main__":
    main()
