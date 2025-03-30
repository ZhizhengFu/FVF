import os
import torch
from src.engine import Trainer

CONFIG_NAME = "default"
os.environ["ALBUMENTATIONS_DISABLE_VERSION_CHECK"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    trainer = Trainer(CONFIG_NAME, device)
    trainer.run_loop()


if __name__ == "__main__":
    main()
