import os
import torch
from src.engine import Trainer

CONFIG_NAME = "default"
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    trainer = Trainer(CONFIG_NAME, device)
    trainer.run_loop()


if __name__ == "__main__":
    main()
