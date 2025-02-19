from core.config import Config
from core.engine import Trainer


def main():
    config = Config("configs/config.toml")
    trainer = Trainer(config)
    trainer.run_loop()


if __name__ == "__main__":
    main()
