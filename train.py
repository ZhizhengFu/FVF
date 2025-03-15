from src.config import Config
from src.engine import Trainer

CONFIG_NAME = "default"


def main():
    config = Config.from_toml(f"configs/{CONFIG_NAME}.toml")
    trainer = Trainer(config)
    trainer.run_loop()


if __name__ == "__main__":
    main()
