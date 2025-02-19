from core.config import Config
from core.trainer import Trainer


def main():
    config = Config("configs/config.toml")
    _ = Trainer(config)
    print(config.config_dict)

if __name__ == "__main__":
    main()
