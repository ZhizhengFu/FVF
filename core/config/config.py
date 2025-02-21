import tomllib


class Config(dict):
    def __init__(self, config_path: str):
        super().__init__(self._load_config_dict(config_path))

    def _load_config_dict(self, config_path: str) -> dict:
        with open(config_path, "rb") as f:
            raw_config = tomllib.load(f)
        return raw_config


if __name__ == "__main__":
    config = Config("configs/usrnet.toml")
    for key in config:
        print(f"{key}: {config[key]}")
