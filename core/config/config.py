import tomllib


class Config:
    def __init__(self, config_path: str):
        self.config_dict = self._load_config_dict(config_path)

    def _load_config_dict(self, config_path: str):
        with open(config_path, "rb") as f:
            raw_config = tomllib.load(f)
        return raw_config


if __name__ == "__main__":
    config = Config("/Users/fuzz/Documents/FVF/configs/config.toml")
    print(config.config_dict)
