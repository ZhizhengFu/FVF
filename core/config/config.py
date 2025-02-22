import tomllib


class Config(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for key, value in self.items():
            if isinstance(value, dict):
                self[key] = Config(value)

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{name}'"
            )

    def __setattr__(self, name, value):
        self[name] = Config(value) if isinstance(value, dict) else value

    @classmethod
    def from_toml(cls, config_path: str):
        with open(config_path, "rb") as f:
            data = tomllib.load(f)
        return cls(data)


if __name__ == "__main__":
    config = Config.from_toml("configs/usrnet.toml")
    print(config.datasets.train.scales)
