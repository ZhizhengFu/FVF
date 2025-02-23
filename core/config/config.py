import tomllib
import warnings


class Config(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.update(
            {
                key: Config(value)
                if isinstance(value, dict)
                else None
                if value == ""
                else value
                for key, value in self.items()
            }
        )

    def __getattr__(self, name):
        return self[name]

    def __setattr__(self, name, value):
        self[name] = Config(value) if isinstance(value, dict) else value

    def __missing__(self, key):
        warnings.warn(
            f"Key '{key}' is missing in the configuration. Returning None.",
            UserWarning,
            stacklevel=3,
        )
        return None

    @classmethod
    def from_toml(cls, config_path: str):
        with open(config_path, "rb") as f:
            data = tomllib.load(f)
        return cls(data)


if __name__ == "__main__":
    config = Config.from_toml("configs/usrnet.toml")
    print(config.model.pretrained_path)  # return None
    print(config.model.pretrained_pat)  # return Warning + None
