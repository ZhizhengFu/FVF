import tomllib
from typing import Any


class Config:
    def __init__(self, config_path: str):
        self.config_dict = self._load_config_dict(config_path)

    def _load_config_dict(self, config_path: str) -> dict:
        with open(config_path, "rb") as f:
            raw_config = tomllib.load(f)
        return raw_config

    def __getitem__(self, key: str) -> Any:
        return self.config_dict[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.config_dict[key] = value

    def __delitem__(self, key: str) -> None:
        del self.config_dict[key]

    def __iter__(self):
        return iter(self.config_dict)

    def __len__(self) -> int:
        return len(self.config_dict)

    def __contains__(self, key: str) -> bool:
        return key in self.config_dict


if __name__ == "__main__":
    config = Config("/Users/fuzz/Documents/FVF/configs/config.toml")
    print(config.config_dict)
    for key in config:
        print(f"{key}: {config[key]}")
