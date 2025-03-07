import tomllib
import warnings
from pathlib import Path
from typing import Any, Self


class Config(dict):
    def __init__(self, data: dict[str, Any] | None = None):
        data = data or {}
        super().__init__(self._convert_data(data))

    def _convert_data(self, data: dict[str, Any]) -> dict[str, Any]:
        return {
            key: Config(value)
            if isinstance(value, dict)
            else None
            if value == ""
            else value
            for key, value in data.items()
        }

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            warnings.warn(
                f"Key '{name}' is missing in the configuration. Returning None.",
                UserWarning,
                stacklevel=2,
            )
            return None

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = Config(value) if isinstance(value, dict) else value

    @classmethod
    def from_toml(cls, config_path: str | Path) -> Self:
        path = Path(config_path)
        with path.open("rb") as f:
            data = tomllib.load(f)
        print(f"Configuration loaded from {path.resolve()}")
        return cls(data)

    def __repr__(self) -> str:
        return f"Config({super().__repr__()})"


if __name__ == "__main__":
    config = Config.from_toml("configs/usrnet.toml")
    print(config.model.name)
    print(config.model.pretrained_path)  # None
    print(config.model.pretrained_pat)  # Warning + None
