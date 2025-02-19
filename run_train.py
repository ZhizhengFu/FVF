from core.config import Config
from core.trainer import Trainer


def main():
    config = Config("")
    _ = Trainer(config)
