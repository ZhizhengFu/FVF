from core.config import Config


class Trainer:
    def __init__(self, config: Config):
        self.config = config

    def run_loop(self):
        print("Running loop")
