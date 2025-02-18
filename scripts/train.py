import wandb
import tomllib
from core.models.backbone.DepthwiseDenoising import DepthwiseDenoisingBlock

with open("config/usrnet.toml", "rb") as f:
    data = tomllib.load(f)

run = wandb.init(project="")

model = DepthwiseDenoisingBlock()

run.watch(model)
