import wandb
import tomllib

with open("config/usrnet.toml", "rb") as f:
    data = tomllib.load(f)

# 1. Start a new run
run = wandb.init(project="")

# 2. Save model inputs and hyperparameters
config = run.config
config.dropout = 0.01

# 3. Log gradients and model parameters
run.watch(model)
for batch_idx, (data, target) in enumerate(train_loader):
    if batch_idx % args.log_interval == 0:
        # 4. Log metrics to visualize performance
        run.log({"loss": loss})
