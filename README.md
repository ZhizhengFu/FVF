# FVF

```text
FVF/
├── configs/
│   └── model_config.toml
├── core/
│   ├── config/
│   │   └── config.py
│   ├── data/
│   │   ├── datasets.py
│   │   ├── transforms.py
│   │   └── dataloader.py
│   ├── models/
│   │   ├── factory.py
│   │   └── custom_model.py
│   ├── engine/
│   │   ├── trainer.py
│   │   └── callbacks.py
│   ├── utils/
│   │   ├── logger.py
│   │   ├── wandb_integration.py
│   │   └── checkpoint.py
│   └── __init__.py
├── scripts/
│   ├── train.py
│   └── inference.py
├── experiments/
└── README.md
```

## torch settings

In some cases, you may want to use CPU-only builds in some cases, but CUDA-enabled builds in others, with the choice toggled by a user-provided extra.

```bash
uv sync --extra cpu
```
```bash
uv sync --extra cu124
```

---
how to run the code:
```bash
uv run hello.py
```

how to format:
```bash
uv run ruff format
```
```bash
uv run ruff check
```

how to update package:
```bash
uv lock --upgrade
```

## TODO list:

- [x] Basic file structure
- [x] Toml config file design
- [x] wandb settings
- [x] uv environment
  - [x] torch / torch vision
- [x] ruff
- [x] init seed
- [ ] logger
- [ ] save model
- [ ] init wandb
- [ ] multi-GPU
- [ ] multi_model support
  - [ ] config file design

## Reference

<https://github.com/RL-VIG/LibFewShot>

<https://github.com/cszn/KAIR>
