# FVF

## Say Hello!

```bash
uv run hello.py
```

---

```text
FVF/
├── configs/
│   └── model_name.toml
├── src/
│   ├── config/
│   │   └── config.py
│   ├── data/
│   │   ├── datasets.py
│   │   └── ......
│   ├── models/
│   │   ├── backbone/
│   │   ├── model_name.py
│   │   └── ......
│   ├── engine/
│   │   ├── trainer.py
│   │   └── ......
│   ├── utils/
│   │   ├── logger.py
│   │   ├── utils.py
│   │   └── ......
│   └── __init__.py
├── datasets/
├── experiments/
├── train.py
├── test.py
└── README.md
```

## env sync / script

In some cases, you may want to use CPU-only builds in some cases, but CUDA-enabled builds in others, with the choice toggled by a user-provided extra.

```bash
uv sync --extra cpu
```
```bash
uv sync --extra cu124
```

---
run the code:
```bash
uv run hello.py
```

code format:
```bash
uv run ruff format
```
```bash
uv run ruff check
```

package:
```bash
uv add xxx
uv remove xxx
uv lock --upgrade
```

## TODO list:

- [x] Basic file structure
- [x] Toml config file design
- [x] wandb settings
- [x] uv environment
  - [x] torch / torch vision
- [x] ruff
- [x] init seed / wandb
- [ ] logger
- [ ] save model
- [ ] multi-GPU
- [ ] multi_model support
  - [ ] config file design

## Reference

<https://github.com/XPixelGroup/BasicSR>

<https://github.com/lyuwenyu/RT-DETR>

<https://github.com/RL-VIG/LibFewShot>

<https://github.com/cszn/KAIR>
