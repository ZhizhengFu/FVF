# FVF

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
ruff format
```

## TODO list:

- [x] Basic file structure
- [x] Toml config file design
- [x] wandb settings
- [x] uv environment
  - [x] torch / torch vision
- [x] ruff
- [ ] init seed
- [ ] save model
- [ ] init wandb
- [ ] multi-GPU
- [ ] multi_model support
  - [ ] config file design
