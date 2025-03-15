# FVF

## Say Hello!

```bash
uv run hello.py
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
uv run -m src.utils.utils_inpaint
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

## Reference

<https://github.com/XPixelGroup/BasicSR>

<https://github.com/lyuwenyu/RT-DETR>

<https://github.com/RL-VIG/LibFewShot>

<https://github.com/cszn/KAIR>
