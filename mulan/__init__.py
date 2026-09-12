import os

# Route the handful of ops without an MPS kernel (some ProstT5/T5 paths) to the CPU
# instead of erroring. Set before torch initializes MPS; setdefault respects an explicit
# override and it is a harmless no-op on CUDA/CPU machines.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# The model-side names are resolved on first access (PEP 562) rather than at import.
#
# `mulan.metrics` needs only numpy and scipy, and rescoring a shipped predictions file is the one
# workflow this package is meant to support without torch -- it is what `docs/REPRODUCE.md` and
# `results/README.md` mean by "without a GPU or a large download". Importing `.plm` eagerly pulled
# torch in, so `from mulan import metrics` required the full install to read a text file.
#
# Everything below still resolves exactly as before, just on first use: `mulan.load_pretrained`,
# `mulan.get_available_plms()`, `mulan.LightAttModel` and the rest are unchanged, and a missing
# torch surfaces at the point a model-side name is touched instead of at import.
_LAZY = {
    "plm": ("mulan.plm", None),
    "MulanConfig": ("mulan.config", "MulanConfig"),
    "LightAttModel": ("mulan.modules", "LightAttModel"),
    "load_pretrained": ("mulan.utils", "load_pretrained"),
    "load_pretrained_plm": ("mulan.utils", "load_pretrained_plm"),
    "get_available_models": ("mulan.utils", "get_available_models"),
    "get_available_plms": ("mulan.utils", "get_available_plms"),
    "get_device": ("mulan.utils", "get_device"),
    "get_plm_spec": ("mulan.plm", "get_spec"),
    "load_plm_backend": ("mulan.plm", "load_backend"),
}

__all__ = sorted(_LAZY)


def __getattr__(name):
    try:
        module_name, attr = _LAZY[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    import importlib

    module = importlib.import_module(module_name)
    value = module if attr is None else getattr(module, attr)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY))
