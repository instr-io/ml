"""Inference package."""

__all__ = ["VocalRemover"]


def __getattr__(name: str):
    if name == "VocalRemover":
        from .infer import VocalRemover

        return VocalRemover
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
