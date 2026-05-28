__all__ = [
    "TimesFM",
    "TTM",
    "TTMR3",
]


def __getattr__(name: str):
    if name == "TimesFM":
        from .timesfm import TimesFM

        return TimesFM
    if name == "TTM":
        from .ttm import TTM

        return TTM
    if name == "TTMR3":
        from .ttm_r3 import TTMR3

        return TTMR3
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
