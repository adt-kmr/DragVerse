from .sim import SimRobot
from .unoq import UnoQRobot

ADAPTERS = {"sim": SimRobot, "unoq": UnoQRobot}


def get_robot(kind: str = "sim", **kwargs):
    if kind not in ADAPTERS:
        raise ValueError(f"unknown robot kind {kind!r}; expected one of {sorted(ADAPTERS)}")
    return ADAPTERS[kind](**kwargs)
