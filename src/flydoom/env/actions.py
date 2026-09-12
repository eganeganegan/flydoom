"""Small, reproducible action vocabulary shared by mock and VizDoom environments."""

from enum import IntEnum


class DoomAction(IntEnum):
    NO_OP = 0
    MOVE_FORWARD = 1
    TURN_LEFT = 2
    TURN_RIGHT = 3
    SHOOT = 4


ACTION_NAMES = tuple(action.name.lower() for action in DoomAction)


def action_count() -> int:
    return len(DoomAction)
