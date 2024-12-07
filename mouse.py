from enum import Enum
from typing import Optional

from pynput.mouse import Button, Controller as Mouse
from loguru import logger
import time
import random


class MouseButton(str, Enum):
    left = "left"
    middle = "middle"
    right = "right"
    scroll_up = "scroll_up"
    scroll_down = "scroll_down"


def mouse_action(mouse: Mouse, button: MouseButton, desc: Optional[str] = None):
    if desc:
        logger.info("mouse [{}]: {}", button.value, desc)
    match button:
        case MouseButton.left | MouseButton.middle | MouseButton.right:
            btn = Button[button.value]
            mouse.press(btn)
            random_wait(0.1, 0.02)
            mouse.release(btn)
        case MouseButton.scroll_up:
            mouse.scroll(0, 2)
        case MouseButton.scroll_down:
            mouse.scroll(0, -2)
        case _:
            logger.warning("unknown mouse button: {}", button)


def random_wait(mean: float, error: float):
    return time.sleep((random.random() - 0.5) * error + mean)
