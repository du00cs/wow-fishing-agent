from enum import Enum
from functools import partial
from typing import Optional, List

from pynput.mouse import Button, Controller as Mouse
from loguru import logger
import time
import random
from pynput import keyboard


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


def keyboard_listener(key: keyboard.Key | keyboard.KeyCode, status: List[bool]):
    def on_press(_key: keyboard.Key | keyboard.KeyCode, _status: List[bool]):
        if _key == key:
            _status[0] = not _status[0]
            if _status[0]:
                logger.info("pressed {}: resume", key)
            else:
                logger.info("pressed {}: paused", key)

    listener = keyboard.Listener(on_press=partial(on_press, _status=status))
    listener.start()

