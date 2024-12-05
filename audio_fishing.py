import random
import time
from multiprocessing import Process, Queue
from queue import Empty
from time import sleep
from typing import Literal

import fire
from loguru import logger
from pynput.mouse import Button, Controller as Mouse

from fishing import detect_splashing, BiteSuite, random_wait, AudioChunk


def mouse_action(mouse: Mouse, button: Button):
    match button:
        case Button.left | Button.middle | Button.right:
            mouse.press(button)
            random_wait(0.1, 0.02)
            mouse.release(button)
        case Button.scroll_up:
            mouse.scroll(0, 2)
        case Button.scroll_down:
            mouse.scroll(0, -2)
        case _:
            logger.warning("unknown mouse button: {}", button)


LIMITED_MOUSE_ACTIONS = Literal[Button.left, Button.middle, Button.right, Button.scroll_up, Button.scroll_down]


def main(save_suite: Literal["none", "nok", "all"] = "nok",
         mouse_start: LIMITED_MOUSE_ACTIONS = Button.middle,
         mouse_end: LIMITED_MOUSE_ACTIONS = Button.scroll_down):
    # 接收声音识别序列
    bite_queue = Queue()

    # 持续水花检测
    splashing = Process(
        target=detect_splashing, args=(bite_queue,), name="detect splashing"
    )
    splashing.start()

    mouse = Mouse()

    while True:
        suite = BiteSuite(audio_chunks=[])
        # 甩杆
        mouse_action(mouse, mouse_start)

        # 倾听水花的声音
        start = time.time()
        logger.info("倾听水花的声音", enqueue=True)
        caught = False
        while time.time() - start < 15 and not caught:
            try:
                event = bite_queue.get(block=False)
            except Empty as e:
                time.sleep(0.1)
                continue
            if event["ts"] < start:  # 丢弃早于开始监听的事件
                continue
            suite.audio_chunks.append(AudioChunk(label=event["label"], chunk=event["audio"]))
            if event["label"] == "bite":
                logger.info("检测到事件 {} , 收竿", event["label"], enqueue=True)
                mouse_action(mouse, mouse_end)
                caught = True

        if save_suite == "all" or \
                save_suite == "nok" and not caught:
            suite.save(full=True if save_suite == "all" else False)

        sleep(random.random() * 2 + 1)

    splashing.terminate()
    exit(0)


if __name__ == "__main__":
    fire.Fire(main)
