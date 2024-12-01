"""采集数据"""

import os
import time
from multiprocessing import Queue

import fire
import numpy as np
import pyaudiowpatch as pyaudio
import torch
import torchaudio
from loguru import logger
from pynput import mouse

from sound_ei import loopback
import shutil

from sound_ei.loopback import AudioDevice


class MouseEvent:
    queue: Queue

    def __init__(self, queue: Queue):
        self.queue = queue

    def on_click(self, x, y, button, pressed):
        if pressed:
            match button:
                case mouse.Button.middle:
                    s = f"start|{int(time.time())}"
                    self.queue.put(s)
                    logger.info("Pressed {}: {}", button, s)
                case mouse.Button.right:
                    logger.info("Pressed {}: stop", button)
                    self.queue.put("stop")

    def on_scroll(self, x, y, dx, dy):
        if dy < 0:  # scroll down
            logger.info("Scrolled down: stop")
            self.queue.put("stop")


def examples(device: AudioDevice = loopback.default_device, seconds=1.0, channels=2):
    chunk = int(seconds * device.sample_rate)

    queue = Queue()

    event = MouseEvent(queue)

    # listen in thread
    listener = mouse.Listener(on_click=event.on_click, on_scroll=event.on_scroll)
    listener.start()

    with loopback.loopback_stream(device=device, seconds=seconds) as stream:
        i = -1
        logger.info("recording ...")
        while True:
            wave = (
                np.frombuffer(stream.read(chunk), dtype=np.float32)
                .reshape((-1, 2))
                .transpose()
            )
            try:
                instruct = queue.get(block=False)
            except:
                instruct = ""
            if instruct.startswith("start|"):
                folder = "datasets/record/unchecked/" + instruct[len("start|") :]
                os.makedirs(folder, exist_ok=True)
                i = 0

            if i >= 0:
                if i <= 3 and instruct == "stop":
                    logger.info("skip this too short case")
                    shutil.rmtree(folder)
                else:
                    label = 1 if instruct == "stop" else 0
                    filename = f"{folder}/{i:02d}_{label}.wav"
                    torchaudio.save(
                        uri=filename,
                        src=torch.from_numpy(wave),
                        sample_rate=device.sample_rate,
                    )

            i = -1 if instruct == "stop" or i == -1 else i + 1

            if i == 17:
                i = -1
                logger.warning("instruct timeout")


if __name__ == "__main__":
    fire.Fire(examples)
