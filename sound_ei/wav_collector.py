"""采集数据"""
import datetime
import os
import time
from typing import List

import fire
import numpy as np
import torch
import torchaudio
from loguru import logger
from pynput import mouse

from sound_ei import loopback
from sound_ei.loopback import AudioDevice


class MouseEvent:
    start: mouse.Button
    stop: mouse.Button
    status: List[str]

    def __init__(self, status: List[str],
                 start: mouse.Button = mouse.Button.middle, stop: mouse.Button = mouse.Button.right):
        self.stop = stop
        self.start = start
        self.status = status

    def on_click(self, x, y, button, pressed):
        if pressed:
            match button:
                case self.start:
                    self.status[0] = f'start|{int(time.time())}'
                    logger.info("Pressed {}: start", button)
                case self.stop:
                    self.status[0] = 'stop'
                    logger.info("Pressed {}: stop", button)

    def on_scroll(self, x, y, dx, dy):
        if dy < 0:  # scroll down
            self.status[0] = 'stop'
            logger.info("Scrolled down: stop")


def background(scene: str, seconds: int = 60, device: AudioDevice = loopback.default_device, channels=2):
    chunk = int(seconds * device.sample_rate)

    with loopback.loopback_stream(device=device, chunk_seconds=seconds) as stream:
        logger.info("recording ...")

        wave = np.frombuffer(stream.read(chunk), dtype=np.float32).reshape((-1, channels)).transpose()  # 2 x frame

        ts = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
        base_dir = f"datasets/record/{scene}"
        os.makedirs(base_dir, exist_ok=True)
        filename = f"{base_dir}/{ts}_{seconds}_bg.ogg"
        torchaudio.save(
            uri=filename,
            src=torch.from_numpy(wave),
            sample_rate=device.sample_rate,
        )
        logger.info("save background example {}", filename)


def manual(scene: str, device: AudioDevice = loopback.default_device, seconds=1.0, channels=2):
    chunk = int(seconds * device.sample_rate)

    status = ['stop']
    event = MouseEvent(status)

    # listen in thread
    listener = mouse.Listener(on_click=event.on_click, on_scroll=event.on_scroll)
    listener.start()

    with loopback.loopback_stream(device=device, chunk_seconds=seconds) as stream:
        waves = []
        logger.info("recording ...")
        last_start = None
        while True:
            wave = np.frombuffer(stream.read(chunk), dtype=np.float32).reshape((-1, channels)).transpose()  # 2 x frame
            if status[0].startswith('start'):
                if last_start != status[0]:
                    logger.info("restart a new record")
                    last_start = status[0]
                    waves = [wave]
                elif len(waves) == 0:
                    waves = [wave]
                else:
                    waves.append(wave)
            elif waves:
                if len(waves) <= 17:
                    merged = np.concat(waves, axis=1)
                    ts = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
                    base_dir = f"datasets/record/{scene}"
                    os.makedirs(base_dir, exist_ok=True)
                    filename = f"{base_dir}/{ts}_{len(waves)}.ogg"
                    torchaudio.save(
                        uri=filename,
                        src=torch.from_numpy(merged),
                        sample_rate=device.sample_rate,
                    )
                    logger.info("save example {}", filename)
                else:
                    logger.warning("instruct timeout")
                waves = []


if __name__ == "__main__":
    fire.Fire()
