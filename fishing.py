import datetime
import os
import time
from enum import Enum
from typing import Optional, List, Any

import torchaudio
from loguru import logger
from pydantic import BaseModel
from pynput import keyboard
from pynput.mouse import Controller as MouseController

import od_predict
from keyboard_mouse import random_wait, mouse_action, MouseButton, keyboard_listener
from od_predict import ScreenCapture
from sound_ei.infer import stream
from sound_ei.loopback import default_device


class BiteSuite(BaseModel):
    scope_captures: List[ScreenCapture]  # 有效范围检测
    audio_chunks: Optional[Any] = None  # 16秒的 audio tensor
    result_capture: Optional[ScreenCapture] = None  # 结果校验 (对应点击过早的情况)

    def save(
            self,
            full: bool = False,
            base_path: str = "datasets/record",
            sample_rate: int = default_device.sample_rate,
    ):
        ts = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
        if full:
            path = f"{base_path}/suites/{int(time.time())}"
            os.makedirs(path, exist_ok=True)
            with open(f"{path}/od.txt", "w") as f:
                if self.scope_captures:
                    for idx, capture in enumerate(self.scope_captures):
                        f.write(f"scope[{idx}]: {capture}\n")
                        capture.image.save(f"{path}/scope_{idx}.jpg")

                if self.audio_chunks:
                    torchaudio.save(f"{path}/{idx}_{n}.ogg", self.audio_chunks, sample_rate, )
                if self.result_capture:
                    f.write(f"result: {self.result_capture}\n")
                    self.result_capture.image.save(f"{path}/result.jpg")
        elif self.result_capture is None:  # miss bite
            if self.audio_chunks:
                _path = f"datasets/miss-bite/{ts}"
                logger.warning("save miss bite wav to {}", _path)
                torchaudio.save(f"{_path}_16_miss.ogg", self.audio_chunks, sample_rate)
        elif self.result_capture.miss > 0.5:  # wrong bite
            if self.audio_chunks:
                ts = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
                _path = f"{base_path}/wrong-bite/{ts}_zero.ogg"
                logger.info("save wrong bite wav: {}", _path)
                torchaudio.save(_path, self.audio_chunks, sample_rate)


class SuiteSaveOption(str, Enum):
    none = "none"
    nok = "nok"
    all = "all"


def effective_scope(
        suite: BiteSuite, mouse: MouseController, retry: int = 20, valid_conf: float = 0.5,
        mouse_start: MouseButton = MouseButton.middle
) -> Optional[ScreenCapture]:
    """下竿，检测鱼漂和提示信息"""
    capture = None
    for i in range(retry):
        # 甩杆
        mouse_action(mouse, mouse_start, desc="🆕 cast fishing")
        random_wait(1.5, 0.2)
        capture = od_predict.predict(grab=True)
        suite.scope_captures.append(capture)
        logger.info("🎣 cast fishing, detect valid scope [{}]: {}", i + 1, capture)
        if capture.good > valid_conf:  # or capture.float_xyxy:
            return capture
        else:
            random_wait(0.5, 0.1)

    if capture is not None:
        capture.image.save("detect_error.jpg")
    return None


def task(mouse: MouseController, cast_retry: int, valid_conf: float, mouse_start: MouseButton, mouse_end: MouseButton,
         listen_seconds: int, window: int,
         save_suite: SuiteSaveOption):
    suite = BiteSuite(scope_captures=[], audio_chunks=[])
    # 甩杆至“有效交互范围”
    capture = effective_scope(suite, mouse, retry=cast_retry, mouse_start=mouse_start, valid_conf=valid_conf)
    if capture is None:
        suite.save()
        logger.error("🔴 no valid scope detected", enqueue=True)
        return 'pause'

    logger.info("🎤 listen for water splash", enqueue=True)

    pred, audio_tensor = stream(window=window, maxlen=listen_seconds)

    caught = True if pred == "bite" else False
    suite.audio_chunks = audio_tensor

    if pred == "bite":
        logger.info("🐟 event bite detected, finish", pred, enqueue=True)
        mouse_action(mouse, mouse_end, desc="interaction click")
        # 截图，检测是在有鱼上钩，如果没有则准备一个待分析用例
        random_wait(0.9, 0.1)
        capture = od_predict.predict(grab=True)
        suite.result_capture = capture
        if capture.miss > 0.5:
            logger.warning("🐟❌ bite check: wrong bite (too early!)")
            caught = False
    else:
        logger.warning("🐟❌ no splash detected")

    if save_suite == SuiteSaveOption.all or \
            save_suite == SuiteSaveOption.nok and not caught:
        suite.save(full=True if save_suite == SuiteSaveOption.all else False)
    return 'finish'


def main(
        valid_conf: float = 0.6,
        cast_retry: int = 20,
        window: int = 3,
        listen_seconds: int = 16,
        save_suite: SuiteSaveOption = SuiteSaveOption.nok,
        mouse_start: MouseButton = MouseButton.middle, mouse_end: MouseButton = MouseButton.scroll_down,
        pause_key: Optional[str] = 'f12'
):
    # 暂停检测
    status = [False]
    if pause_key:
        keyboard_listener(keyboard.Key[pause_key], status)
    logger.warning("⏸️ pause: wait for trigger")

    mouse = MouseController()

    while True:
        if status[0]:
            result = task(mouse, cast_retry=cast_retry, mouse_start=mouse_start, mouse_end=mouse_end,  # 鼠标
                          valid_conf=valid_conf,  # 图像
                          listen_seconds=listen_seconds, window=window,  # 声音
                          save_suite=save_suite)
            if result == 'pause':
                status[0] = False
            random_wait(2.0, 1.0)
        else:
            random_wait(1.0, 0.1)

    exit(0)


if __name__ == "__main__":
    import typer

    typer.run(main)
