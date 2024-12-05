import os
import random
import time
from multiprocessing import Process, Queue
from queue import Empty
from time import sleep
from typing import Any, Literal, Optional, List

import fire
import torchaudio
from loguru import logger
from pydantic import BaseModel
from pynput.mouse import Button, Controller

import od_predict
from od_predict import ScreenCapture
from sound_ei.infer import stream as audio_infer_stream
from sound_ei.loopback import default_device


class AudioChunk(BaseModel):
    label: str
    chunk: Any


class BiteSuite(BaseModel):
    scope_captures: List[ScreenCapture]  # 有效范围检测
    audio_chunks: Optional[List[AudioChunk]] = None  # 声音检测
    result_capture: Optional[ScreenCapture] = None  # 结果校验 (对应点击过早的情况)

    def save(
            self,
            full: bool = False,
            base_path: str = "datasets/record",
            sample_rate: int = default_device.sample_rate,
    ):
        if full:
            path = f"{base_path}/suites/{int(time.time())}"
            os.makedirs(path, exist_ok=True)
            with open(f"{path}/od.txt", "w") as f:
                if self.scope_captures:
                    for idx, capture in enumerate(self.scope_captures):
                        f.write(f"scope[{idx}]: {capture}\n")
                        capture.image.save(f"{path}/scope_{idx}.jpg")

                if self.audio_chunks:
                    for idx, audio_chunk in enumerate(self.audio_chunks):
                        torchaudio.save(
                            f"{path}/{idx}_{audio_chunk.label}.wav",
                            audio_chunk.chunk,
                            sample_rate,
                        )
                if self.result_capture:
                    f.write(f"result: {self.result_capture}\n")
                    self.result_capture.image.save(f"{path}/result.jpg")
        elif self.result_capture is None:  # miss bite
            if self.audio_chunks:
                _path = f"{base_path}/miss-bite/{int(time.time())}"
                logger.warning("save miss bite wav to {}", _path)
                for idx, audio_chunk in enumerate(self.audio_chunks):
                    torchaudio.save(
                        f"{_path}_{idx}_1.wav",
                        audio_chunk.chunk,
                        sample_rate,
                    )
        elif self.result_capture.miss > 0.5:  # wrong bite
            if self.audio_chunks:
                _path = f"{base_path}/wrong-bite/{int(time.time())}_0.wav"
                logger.info("save wrong bite wav: {}", _path)
                torchaudio.save(_path, self.audio_chunks[-1].chunk, sample_rate)


def random_wait(mean: float, error: float):
    return sleep((random.random() - 0.5) * error + mean)


def effective_scope(
        suite: BiteSuite, mouse: Controller, retry: int = 20, valid_conf: float = 0.5
) -> Optional[ScreenCapture]:
    """下竿，检测鱼漂和提示信息"""
    capture = None
    for i in range(retry):
        mouse.press(Button.middle)
        random_wait(0.1, 0.02)
        mouse.release(Button.middle)
        random_wait(1.5, 0.2)
        capture = od_predict.predict(grab=True)
        suite.scope_captures.append(capture)
        logger.info("施放钓鱼技能，检测交互范围 [{}]: {}", i + 1, capture)
        if capture.good > valid_conf:  # or capture.float_xyxy:
            return capture
        else:
            random_wait(1.0, 0.1)

    if capture is not None:
        capture.image.save("detect_error.jpg")
    return None


def detect_splashing(queue: Queue):
    for t, label, audio in audio_infer_stream():
        queue.put({"ts": t, "label": label, "audio": audio})


def main(
        valid_conf: float = 0.6,
        cast_retry: int = 20,
        save_suite: Literal["none", "nok", "all"] = "none",
):
    # 接收声音识别序列
    bite_queue = Queue()

    # 持续水花检测
    splashing = Process(
        target=detect_splashing, args=(bite_queue,), name="detect splashing"
    )
    splashing.start()

    mouse = Controller()

    while True:
        suite = BiteSuite(scope_captures=[], audio_chunks=[])
        # 甩杆至“有效交互范围”
        capture = effective_scope(suite, mouse, retry=cast_retry)
        if capture is None:
            suite.save()
            logger.error("未检测到有效范围标识，退出", enqueue=True)
            break

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
                if capture.good > valid_conf:
                    mouse.scroll(0, -2)
                caught = True
        if caught:  # 截图，检测是在有鱼上钩，如果没有则准备一个待分析用例
            random_wait(0.9, 0.1)
            capture = od_predict.predict(grab=True)
            suite.result_capture = capture
            logger.info("bite check: {}", capture)

        if save_suite == "all" or \
                save_suite == "nok" and (suite.result_capture is None or suite.result_capture.miss > 0.5):
            suite.save(full=True if save_suite == "all" else False)

        sleep(random.random() * 2 + 1)

    splashing.terminate()
    exit(0)

if __name__ == "__main__":
    fire.Fire(main)
