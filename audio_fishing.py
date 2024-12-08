import random
import time
from multiprocessing import Process, Queue
from queue import Empty
from time import sleep

from loguru import logger
from pynput.mouse import Controller as Mouse

from fishing import detect_splashing, BiteSuite, AudioChunk, SuiteSaveOption
from keyboard_mouse import MouseButton, mouse_action


def main(mouse_start: MouseButton = MouseButton.middle, mouse_end: MouseButton = MouseButton.scroll_down,
         save_suite: SuiteSaveOption = SuiteSaveOption.none):
    # 接收声音识别序列
    bite_queue = Queue()

    # 持续水花检测
    splashing = Process(
        target=detect_splashing, args=(bite_queue,), name="detect splashing"
    )
    splashing.start()

    mouse = Mouse()

    while True:
        suite = BiteSuite(scope_captures=[], audio_chunks=[])
        # 甩杆
        mouse_action(mouse, mouse_start, desc="施放钓鱼技能")

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
            if event["ts"] < start + 1:  # 丢弃早于开始监听的事件
                continue
            suite.audio_chunks.append(AudioChunk(label=event["label"], chunk=event["audio"]))
            if event["label"] == "bite":
                logger.info("检测到事件 {} , 收竿", event["label"], enqueue=True)
                mouse_action(mouse, mouse_end, desc="互动（收竿）")
                caught = True
        if not caught:
            logger.warning("未检测到水声")
        if save_suite == SuiteSaveOption.all or \
                save_suite == SuiteSaveOption.nok and not caught:
            suite.save(full=True if save_suite == SuiteSaveOption.all else False)

        sleep(random.random() * 2 + 1)

    splashing.terminate()
    exit(0)


if __name__ == "__main__":
    import typer

    typer.run(main)
