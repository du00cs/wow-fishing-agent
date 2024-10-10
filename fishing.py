import random
import time
from datetime import datetime
from multiprocessing import Process, Queue
from queue import Empty
from time import sleep

import numpy as np
import pyaudio
import sounddevice as sd
from loguru import logger
from pynput.mouse import Button, Controller

import object_detect_predict
from keras_yamnet import params
from keras_yamnet.preprocessing import preprocess_input
from keras_yamnet.yamnet import YAMNet, class_names
from object_detect_predict import ScreenCapture


def detect_splashing(queue: Queue):
    ################### SETTINGS ###################
    FORMAT = pyaudio.paFloat32
    CHANNELS = 1
    RATE = params.SAMPLE_RATE
    WIN_SIZE_SEC = 0.975
    CHUNK = int(WIN_SIZE_SEC * RATE)

    logger.info(sd.query_devices())
    MIC = None

    #################### MODEL #####################

    model = YAMNet(weights='keras_yamnet/yamnet.h5')
    yamnet_classes = class_names('keras_yamnet/yamnet_class_map.csv')

    #################### STREAM ####################
    audio = pyaudio.PyAudio()

    # start Recording
    stream = audio.open(format=FORMAT,
                        input_device_index=MIC,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)
    logger.info("recording...", enqueue=True)

    last = None
    while True:
        # Waveform
        data = preprocess_input(np.fromstring(
            stream.read(CHUNK), dtype=np.float32), RATE)
        prediction = model.predict(np.expand_dims(data, 0), verbose=0)[0]

        event = yamnet_classes[np.argmax(prediction)]
        if event != 'Silence' and event != last:
            # ['Rowboat, canoe, kayak', 'Boat, Water vehicle']
            logger.info("sound event: {}", event, enqueue=True)
            queue.put({"ts": datetime.now().timestamp(), "label": event})
        last = event

    logger.info("finished recording", enqueue=True)

    # stop Recording
    stream.stop_stream()
    stream.close()
    audio.terminate()


def effective_scope(mouse, i: int) -> ScreenCapture | None:
    """下竿，检测鱼漂和提示信息"""
    mouse.press(Button.left)
    sleep(0.1)
    mouse.release(Button.left)

    sleep(0.5)
    try:
        logger.info("施放钓鱼技能，检测交互范围 x {}", i, enqueue=True)
        sleep(1)
        return object_detect_predict.predict(grab=True)
    except:
        return None


if __name__ == '__main__':
    queue = Queue()

    # 持续水花检测
    splashing = Process(target=detect_splashing, args=(queue,), name="detect splashing")
    splashing.start()

    mouse = Controller()

    good_bad_conf = .6

    while True:
        # fishing
        for i in range(10):
            capture = effective_scope(mouse, i)
            logger.info("图像检测: {}", capture)
            if capture.good > good_bad_conf:  # or capture.float_xyxy:
                break
        if not capture.good > good_bad_conf:  # and not capture.float_xyxy:
            logger.error("图像检测出错，退出", enqueue=True)
            capture.image.save("detect_error.jpg")
            break

        start = time.time()
        logger.info("倾听水花的声音", enqueue=True)
        caught = False
        while datetime.now().timestamp() - start < 15 and not caught:
            try:
                event = queue.get(block=False)
            except Empty as e:
                time.sleep(0.1)
                continue
            if event['ts'] < start:  # filter history event
                continue
            elif event['label'] in ['Rowboat, canoe, kayak', 'Boat, Water vehicle', 'Explosion', 'Splash, splatter',
                                    'Crack', 'Crunch', 'Water']:
                logger.info("检测到事件 {} , 收竿", event['label'], enqueue=True)
                if capture.good > good_bad_conf:
                    mouse.scroll(0, -2)
                elif capture.float_xyxy:
                    mouse.move()
                    mouse.click(Button.right)
                caught = True
        if not caught:
            logger.info("未检测到事件, 超时收竿", enqueue=True)
        sleep(random.random() * 2 + 1)

    splashing.terminate()
    exit(0)
