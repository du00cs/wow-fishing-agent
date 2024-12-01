import glob
import os
import time

import fire
import numpy as np
import pyaudio
import torch
import torchaudio
import torchaudio.functional as F
from transformers import AutoModelForAudioClassification

from sound_ei import loopback
from sound_ei.dataset_bite import feature_extractor
from sound_ei.loopback import loopback_stream, AudioDevice
from loguru import logger
from sound_ei.dataset_bite import BiteDateset


def get_best_checkpoint(path: str = "models/bite_model"):
    n = 1000000
    model = None
    for ckpt in glob.glob(f"{path}/checkpoint-*"):
        no = int(ckpt.split("-")[-1])
        if no < n:
            n = no
            model = ckpt
    if not model:
        raise Exception("model checkpoint not found")
    logger.info("get best checkpoint: {}", model)
    return model


def infer(
    model,
    wave: np.ndarray | torch.Tensor,
    sample_rate: int = feature_extractor.sampling_rate,
) -> torch.Tensor:
    """数据需要是channel first"""
    if type(wave) == np.ndarray:
        wave = torch.from_numpy(wave)
    if len(wave.shape) == 2:
        wave = wave[0, :]
    if sample_rate != feature_extractor.sampling_rate:
        wave = F.resample(wave, sample_rate, feature_extractor.sampling_rate)
    inputs = feature_extractor(
        wave, sampling_rate=feature_extractor.sampling_rate, return_tensors="pt"
    )

    logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=1)
    return probs


def load_model(ckpt: str):
    start = time.time()
    model = AutoModelForAudioClassification.from_pretrained(ckpt)
    logger.info("load model in {}", time.time() - start)
    return model


def stream(
    device: AudioDevice = loopback.default_device,
    ckpt: str = get_best_checkpoint(),
    window_seconds: float = 1.0,
):
    with loopback_stream(device=device, seconds=window_seconds) as stream:
        chunk = int(window_seconds * device.sample_rate)

        model = load_model(ckpt)

        with torch.no_grad():
            while True:
                data = torch.from_numpy(
                    np.frombuffer(stream.read(chunk), dtype=np.float32).reshape((-1, 2))
                ).reshape((2, -1))
                t = time.time()
                probs = infer(model, data, device.sample_rate)
                pred = "bite" if probs.numpy()[0, 1] > 0.5 else "other"
                yield t, pred, data


def bite_as_negative(
    save_bite_limit: int = 5,
    save_path: str = f"datasets/record/negative_{int(time.time())}",
):
    device: AudioDevice = loopback.default_device
    i = 0
    for t, label, data in stream(device):
        if label == "bite":
            os.makedirs(save_path, exist_ok=True)
            torchaudio.save(f"{save_path}/{i}_0.wav", data, device.sample_rate)
            i += 1
            if i >= save_bite_limit:
                break


def eval_dataset(
    model_path: str = get_best_checkpoint(),
    folder_glob: str = "datasets/record/unchecked/*/*.wav",
):
    ds = BiteDateset(folder_glob)
    model = load_model(model_path)
    ok, nok = 0, 0
    for one in ds:
        probs = infer(model, one["input_values"])
        pred = 1 if probs.detach().numpy()[0, 1] > 0.5 else 0
        if one["label"] != pred:
            logger.info("error {}({}) => {}", one["label"], pred, one["file"])
            nok += 1
        else:
            ok += 1
    logger.info("accuracy: {}/({}+{}) = {}", ok, ok, nok, ok / (ok + nok))


if __name__ == "__main__":
    fire.Fire()
