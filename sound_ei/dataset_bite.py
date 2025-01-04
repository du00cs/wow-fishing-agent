import random
import re
import time
from glob import glob

import torchaudio
import torchaudio.functional as F
import tqdm
from loguru import logger
from torch.utils.data import Dataset
from transformers import AutoFeatureExtractor

feature_extractor = AutoFeatureExtractor.from_pretrained("./wav2vec2-base")


class BiteDatesetN(Dataset):
    # datasets/record/miss-bite/1734187957_16_miss_5.ogg 表示 miss 的音频，正样本手工标记在第5秒（第5秒之前，不含，为正样本）
    FILE_PATTERN = re.compile(
        r"datasets/record/(?P<scene>.*?)/.*?_(?P<seconds>\d+)(_(?P<bg>bg))?.ogg")

    def __init__(self, files_glob: str, window: int = 3):
        ts = time.time()
        files = glob(files_glob)
        logger.info("files {}", len(files))

        self.examples = []
        for file in tqdm.tqdm(files, desc="audio loading & preprocess"):
            file = file.replace("\\", "/")
            m = self.FILE_PATTERN.search(file)
            if not m:
                logger.warning("file pattern miss match [{}] vs {}", file, self.FILE_PATTERN.pattern)
            else:
                seconds = int(m.group("seconds"))
                all_zeros = file.endswith("_bg.ogg")

                if seconds < window + 2:
                    logger.info("drop auduio < {}", window + 2)
                    continue

                waveform, _sr = torchaudio.load(file)
                sr = feature_extractor.sampling_rate
                if _sr != sr:
                    waveform = F.resample(waveform, _sr, sr)

                if all_zeros:
                    indices = [i for i in range(0, seconds - window - 2, window)]
                else:
                    negatives = [i for i in range(seconds - window - 2 + 1)]
                    k = len(negatives) if len(negatives) < 2 else 2
                    indices = random.choices([i for i in range(seconds - window - 2 + 1)], k=k)
                    indices.append(seconds - window)

                for i in indices:
                    wave_n = waveform[:, i * sr: (i + window) * sr]
                    features = feature_extractor(wave_n, sampling_rate=sr, max_length=sr * window, truncation=True)
                    label = 0 if (all_zeros or (i != seconds - window)) else 1
                    self.examples.append((features['input_values'][0][0, :], label, f"{file}@[{i}, {i + window})"))

        logger.info("load {} examples in {:.2f}s", len(self.examples), time.time() - ts)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index) -> dict:
        return {"input_values": self.examples[index][0],
                "label": self.examples[index][1],
                "file": self.examples[index][2]}


if __name__ == "__main__":
    ds = BiteDatesetN("datasets/record/*/*.ogg", 3)
    from collections import Counter

    print(Counter([ex['label'] for ex in ds]))

