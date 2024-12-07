import re
import time
from collections import namedtuple
from glob import glob
from typing import Any

import torchaudio
import tqdm
from loguru import logger
from torch.utils.data import Dataset
from transformers import AutoFeatureExtractor
from torch.utils.data import random_split
import torchaudio.functional as F

feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base", local_files_only=True)


class BiteDateset(Dataset):
    FILE_PATTERN = re.compile(".*?_([01]).wav")

    def __init__(self, files_glob: str):
        ts = time.time()
        files = glob(files_glob)

        self.examples = []
        for file in tqdm.tqdm(files, desc="audio loading & preprocess"):
            m = self.FILE_PATTERN.search(file)
            if not m:
                logger.warning("file pattern miss match [{}] vs {}", file, self.FILE_PATTERN.pattern)
            else:
                waveform, sample_rate = torchaudio.load(file)
                waveform = F.resample(waveform, sample_rate, feature_extractor.sampling_rate)

                features = feature_extractor(
                    waveform, sampling_rate=feature_extractor.sampling_rate, max_length=feature_extractor.sampling_rate,
                    truncation=True
                )

                self.examples.append((features['input_values'][0][0, :], int(m.group(1)), file))
        logger.info("load {} examples in {:.2f}s", len(self.examples), time.time() - ts)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index) -> dict:
        return {"input_values": self.examples[index][0],
                "label": self.examples[index][1],
                "file": self.examples[index][2]}


if __name__ == "__main__":
    ds = BiteDateset("record/*/*.wav")
