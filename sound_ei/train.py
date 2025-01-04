import random

import evaluate
import numpy as np
from loguru import logger
from torch.utils.data import random_split, Subset
from transformers import AutoModelForAudioClassification, TrainingArguments, Trainer, EarlyStoppingCallback

from sound_ei.dataset_bite import BiteDatesetN

accuracy = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=eval_pred.label_ids)

model = AutoModelForAudioClassification.from_pretrained(
    # "facebook/wav2vec2-base",
    "./wav2vec2-base", # 先下下来，免得好提示一些有的没的
    num_labels=2,
    label2id={'other': 0, 'bite': 1},
    id2label={0: 'other', 1: 'bite'},
)

training_args = TrainingArguments(
    output_dir="models/bite_model",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=20,
    warmup_ratio=0.1,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    save_total_limit=2,
)

ds = BiteDatesetN("datasets/record/*/*.ogg", 3)

random_idx = [i for i in range(len(ds))]

random.shuffle(random_idx)

n_valid = 100

train_idx = random_idx[:-n_valid]
random.shuffle(train_idx)

ds_train = Subset(ds, train_idx)
ds_val = Subset(ds, random_idx[-n_valid:])
logger.info("train: {}, val: {}", len(ds_train), len(ds_val))

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds_train,
    eval_dataset=ds_val,
    compute_metrics=compute_metrics,
)

trainer.train()
