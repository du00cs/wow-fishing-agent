import re

import fire
import tqdm
from loguru import logger
from pydantic import BaseModel

from sound_ei.dataset_bite import BiteDateset
from sound_ei.infer import get_best_checkpoint, load_model
from sound_ei.infer import infer
import shutil

FILE_PATTERN = re.compile(r".*?(?P<g>\d+)_(?P<no>\d+)_(?P<y>\d).wav")


class PredCase(BaseModel):
    case_id: int
    seq_id: int
    label: int
    pred: float
    file: str


def eval_dataset(model_path: str = get_best_checkpoint(), folder_glob: str = "datasets/record/miss-bite/*.wav",
                 checkout_highest_prob: bool = True):
    ds = BiteDateset(folder_glob)
    model = load_model(model_path)
    cases = []
    for one in tqdm.tqdm(ds, "pred"):
        probs = infer(model, one["input_values"])
        if m := FILE_PATTERN.fullmatch(one["file"]):
            c = PredCase(case_id=int(m.group("g")), seq_id=int(m.group("no")), label=int(m.group("y")),
                         pred=probs.detach().numpy()[0, 1], file=one["file"])
            cases.append(c)

    case_group = {}
    for c in cases:
        if c.case_id not in case_group:
            case_group[c.case_id] = []
        case_group[c.case_id].append(c)

    _list = [(id, cs) for id, cs in case_group.items()]
    _list = sorted(_list, key=lambda x: x[0])

    for id, cs in _list:
        p = -1
        k = -1
        for i in range(len(cs)):
            if cs[i].pred > p:
                p = cs[i].pred
                k = i
        logger.info("{}: prob {:.3f}: {}", id, cs[k].pred, cs[k].seq_id)
        if checkout_highest_prob:
            shutil.copy(cs[k].file, f"datasets/record/unchecked/{cs[k].case_id}_{cs[k].seq_id}_1.wav")


if __name__ == "__main__":
    fire.Fire(eval_dataset)
