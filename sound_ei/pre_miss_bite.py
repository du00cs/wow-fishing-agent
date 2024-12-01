import fire
from pydantic import BaseModel
from sound_ei.infer import infer
from sound_ei.dataset_bite import BiteDateset
from sound_ei.infer import get_best_checkpoint, load_model
from loguru import logger
import re

FILE_PATTERN = re.compile(r".*?(?P<g>\d+)_(?P<no>\d+)_(?P<y>\d).wav")

class PredCase(BaseModel):
    case_id: int
    seq_id: int
    label: int
    pred: float

def eval_dataset(
    model_path: str = get_best_checkpoint(),
    folder_glob: str = "datasets/record/miss-bite/*.wav",
):
    ds = BiteDateset(folder_glob)
    model = load_model(model_path)
    ok, nok = 0, 0
    cases = []
    for one in ds:
        probs = infer(model, one["input_values"])
        if m := FILE_PATTERN.fullmatch(one["file"]):
            c = PredCase(case_id=int(m.group("g")), seq_id=int(m.group("no")), label=int(m.group("y")), pred=probs.detach().numpy()[0, 1])
            cases.append(c)
    
    case_group = {}
    for c in cases:
        if c.case_id not in case_group:
            case_group[c.case_id] = []
        case_group[c.case_id].append(c)
    
    _list = [(id, cs) for id, cs in case_group.items()]
    _list = sorted(_list, key=lambda x:x[0])
    
    for id, cs in _list:
        p = -1
        k = -1
        for i in range(len(cs)):
            if cs[i].pred > p:
                p = cs[i].pred
                k = i
        logger.info("{}: prob {}: {}", id, cs[k].pred, cs[k].seq_id)
    


if __name__ == "__main__":
    fire.Fire()