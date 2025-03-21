from typing import Tuple, Any

import fire
from PIL import ImageGrab
from pydantic import BaseModel
from ultralytics import YOLO

# Load a model
model = YOLO("models/od/best.pt", task='detect', verbose=False)


class ScreenCapture(BaseModel):
    good: float = -1
    bad: float = -1
    miss: float = -1
    float_xyxy: Tuple[float, float, float, float] | None = None
    image: Any = None

    def float_position(self):
        x1, y1, x2, y2 = self.float_xyxy
        return int((x1 + x2) / 2), int((y1 + y2) / 2)

    def __str__(self):
        sb = ""
        if self.good != -1:
            sb += f"good[{float(self.good):.3f}] "
        if self.bad != -1:
            sb += f"bad[{float(self.good):.3f}] "
        if self.miss != -1:
            sb += f"miss[{float(self.good):.3f}] "
        return sb


def predict(path: str = None, grab: bool = False, save_as_file: str = None, conf: float = 0.2):
    image = ImageGrab.grab() if grab else path
    results = model.predict(image, conf=conf, verbose=False)[0]

    capture = ScreenCapture(image=image)
    cls = results.boxes.cls.cpu().numpy()
    confs = results.boxes.conf.cpu().numpy()
    for i in range(len(cls)):
        c = cls[i]
        conf = confs[i]
        label = results.names[c]

        match label:
            case 'good':
                capture.good = max(capture.good, conf)
            case 'bad':
                capture.bad = max(capture.bad, conf)
            case 'miss':
                capture.miss = max(capture.miss, conf)
            case _:
                capture.float_xyxy = results.boxes.xyxy[0]
    if save_as_file:
        results.save(save_as_file)
    return capture


if __name__ == '__main__':
    fire.Fire()
