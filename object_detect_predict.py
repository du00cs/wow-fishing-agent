from typing import Tuple, Any

from PIL import ImageGrab
from pydantic import BaseModel
from ultralytics import YOLO

# Load a model
model = YOLO("runs/detect/best.pt", task='detect')


class ScreenCapture(BaseModel):
    good: float = -1
    bad: float = -1
    float_xyxy: Tuple[float, float, float, float] | None = None
    image: Any = None

    def float_position(self):
        x1, y1, x2, y2 = self.float_xyxy
        return int((x1 + x2) / 2), int((y1 + y2) / 2)


def predict(path: str = None, grab: bool = False, show: bool = False, conf: float = 0.2):
    image = ImageGrab.grab() if grab else path
    results = model.predict(image, conf=conf)[0]

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
            case _:
                capture.float_xyxy = results.boxes.xyxy[0]
    if show:
        results.show()
    return capture


if __name__ == '__main__':
    print(predict('detect_error.jpg', show=True))
