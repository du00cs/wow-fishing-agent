from ultralytics import YOLO

# Load a model
model = YOLO("yolo11s.pt", task='detect')

# Train the model
train_results = model.train(
    data="datasets/yolo/fish.yaml",  # path to dataset YAML
    epochs=100,  # number of training epochs
    device="0",  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
    workers=0,
)

# Evaluate model performance on the validation set
metrics = model.val()
