"""https://github.com/ThijsCol/Anylabeling-LabelMe-json-to-yolo-txt/blob/main/jsontoyolo.py"""
import os
import json
import shutil
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Define the class labels
class_labels = {"good": 0, "bad": 1, "float": 2} # Change/add more for your database

# Define the directories
input_dir = 'datasets/anylabeling/Screenshots' # Replace with your directory
output_dir = 'datasets/yolo' # Replace with your directory

# Define the train-validate split
n_val = 3 # 20% of the data will go to the validation set

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Create train and validate directories
train_dir = os.path.join(output_dir, 'train')
os.makedirs(train_dir, exist_ok=True)

validate_dir = os.path.join(output_dir, 'val')
if n_val > 0:
    os.makedirs(validate_dir, exist_ok=True)

json_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
image_files = [f for f in os.listdir(input_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

if n_val > 0:
    train_images, validate_images = train_test_split(image_files, test_size=n_val)
else:
    train_images = image_files

# Copy all images to train and validate directories
for image_file in tqdm(image_files, desc="Copying images"):
    current_output_dir = train_dir if image_file in train_images else validate_dir
    shutil.copy(os.path.join(input_dir, image_file), current_output_dir)

# Use tqdm for progress bar
for filename in tqdm(json_files, desc="Converting annotations"):
    with open(os.path.join(input_dir, filename)) as f:
        data = json.load(f)

    image_filename = filename.replace('.json', '')
    if any(os.path.isfile(os.path.join(input_dir, image_filename + ext)) for ext in ['.jpg', '.png', '.jpeg']):
        if image_filename + '.jpg' in train_images or image_filename + '.png' in train_images or image_filename + '.jpeg' in train_images:
            current_output_dir = train_dir
        else:
            current_output_dir = validate_dir

        with open(os.path.join(current_output_dir, filename.replace('.json', '.txt')), 'w') as out_file:
            for shape in data['shapes']:
                class_label = shape['label']
                if class_label in class_labels:
                    x1, y1 = shape['points'][0]
                    x2, y2 = shape['points'][1]

                    dw = 1. / data['imageWidth']
                    dh = 1. / data['imageHeight']
                    w = x2 - x1
                    h = y2 - y1
                    x = x1 + (w / 2)
                    y = y1 + (h / 2)

                    x *= dw
                    w *= dw
                    y *= dh
                    h *= dh

                    out_file.write(f"{class_labels[class_label]} {x} {y} {w} {h}\n")

print("Conversion and split completed successfully!")
