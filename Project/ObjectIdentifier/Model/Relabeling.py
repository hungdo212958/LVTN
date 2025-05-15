import cv2
from ultralytics import YOLO
from PIL import Image


# Load a YOLO model (you can specify a pre-trained model or a custom one)
model = YOLO('yolov8n.pt')  # Replace 'yolov8n.pt' with the desired YOLO model version

# Train the model on your dataset
model.train(
    data='Project/ObjectIdentifier/Dataset/child_detection_v3i_yolov11/data.yaml',  # Path to your dataset configuration file
    epochs=50,                         # Number of training epochs
    imgsz=640,                         # Image size for training
    batch=16,                          # Batch size
    name='children_detector',          # Name of the training run
    device=0                           # Specify GPU (0) or CPU (-1)
)


# Load an image for detection
image_path = 'Project/ObjectIdentifier/Dataset/test_images/sample.jpg'  # Replace with your image path
image = Image.open(image_path)

# Perform detection
results = model.predict(source=image_path, save=False, conf=0.25)  # Adjust confidence threshold as needed

# Convert results to OpenCV format and display
annotated_image = results[0].plot()  # Annotate the image with bounding boxes
# cv2.imshow('Detected Image', annotated_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()