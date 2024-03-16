from ultralytics import YOLO

# Load a model
model = YOLO('models/blue_feb_3X++.pt')  # load a custom trained model

# Export the model
model.export(format='coreml', imgsz=[640, 640], int8=False, nms=True)