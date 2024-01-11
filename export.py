from ultralytics import YOLO

# Load a model
model = YOLO('../YOLO-NAS/best.pt')  # load a custom trained model

# Export the model
model.export(format='coreml', imgsz=[224, 384], int8=True, nms=True)