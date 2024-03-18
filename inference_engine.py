import coremltools
from ultralytics import YOLO

class IE:
    def __init__(self, path, device) -> None:
        if device=='mps':
            self.model = ModelCoreML(path)
        if device=='ultralytics':
            self.model = ModelUltralytics(path)
    def forward(self, input):
        return self.model.forward(input)


class ModelOpenvino:
    def __init__(self) -> None:
        pass
    def forward(self):
        pass

class ModelUltralytics:
    def __init__(self, path) -> None:
        self.model = YOLO(path)
    def forward(self, input):
        results = self.model.predict(source=input, imgsz=[640, 640])
        ret = {'confidence':[], 'coordinates':[]}
        for box in results[0].boxes:
            conf = float(box.conf)
            xn, yn, widthn, heightn = map(float, box.xywhn[0])
            ret['coordinates'].append([xn, yn, widthn, heightn])
            ret['confidence'].append([conf])
        return ret
    

class ModelCoreML:
    def __init__(self, path) -> None:
        self.model = coremltools.models.MLModel(path)
    def forward(self, input):
        return self.model.predict({'image': input,
                                'iouThreshold': 0.8, 
                                'confidenceThreshold': 0.3})