#Loading Libraries

import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
import numpy as np

class Classification:

    def __init__(self,class_names= [],model_path = ''):
        self.class_names = class_names
        self.transform = transforms.Compose([transforms.ToPILImage(),transforms.Resize(256),transforms.CenterCrop(224),
                    transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.model = self.load_model(model_path,len(class_names))

    def load_model(self,model_path, num_classes):
        model = models.resnet18(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        model.load_state_dict(torch.load(model_path, map_location=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")))
        model.eval()
        return model
    
    def predict(self,image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image).unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            outputs = self.model(image)
            _, preds = torch.max(outputs, 1)
        predicted_label = self.class_names[preds[0].item()]
        return predicted_label
    

if __name__ == "__main__":
    Model_path = 'pdi_resnet18_06082024.pt'
    Class_names = ['flipped', 'normal','partial','two_sticker']
    classfication = Classification(class_names = Class_names,model_path = Model_path)
    # input image
    img = cv2.imread('test_images/1.jpg')
    # Predict the label
    predicted_label = classfication.predict(img)
    print(f'Predicted Label: {predicted_label}')