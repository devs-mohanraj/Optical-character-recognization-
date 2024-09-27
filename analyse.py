import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np


class NumberCNN(nn.Module):
    def __init__(self):
        super(NumberCNN , self).__init__()
        self.con1 = nn.Conv2d(1,32,kernel_size=3)
        self.con2 = nn.Conv2d(32 , 64 , kernel_size= 3)
        self.fc1 = nn.Linear(64*5*5 , 128)
        self.fc2 = nn.Linear(128 , 10)
        
    def forward(self  , x):
        x = F.relu(self.con1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.con2(x))
        x = F.max_pool2d(x , 2)
        x = x.view(x.size(0) , -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x           
    
#the vision part

model = NumberCNN()
model.load_state_dict(torch.load('./models/numbers_model_path.pth' , weights_only=True))  
model.eval()  


cap = cv2.VideoCapture(0)  

while True:
    ret, frame = cap.read()  
    if not ret:
        print("Failed to grab frame.")
        break
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray_image, (28, 28))  
    normalized = resized / 255.0  
    tensor_image = torch.tensor(normalized).float().unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, 28, 28)

    with torch.no_grad():  

        prediction = model(tensor_image)
        print(type(prediction))  # Should output <class 'torch.Tensor'>

    digit = torch.argmax(prediction).item()
    print(f"Predicted Digit: {digit}")
    
    cv2.putText(frame, f'Prediction: {digit}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow('Digit Recognition', frame)  

    if cv2.waitKey(1) & 0xFF == ord('space'):  
        break

cap.release()  
cv2.destroyAllWindows()  
