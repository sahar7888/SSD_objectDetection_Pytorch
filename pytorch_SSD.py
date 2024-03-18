"""Ref:https://github.com/Tech-Watt/YOUTUBE-TUTORIAL-CODES """


import torch
import torchvision
from torchvision import transforms as T
import cv2
import cvzone

"""Create the model"""
model = torchvision.models.detection.ssd300_vgg16(weights=True)

model.eval()
"""class names"""
classnames = []
with open('classes.txt', 'r') as f:
    classnames = f.read().splitlines()

# print(classnames[0])
"""Read the image"""
image = cv2.imread('dog1.jpg') # it is a numpy array
img = image.copy()
# convert it to a tensor
imgtranform = T.ToTensor()
image = imgtranform(image)
# print(type(image))

"""Make a prediction using the model"""
with torch.no_grad():
    ypred = model([image])
    print(ypred[0].keys())

    bbox,scores,labels = ypred[0]['boxes'], ypred[0]['scores'], ypred[0]['labels']
    nums = torch.argwhere(scores > 0.80).shape[0]
    print(nums)
    for i in range(nums):
        x,y,w,h = bbox[i].numpy().astype('int')
        cv2.rectangle(img, (x,y), (w,h), (0,0,255),5)
        classname = (labels[i].numpy().astype('int'))-1
        classdetected = classnames[classname-1]
        cvzone.putTextRect(img,classdetected,[x,y-10], scale=2, border=2)




cv2.imshow('frame', img)
cv2.waitKey(0)




