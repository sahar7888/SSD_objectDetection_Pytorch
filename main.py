
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




"""load video"""
# Create a VideoCapture object and read from input file
cap = cv2.VideoCapture('cars2.mp4')
# Check if camera opened successfully
if (cap.isOpened() == False):
  print("Error opening video file")


  # Read until video is completed
while (cap.isOpened()):

  # Capture frame-by-frame
  ret, frame = cap.read()
  if ret == True:


    # Display the resulting frame

    img = frame.copy()
    # print(type(img))
    # convert it to a tensor
    imgtranform = T.ToTensor()
    image = imgtranform(frame)
    # print(type(image))
    with torch.no_grad():
      ypred = model([image])
      # print(ypred[0].keys())
      bbox, scores, labels = ypred[0]['boxes'], ypred[0]['scores'], ypred[0]['labels']
      nums = torch.argwhere(scores > 0.80).shape[0]
      for i in range(nums):
        x, y, w, h = bbox[i].numpy().astype('int')
        cv2.rectangle(img, (x, y), (w, h), (0, 0, 255), 3)
        classname = labels[i].numpy().astype('int') - 1
        classdetected = classnames[classname]
        print(classdetected)
        cvzone.putTextRect(img, classdetected, [x, y - 20], scale=2, border=2)

    cv2.imshow('DFrame', img)
    # cv2.imshow('Frame', frame)

    # Press Q on keyboard to exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break

  # Break the loop
  else:
    break