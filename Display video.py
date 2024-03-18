# importing libraries
import cv2

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
    cv2.imshow('Frame', frame)
    print(type(frame))

    # Press Q on keyboard to exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break

  # Break the loop
  else:
    break
