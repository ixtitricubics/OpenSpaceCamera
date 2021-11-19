import cv2

file_name ="cam192.168.1.101_2021_11_17_15_40_34.avi"
cap = cv2.VideoCapture(file_name)
counter = 0
while(True):
    ret, frame = cap.read()
    if(not ret):
        break
    counter +=1
print(counter)