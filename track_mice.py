#!/usr/bin/python3

# depends
import os, sys
import cv2
import time
import numpy as np
import torch


# load yolov5 model
from yolov5 import detect
weights = '/home/ash/src/yolov5/runs/train/exp18/weights/best.pt'
model = torch.hub.load('ultralytics/yolov5','custom',path=weights)

# trial name + video path
trial_name = 'ctrl2'
vid_path = trial_name + '/Cam0.mp4'
vid_cap = cv2.VideoCapture(vid_path)

counter = 0 
while vid_cap.isOpened():
    
    
    # read frame
    ret, frame = vid_cap.read()
    if not ret:
        break
    else:
        counter = counter + 1
    
    if(counter%10==0):
    
        # run inference
        results = model(frame)
        infer_data = results.pandas().xyxy[0]
        # print(infer_data)    
        mouse_pos = infer_data[infer_data.name=='mice']
        # print(mouse_pos)
        
        try:
            if(mouse_pos['confidence'].max()<0.6):
                print("no mouse")
                        
            else:
                mouse_x_center = int((mouse_pos.xmin + mouse_pos.xmax)/ 2 )
                mouse_y_center = int((mouse_pos.ymin + mouse_pos.ymax) / 2 )
                self.mouse_pose.x = mouse_x_center
                self.mouse_pose.y = mouse_y_center
                self.pub_mouse2d.publish(self.mouse_pose)
                print("mouse_detected")


            cv2.circle(frame,(mouse_x_center,mouse_y_center),10,(255,255,255), -1)

        except:
            print("detection failed")
            # proximity_counter = 0


        # Display the frame
        cv2.imshow('Frame', frame)

    # press q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):        
        break


# cleanup -- close video and windows
vid_cap.release()
cv2.destroyAllWindows()