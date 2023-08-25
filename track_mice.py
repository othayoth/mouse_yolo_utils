#!/usr/bin/python3

# depends
import os, sys
import cv2
import time
import numpy as np
import torch
import pandas as pd
import argparse

# load yolov5 model
from yolov5 import detect
weights = '/home/ash/src/yolov5/runs/train/exp19/weights/best.pt'
model = torch.hub.load('ultralytics/yolov5','custom',path=weights)



# Argument parsing
parser = argparse.ArgumentParser(description="Track mice in a cropped video.")
parser.add_argument("trial_date", help="Date of the trial (e.g., '23_08_04')")
parser.add_argument("trial_name", help="Name of the trial (e.g., 'pd1')")
args = parser.parse_args()



# trial name + video path + output csv path
trial_name = args.trial_name
trial_date = '../' + args.trial_date
vid_path = trial_date + '/' + trial_name + '/' +  trial_name + '_cropped.mp4'
output_csv_path = trial_date + '/' + trial_name + '/tracked_data.csv'

mouse_positions = []

vid_cap = cv2.VideoCapture(vid_path)
print("opened video:  "+ vid_path)

total_frames = int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))

counter = 0 



while vid_cap.isOpened():
    
    
    # read frame
    ret, frame = vid_cap.read()
    frame  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if not ret:
        break
    else:
        counter = counter + 1
    
    
    
    
    # run inference
    results = model(frame)
    infer_data = results.pandas().xyxy[0]    
    mouse_pos = infer_data[infer_data.name=='mouse']
    
    
    # append frame number and center of bounding box
    mouse_pos.insert(0, 'frame_num', counter)
    mouse_pos.insert(1, 'x_center', (mouse_pos.xmin + mouse_pos.xmax)/ 2)
    mouse_pos.insert(2, 'y_center', (mouse_pos.ymin + mouse_pos.ymax)/ 2)
    
    
    mouse_positions.append(mouse_pos)
        
        ## draw circle at centroid of bounding box
        # try:
        #     if(mouse_pos['confidence'].max()<0.5):
        #         print("no mouse")
                        
        #     else:
        #         mouse_x_center = int((mouse_pos.xmin + mouse_pos.xmax)/ 2 )
        #         mouse_y_center = int((mouse_pos.ymin + mouse_pos.ymax) / 2 )               
        #         print("mouse_detected")


        #     cv2.circle(frame,(mouse_x_center,mouse_y_center),10,(255,255,255), -1)

        # except:
        #     print("detection failed")
        #     # proximity_counter = 0


        # Display the frame
    
    # occasionally display frame and tracking info
    if(counter%100==0):        
        cv2.imshow('Frame', frame)
        print(mouse_pos)
        print(f'Processing frame {counter} / {total_frames}')    # cv2.imshow('Frame', frame)

    # press q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):        
        break
    
    
    
# write mouse positions to csv
if mouse_positions:
    all_mouse_positions = pd.concat(mouse_positions, ignore_index=True)
    all_mouse_positions.to_csv(output_csv_path, index=False)
    print(f'Mouse positions written to {output_csv_path}')


# cleanup -- close video and windows
vid_cap.release()
cv2.destroyAllWindows()