#!/usr/bin/python3

import os, sys
import cv2
import argparse

# Argument parsing
parser = argparse.ArgumentParser(description="Process a video by cropping and resizing.")
parser.add_argument("trial_name", help="Name of the trial (e.g., 'pd1')")
parser.add_argument("--only_display", action="store_true", help="Only display the video without writing to a file")
args = parser.parse_args()
trial_name = args.trial_name
only_display = args.only_display

# Load the video
video_path = trial_name + '/Cam0.mp4'
cap = cv2.VideoCapture(video_path)

# Specify the cropping coordinates (x, y, width, height) for the ROI
crop_x = 800
crop_y = 0
crop_width = 2200
crop_height = 2200

# Specify the desired output frame size
output_width = 640
output_height = 640

# Create an output video writer
if not only_display:
    output_path = trial_name + '/'   + trial_name + '_cropped.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (output_width, output_height))


counter = 0
while cap.isOpened():
    counter = counter + 1
    ret, frame = cap.read()
    if not ret:
        break
    else:
        counter = counter+1
    
    
    if(counter%10==0):
        print("Frame: " + str(counter))
    
    # Crop the frame to the specified ROI
    cropped_frame = frame[crop_y:crop_y + crop_height, crop_x:crop_x + crop_width]
    
    # Resize the cropped frame to the desired output size
    resized_frame = cv2.resize(cropped_frame, (output_width, output_height))
    
    # # Write the resized frame to the output video
    if not only_display:
        out.write(resized_frame)
    else:
    # Display the resized frame (optional)
        cv2.imshow('Resized Frame', resized_frame)
    
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release video capture and writer
cap.release()
if(not only_display):
    out.release()
else:
    cv2.destroyAllWindows()
