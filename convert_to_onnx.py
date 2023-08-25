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
weights = 'weights/best.pt'
model = torch.hub.load('ultralytics/yolov5','custom',path=weights)

output_model_path = 'exp18_best.onnx'
# from torchsummary import summary
# summary(model, input_size=(3, 640, 640))  # Change input_size as needed

model.eval()  # Set the model to evaluation mode

try:
    dummy_input = torch.randn(1, 3, 640, 640)  # Change the size as needed
    output = model(dummy_input)
    # Convert and export the model to ONNX format
    torch.onnx.export(model, dummy_input, output_model_path, verbose=True)

    print(f"Model successfully converted and saved to {output_model_path}")
except RuntimeError as e:
    print(e)

