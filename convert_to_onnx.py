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





# print(model)

# import torch.onnx 

# #Function to Convert to ONNX 
# def Convert_ONNX(): 

#     # set the model to inference mode 
#     model.eval() 

#     # Let's create a dummy input tensor  
#     dummy_input = torch.randn(1, input_size, requires_grad=True)  

#     # Export the model   
#     torch.onnx.export(model,         # model being run 
#          dummy_input,       # model input (or a tuple for multiple inputs) 
#          "ImageClassifier.onnx",       # where to save the model  
#          export_params=True,  # store the trained parameter weights inside the model file 
#          opset_version=10,    # the ONNX version to export the model to 
#          do_constant_folding=True,  # whether to execute constant folding for optimization 
#          input_names = ['modelInput'],   # the model's input names 
#          output_names = ['modelOutput'], # the model's output names 
#          dynamic_axes={'modelInput' : {0 : 'batch_size'},    # variable length axes 
#                                 'modelOutput' : {0 : 'batch_size'}}) 
#     print(" ") 
#     print('Model has been converted to ONNX')

# if __name__ == "__main__": 

#     # Let's build our model 
#     #train(5) 
#     #print('Finished Training') 

#     # Test which classes performed well 
#     #testAccuracy() 

#     # Let's load the model we just created and test the accuracy per label 
#     model = Network() 
#     path = "myFirstModel.pth" 
#     model.load_state_dict(torch.load(path)) 

#     # Test with batch of images 
#     #testBatch() 
#     # Test how the classes performed 
#     #testClassess() 
 
#     # Conversion to ONNX 
#     Convert_ONNX()    