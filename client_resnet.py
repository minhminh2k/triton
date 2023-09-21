import numpy as np
from torchvision import transforms
from PIL import Image
import tritonclient.http as httpclient
from tritonclient.utils import triton_to_np_dtype
import torch
import cv2
import albumentations as A
from albumentations import Compose
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import Dataset

import os
import glob
import tritonclient.grpc as grpcclient


# Setting up client HTTP and GRPC
# client = httpclient.InferenceServerClient(url="localhost:8000")
client = grpcclient.InferenceServerClient(url="localhost:8001")


# Preprocess Image
def preprocess_image(img_path):
    img = Image.open(img_path).convert("RGB")
    
    preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
           
        ])
    return preprocess(img).numpy()

# Image folder
folder_path = "./input/"
output_path = "./output/"

# List of images
image_files = [os.path.basename(file) for file in glob.glob(os.path.join(folder_path, "*.jpg"))]
# print(image_files)


# Batch Size
batch_size = 4

# Preprocess all images
list_data = []
for image_path in image_files:
    transformed_img = preprocess_image(f"{folder_path}{image_path}")
    list_data.append(transformed_img)
    
list_data = np.array(list_data)
    
count = 0

'''
HTTP Protocol
inputs = httpclient.InferInput("input__0", transformed_img.shape, datatype="FP32")
inputs.set_data_from_numpy(transformed_img, binary_data=True)
outputs = httpclient.InferRequestedOutput("output__0", binary_data=True)#, class_count=1000)
'''

for i in range(0, len(list_data), batch_size):
    batch_images = list_data[i:i+batch_size]
    # print(batch_images.shape)
    
    # GRPC
    inputs = grpcclient.InferInput("input__0", batch_images.shape, datatype="FP32")
    inputs.set_data_from_numpy(batch_images)
    outputs = grpcclient.InferRequestedOutput("output__0")
        
    # Querying the server
    results = client.infer(model_name="resnet34", inputs=[inputs])

    inference_output = results.as_numpy('output__0')

    inference_output = inference_output * 255

    for output in inference_output:
        # print(output_image.shape)
        print(f"{image_files[count]}:", output)
       
        count += 1