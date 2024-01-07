import torch
import torch.nn as nn
import numpy as np
import os
from tqdm import tqdm
from models.MLP import *
from models.OCNN import *


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
test_data_dir = './processed_data/test'
model_name_irm = 'mlp_10000_20240106.pth'
model_name_ocnn = 'OriginalMNISTClassifier.pth'
predicted_label_list = []
true_label_list = []


def inference_with_irm_mlp(img):
    model = MLP().to(device)
    if os.path.exists('./saved_models/' + model_name_irm):
        model.load_state_dict(torch.load('./saved_models/' + model_name_irm, map_location=device))
        model.eval()
        # load test images
        with torch.no_grad():
            img = img.to(device)
            img = img.unsqueeze(0)
            output = model(img)
            pred = output.argmax(dim=1, keepdim=True)
            return pred[0, 0].item()
    else:
        print('model not found')
        raise FileNotFoundError


def inference_with_original_cnn(img):
    # Note: this function aims to approximate the true label of the images using a pretrained CNN on the original MNIST
    # Since the pretrained CNN has 99% accuracy, it can be used to estimate the accuracy of the IRM model
    model = OCNN().to(device)
    if os.path.exists('./saved_models/' + model_name_ocnn):
        model.load_state_dict(torch.load('./saved_models/' + model_name_ocnn, map_location=device))
        model.eval()
        # find the non-zero axis - pass the original image (filtered out the spurious feature)
        # to the pretrained CNN
        for i in range(10):
            if img[i, :, :].any():
                break
        with torch.no_grad():
            img = img.to(device)
            img = img.unsqueeze(0)  # [1, 10, 28, 28]
            img = img[:, i, :, :].unsqueeze(1)  # [1, 1, 28, 28]
            output = model(img)
            pred = output.argmax(dim=1, keepdim=True)
            return pred[0, 0].item()
    else:
        print('model not found')
        raise FileNotFoundError


if os.path.isdir(test_data_dir):
    # Iterate over each file
    print('Start testing...')
    for file_index in tqdm(range(len(os.listdir(test_data_dir)))):
        file_path = os.path.join(test_data_dir, str(file_index) + '.npy')
        # Load the numpy array and convert it to a Tensor
        npy_img = np.load(file_path)
        tensor_img = torch.from_numpy(npy_img)
        predicted_label = inference_with_irm_mlp(tensor_img)
        true_label = inference_with_original_cnn(tensor_img)
        predicted_label_list.append(int(predicted_label))
        true_label_list.append(int(true_label))

else:
    raise FileNotFoundError

# Write the predicted labels to results.txt
with open('results.txt', 'w') as f:
    for i in range(len(predicted_label_list)):
        f.write('{}.npy '.format(i) + str(predicted_label_list[i]) + '\n')


# Estimate the accuracy of the IRM model
print('estimated accuracy: ' + str(np.mean(np.array(predicted_label_list) == np.array(true_label_list))))
