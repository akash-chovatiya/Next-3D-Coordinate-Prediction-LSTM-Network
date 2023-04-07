# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 13:40:13 2022

@author: Chovatiya
"""

# from Object_dimension import coordinates
import numpy as np
import torch
import torch.nn as nn
from lib_LSTM.model import LSTM
import csv
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# ref_path = r"C:\Users\Chovatiya\sciebo\01_Gitlab_FHSWF\Thesis\odtp-resnet50-end\ref_image_80.jpg"
# vid_path = r"C:\Users\Chovatiya\AR_DATASET\step_4_Action Dataset Generation\Action video_original\pushing.mp4"
# _, loc_3d = coordinates(ref_path, vid_path)
# loc_3d = loc_3d[4].numpy()

percentage = 0.1
hidden_layers = 64
input_size = 3
output_size = 3
test_data_size = 10*7
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# loc_3d = np.array([[368.67510986, 292.02072144,  72.89762115],
#         [367.59777832, 282.29138184,  73.67412567],
#         [364.82244873, 266.52920532,  69.22610474],
#         [352.64648438, 250.10177612,  63.22583389],
#         [337.25469971, 242.3853302 ,  58.57093811],
#         [320.6892395 , 245.54548645,  52.10147476],
#         [342.37133789, 259.50018311,  43.53656769]])

# augment_dataset = np.empty((100,7,3), np.float32)
# for i in range(100):
#     idx = i*percentage
#     loc = loc_3d*((100-(idx))/100)
#     augment_dataset[i:i+1,:,:] = loc

# with open('augmented_dataset.csv', 'w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow(["X","Y","Z"])
#     for i in augment_dataset:
#         for k in range(7):
#             writer.writerow([i[k][0], i[k][1], i[k][2]])

data = pd.read_csv('augmented_dataset.csv')
train_data = data[:-test_data_size]
test_data = data[-test_data_size:]
train_data = train_data.values
test_data = test_data.values

# scaler= MinMaxScaler(feature_range=(-1,1))
# train_data = scaler.fit_transform(train_data.reshape(-1,3))

train_data = torch.FloatTensor(train_data).view(-1,3)
test_data = torch.FloatTensor(test_data).view(-1,3)

def create_inout_sequences(input_data, one_data_size):
    j = 0
    inout_seq = []
    L = len(input_data)
    for i in range(int(L/one_data_size)):
        train_seq = input_data[j:j+one_data_size-1]
        train_label = input_data[j+one_data_size-1:j+one_data_size]
        inout_seq.append((train_seq, train_label))
        j += 7
    return inout_seq

train_data = create_inout_sequences(train_data, 7)
test_data = create_inout_sequences(test_data, 7)

model = LSTM(input_size, hidden_layers, output_size)
model.to(DEVICE)
criterion = nn.MSELoss()
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(params, lr=0.008)
epochs = 1000

for i in range(epochs):
    loss_epoch = []
    for seq, labels in train_data:
        optimizer.zero_grad()
        out = model(seq.to(DEVICE))
        loss = criterion(out, labels.view(-1).to(DEVICE))
        loss_epoch.append(loss.item())
        loss.backward()
        optimizer.step()
    loss = (sum(loss_epoch)/len(loss_epoch))
    if i%10 == 1:
        print("epoch: {} loss: {}".format(i, loss))
        # print(f'epoch: {i:3} loss: {loss.item():10.8f}')
        
    
    with torch.no_grad():
        test_loss_epoch = []
        for seq, labels in test_data:
            pred = model(seq.to(DEVICE))
            loss = criterion(pred, labels.view(-1).to(DEVICE))
            test_loss_epoch.append(loss.item())
        
        test_loss = (sum(test_loss_epoch)/len(test_loss_epoch))    
        print("epoch: {} loss: {}".format(i, test_loss))
        