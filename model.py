import torch
import torch.nn as nn
import numpy as np
import config

# Model Architecture Parameters
CONV_FILTERS = [32, 64, 64]
CONV_KERNELS = [8, 4, 3]
CONV_STRIDES = [4, 2, 1]
FC_UNITS = 512

def create_dqn(action_size):
    conv1 = nn.Conv2d(config.STACK_FRAMES, CONV_FILTERS[0],
                     kernel_size=CONV_KERNELS[0], stride=CONV_STRIDES[0])
    conv2 = nn.Conv2d(CONV_FILTERS[0], CONV_FILTERS[1], 
                     kernel_size=CONV_KERNELS[1], stride=CONV_STRIDES[1])
    conv3 = nn.Conv2d(CONV_FILTERS[1], CONV_FILTERS[2], 
                     kernel_size=CONV_KERNELS[2], stride=CONV_STRIDES[2])
    
    # Calculate conv output size
    test_input = torch.zeros(1, config.STACK_FRAMES, config.IMG_HEIGHT, config.IMG_WIDTH)
    conv_out = conv3(conv2(conv1(test_input)))
    conv_out_size = int(np.prod(conv_out.size()))
    
    fc1 = nn.Linear(conv_out_size, FC_UNITS)
    fc2 = nn.Linear(FC_UNITS, action_size)
    
    layers = [conv1, nn.ReLU(), conv2, nn.ReLU(), conv3, nn.ReLU(), 
              nn.Flatten(), fc1, nn.ReLU(), fc2]
    model = nn.Sequential(*layers)
    
    return model.to(config.DEVICE)

def dqn_forward(model, x):
    x = x.float() / 255.0
    return model(x)