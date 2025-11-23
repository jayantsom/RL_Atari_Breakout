import numpy as np
import torch
import torch.nn.functional as F
import cv2

def preprocess_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)
    return frame

def huber_loss(pred, target):
    return F.smooth_l1_loss(pred, target)

def get_epsilon(frame_idx, config):
    epsilon = config.EPSILON_END + (config.EPSILON_START - config.EPSILON_END) * \
              np.exp(-1. * frame_idx / config.EPSILON_DECAY)
    return max(epsilon, config.EPSILON_END)