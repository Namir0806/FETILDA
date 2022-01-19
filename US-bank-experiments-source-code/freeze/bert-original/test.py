from scipy import stats
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
import os
import random
import sys
import csv
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import transformers
from transformers import AutoConfig, AutoModel, AutoTokenizer 
from transformers import AdamW
from torch.cuda.amp import autocast
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import time
import tensorflow as tf

from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression