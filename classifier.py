import torch
import pandas as pd
import torch.cuda
import torch.version

df = pd.read_csv('ObesityDataSet.csv')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("This program is using device:", device)
