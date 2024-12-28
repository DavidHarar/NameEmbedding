
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
pd.options.mode.chained_assignment = None
import warnings
warnings.filterwarnings("ignore")

training_data = pd.read_csv('/home/david/Desktop/projects/NameEmbedding/data/processed/text/training_names_and_languages.csv')
validation_data = pd.read_csv('/home/david/Desktop/projects/NameEmbedding/data/processed/text/validation_names_and_languages.csv')



def get_ascii_range(word):
    ascii_values = [ord(char) for char in word if ord(char)]
    if ascii_values:
        return min(ascii_values), max(ascii_values), np.mean(ascii_values)
    else:
        return None

# training
training_data['asci_min']=0
training_data['asci_max']=0
training_data['asci_mean']=0
for j in tqdm(range(len(training_data))):
    word = training_data['name'][j]
    min_,max_,mean_ = get_ascii_range(word)
    training_data['asci_min'][j] = min_
    training_data['asci_max'][j] = max_
    training_data['asci_mean'][j] = mean_
training_data.to_csv('/home/david/Desktop/projects/NameEmbedding/data/processed/text/training_names_languages_and_ascii.csv', index=False)

# validation
validation_data['asci_min']=0
validation_data['asci_max']=0
validation_data['asci_mean']=0
for j in tqdm(range(len(validation_data))):
    word = validation_data['name'][j]
    min_,max_,mean_ = get_ascii_range(str(word))
    validation_data['asci_min'][j] = min_
    validation_data['asci_max'][j] = max_
    validation_data['asci_mean'][j] = mean_
validation_data.to_csv('/home/david/Desktop/projects/NameEmbedding/data/processed/text/validation_names_languages_and_ascii.csv', index=False)
