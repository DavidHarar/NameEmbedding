import os
import sys

# os.chdir("/home/david/Desktop/projects/NameEmbedding")
sys.path.append("/home/david/Desktop/projects/NameEmbedding")

from src.experiment import *
from src.train import general_config, tokenizer_config, model_config, train_config


# small version of the model for debugging purposes:
general_config = {
    'model_type':'RoBERTa',
    'experiments_folder':'./experiments',
    # 'training_set_path':'./data/raw/text/training_names_processed_corrected.txt',
    # 'validation_set_path':'./data/raw/text/validation_names_processed.txt',
    'training_set_path':'./data/processed/text/training_names_and_languages_resampled.csv',
    'validation_set_path':'./data/processed/text/validation_names_and_languages.csv',
}
train_config['train_epochs'] = 2


# run
pipeline(general_config, tokenizer_config, model_config, train_config, n = 1000)

