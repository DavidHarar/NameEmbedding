import numpy as np
import pandas as pd
import langid
from tqdm import tqdm

import sys
sys.path.append("/home/david/Desktop/projects/NameEmbedding")

with open('/home/david/Desktop/projects/NameEmbedding/data/raw/text/training_names_processed_corrected.txt', "r") as file:
    training_names_txt = [line.strip() for line in file]


languages = []
for name in tqdm(training_names_txt):
    language, _ = langid.classify(name)
    languages.append(language)

names_and_languages = pd.DataFrame({'name':training_names_txt,
                                    'language':languages})
names_and_languages.to_csv('/home/david/Desktop/projects/NameEmbedding/data/processed/text/training_names_and_languages.csv',index=False)

with open('/home/david/Desktop/projects/NameEmbedding/data/raw/text/validation_names_processed.txt', "r") as file:
    validation_names_txt = [line.strip() for line in file]


languages = []
for name in tqdm(validation_names_txt):
    language, _ = langid.classify(name)
    languages.append(language)

names_and_languages = pd.DataFrame({'name':validation_names_txt,
                                    'language':languages})
names_and_languages.to_csv('/home/david/Desktop/projects/NameEmbedding/data/processed/text/validation_names_and_languages.csv',index=False)
