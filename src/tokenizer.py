# import torch
# import transformers

# from tokenizers import Tokenizer
# from tokenizers.models import BPE
# from tokenizers.trainers import BpeTrainer
# from transformers import RobertaConfig, RobertaForMaskedLM, RobertaTokenizer, DebertaTokenizer, BertTokenizer
# from tokenizers import ByteLevelBPETokenizer

# # from datasets import load_metric, Dataset
# from torch.utils.data import Dataset
# from tqdm import tqdm



from tokenizers import ByteLevelBPETokenizer
from transformers import BertTokenizer, RobertaTokenizer, DebertaTokenizer

class BPE_Based_Tokenizer():
    def __init__(self, config):
        self.tokenizer      = ByteLevelBPETokenizer()
        self.vocab_size     = config.get("vocab_size")
        self.max_len        = config.get("max_len")
        self.min_frequency  = config.get("min_frequency")
        self.saving_path    = config.get("saving_path")
        self.model_type     = config.get("model_type")
        self.tokenizer_location = config.get("tokenizer_location")

        assert self.model_type in ['BERT', 'RoBERTa', 'DeBERTa'], "Model type must be in ['BERT', 'RoBERTa', 'DeBERTa']!"


    def train_and_save(self, training_names):
        self.tokenizer.train_from_iterator(
            training_names,
            vocab_size=self.vocab_size, min_frequency=self.min_frequency,
            show_progress=True,
            special_tokens=[
                "<s>",
                "<pad>",
                "</s>",
                "<unk>",
                "<mask>",
                ])
        print("Saving tokenizer to:", self.tokenizer_location)
        self.tokenizer.save_model(self.tokenizer_location)

    def load_and_wrap_tokenizer(self):
        """
        A function that loads the tokenizer and also expand its functionality according to a model.
        This functionality allows it to do additional things, like `encode_plus`.
        """
        if self.model_type == 'BERT':
            self.tokenizer = BertTokenizer(vocab_file = self.tokenizer_location + '/vocab.json',
                                              merges_file= self.tokenizer_location + '/merges.txt')
            
        if self.model_type == 'RoBERTa':
            self.tokenizer = RobertaTokenizer(vocab_file = self.tokenizer_location + '/vocab.json',
                                              merges_file= self.tokenizer_location + '/merges.txt')
        
        if self.model_type == 'DeBERTa':
            self.tokenizer = DebertaTokenizer(vocab_file = self.tokenizer_location + '/vocab.json',
                                              merges_file= self.tokenizer_location + '/merges.txt')
        else:
            print("Didn't load any tokenizer")

    def encode_plus(self,x):
        return self.tokenizer.encode_plus(x,
                      max_length            = self.max_len,
                      # truncation=True,
                      add_special_tokens    = True,
                      pad_to_max_length     = True,
                      return_attention_mask = True,
                      return_tensors='pt')
