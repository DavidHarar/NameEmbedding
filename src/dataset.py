from torch.utils.data import Dataset
from tqdm import tqdm
import torch
    
class CustomDataset(Dataset):
    def __init__(self, text_list, tokenizer, max_len, include_attention_mask=True):
        """
        Custom Dataset for tokenized text.
        
        Args:
            text_list (list): List of input text strings.
            tokenizer (PreTrainedTokenizer): Hugging Face tokenizer instance.
            max_len (int): Maximum sequence length.
            include_attention_mask (bool): Whether to include attention masks.
        """
        self.examples = []
        self.include_attention_mask = include_attention_mask
        self.tokenizer = tokenizer

        # Preprocess the input text
        print("Preprocessing text into tokenized examples...")
        for example in tqdm(text_list, desc='Tokenizing text'):
            encoded = tokenizer(
                example,
                max_length=max_len,
                truncation=True,
                padding="max_length",
                return_attention_mask=include_attention_mask
            )
            self.examples.append(encoded)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        """
        Retrieve tokenized data at the specified index.
        Returns:
            Dictionary containing input IDs, attention masks (if applicable).
        """
        item = {"input_ids": torch.tensor(self.examples[index]["input_ids"], dtype=torch.long)}
        if self.include_attention_mask:
            item["attention_mask"] = torch.tensor(self.examples[index]["attention_mask"], dtype=torch.long)
        return item
