import torch
import torch.nn as nn
from transformers import BertTokenizer



'''
input: a string
output: a list to feed to model
'''
class TextCoder():
    def __init__(self, model_used):
        self.tokenizer = BertTokenizer.from_pretrained(model_used)


    def __call__(self, input_text):
        encoding = self.tokenizer(input_text, add_special_tokens=False)

        text_tensor  = torch.tensor(encoding['input_ids']  , dtype = torch.long)

        return text_tensor
