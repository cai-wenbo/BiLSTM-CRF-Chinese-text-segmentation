import torch
import torch.nn as nn
from transformers import BertTokenizer



'''
input: a string
output: a list to feed to model
'''
def TextCoder():
    def __init__(self, model_name):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)


    def __call__(self, input_text):
        encoding = self.tokenizer(input_text, add_special_tokens=False)

        text_tensor  = torch.tensor(encoding  , dtype = torch.long)

        print(text_tensor.shape)
        return text_tensor
