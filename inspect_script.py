import torch
import torch.nn as nn
from src.model import BiLSTM_CRF
from src.utils import TextCoder
from transformers import BertTokenizer
import os
import argparse


def insert_char(string, char, loc):
    return string[:loc] + char + string[loc:]


def segment_text(test_config):
    model_name = "bert-base-chinese"
    text_coder = TextCoder(model_name)



    '''
    load model
    '''
    model = BiLSTM_CRF(
            vocab_size       = test_config['vocab_size'],
            embedding_dim    = test_config['embedding_dim'],
            LSTM_hidden_size = test_config['LSTM_hidden_size'],
            LSTM_num_layers  = test_config['LSTM_num_layers'],
            num_labels       = test_config['num_labels'],
            sequence_length  = test_config['sequence_length']
            )

    if os.path.exists(test_config['model_path_src']):
        model_dict = torch.load(test_config['model_path_src'])
        model.load_state_dict(model_dict)

    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')


    while True:
        src = input("enter the src word:\n")
        text_list = tokenizer.tokenize(src)
        text_tensor = text_coder(src).unsqueeze(0)
        s_predicts = model(
                batched_text = text_tensor
                ).squeeze(0).tolist()
        for i in range(len(s_predicts)):
            loc = len(s_predicts) - i - 1
            if (s_predicts[loc] == 0) or (s_predicts[loc] == 1) or (s_predicts[loc] == 4):
                text_list = insert_char(text_list, ['|'], loc)
        print(''.join(text_list))
                                                




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_size"       , type=int   , help="vocab size"                                        , default=21128)
    parser.add_argument("--embedding_dim"    , type=int   , help="embedding dimmention"                              , default=512)
    parser.add_argument("--LSTM_hidden_size" , type=int   , help="hidden_size of the BiLSTM model"                   , default=256)
    parser.add_argument("--LSTM_num_layers"  , type=int   , help="num_layers of the BiLSTM model"                    , default=1)
    parser.add_argument("--num_labels"       , type=int   , help="types of labels"                                   , default=6)
    parser.add_argument("--sequence_length"  , type=int   , help="sequence_length"                                   , default=128)
    parser.add_argument("--model_path_src"   , type=str   , help="the directory to load model"                       , default='./saved_models/saved_dict.pth')

    
    args = parser.parse_args()

    test_config = dict()
    for arg in vars(args):
        test_config[arg] = getattr(args, arg)

    segment_text(test_config)
