import torch
import torch.nn as nn
from src.model import BiLSTM_CRF
from src.utils import TextCoder
import os
import argparse



def segment_text(model_parameter):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 


    model_name = "bert_base_Chinese"
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

    model = model.to(device)


    while True:
        src = input("enter the src word:\n")
        text_tensor = text_coder(src)
        b_predicts = model(
                batched_text = text_tensor
                )
        print(b_predicts.shape)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_size"       , type=int   , help="vocab size"                                        , default=2979)
    parser.add_argument("--embedding_dim"    , type=int   , help="embedding dimmention"                              , default=512)
    parser.add_argument("--LSTM_hidden_size" , type=int   , help="hidden_size of the BiLSTM model"                   , default=256)
    parser.add_argument("--LSTM_num_layers"  , type=int   , help="num_layers of the BiLSTM model"                    , default=256)
    parser.add_argument("--num_labels"       , type=int   , help="types of labels"                                   , default=6)
    parser.add_argument("--sequence_length"  , type=int   , help="sequence_length"                                   , default=128)
    parser.add_argument("--model_path_src"   , type=str   , help="the directory to load model"                       , default='./saved_models/saved_dict.pth')

    
    args = parser.parse_args()

    test_config = dict()
    for arg in vars(args):
        test_config[arg] = getattr(args, arg)

    segment_text(test_config)