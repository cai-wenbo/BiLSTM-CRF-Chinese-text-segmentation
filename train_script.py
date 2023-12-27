from random import shuffle
from huggingface_hub import add_space_secret
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from src.dataset import PKCorpus
from src.model import BiLSTM_CRF
import json
import os
import argparse




def train(training_config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 


    '''
    load model and the history
    '''

    model = BiLSTM_CRF(
            vocab_size       = training_config['vocab_size'],
            embedding_dim    = training_config['embedding_dim'],
            LSTM_hidden_size = training_config['LSTM_hidden_size'],
            num_labels       = training_config['num_labels'],
            sequence_length  = training_config['sequence_length']
            )

    if os.path.exists(training_config['model_path_src']):
        model_dict = torch.load(training_config['model_path_src'])
        model.load_state_dict(model_dict)


    model = model.to(device)


    #  load the losses history
    step_losses = list()
    if os.path.exists(training_config['step_losses_pth']):
        with open(training_config['step_losses_pth'], 'r') as file:
            step_losses = json.load(file)
            file.close()

    train_losses = list()
    if os.path.exists(training_config['train_losses_pth']):
        with open(training_config['train_losses_pth'], 'r') as file:
            train_losses = json.load(file)
            file.close()
    
    test_losses = list()
    if os.path.exists(training_config['test_losses_pth']):
        with open(training_config['test_losses_pth'], 'r') as file:
            test_losses = json.load(file)
            file.close()

    '''
    dataloader
    '''
    train_data = PKCorpus('data/ChineseCorpus_train.txt' , max_length = training_config['sequence_length'])
    test_data  = PKCorpus('data/ChineseCorpus_text.txt'  , max_length = training_config['sequence_length'])

    dataloader_train = DataLoader(train_data , batch_size = training_config['batch_size'] , shuffle = True)
    dataloader_test  = DataLoader(test_data  , batch_size = training_config['batch_size'] , shuffle = False)


    '''
    optimizer
    '''
    optimizer = AdamW(
            model.parameters(),
            lr = training_config['learning_rate'],
            weight_decay=training_config['weight_decay']
            )



    '''
    creterian
    '''
    '''
    We don't need it here
    '''




    '''
    train_loops
    '''
    
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs.")
        model = nn.DataParallel(model)


    for epoch in range(training_config['num_of_epochs']):
        loss_sum_train = 0
        correct        = 0
        
        model.train()
        #  train loop
        for i, batch in enumerate(dataloader_train):
            b_text_tensor, b_label_tensor, b_mask_tensor, b_length_tensor = batch
            b_text_tensor = b_text_tensor.to(device)
            b_label_tensor = b_label_tensor.to(device)
            b_mask_tensor = b_mask_tensor.to(device)

            optimizer.zero_grad()


            loss = -model.module.get_log_likelihood(
                    batched_text  = b_text_tensor,
                    batched_label = b_label_tensor,
                    batched_mask  = b_mask_tensor,
                    lengths       = b_length_tensor
                    ).sum()


            loss.backward()
            optimizer.step()
            loss_scalar = loss.item()
            loss_sum_train += loss_scalar
            step_losses.append(loss_scalar)



        train_loss = loss_sum_train / len(dataloader_train)
        train_losses.append(train_loss)




        loss_sum_test = 0
        err       = 0
        tokens_num    = 0

        model.eval() 
        #  test_loop
        for i, batch in enumerate(dataloader_test):
            b_text_tensor, b_label_tensor, b_mask_tensor, b_length_tensor = batch
            b_text_tensor = b_text_tensor.to(device)
            b_label_tensor = b_label_tensor.to(device)
            b_mask_tensor = b_mask_tensor.to(device)


            optimizer.zero_grad()

            loss = -model.module.get_log_likelihood(
                    batched_text  = b_text_tensor,
                    batched_label = b_label_tensor,
                    batched_mask  = b_mask_tensor,
                    lengths       = b_length_tensor
                    ).sum()


            loss_scalar = loss.item()
            loss_sum_test += loss_scalar
            step_losses.append(loss_scalar)

            b_predicts = model(
                    batched_text = b_text_tensor,
                    batched_mask = b_mask_tensor,
                    lengths      = b_length_tensor
                    )
            err += (b_predicts != b_label_tensor).sum().item()
            tokens_num += b_length_tensor.sum().item()
            
        


        test_loss = loss_sum_test / len(dataloader_test)
        test_losses.append(test_loss)
        test_acc = 1 - err / tokens_num


        print(f'Epoch: {epoch+1} \n Train Loss: {train_loss:.6f}, train Test Loss: {test_loss:.6f}, Test Accuracy: {test_acc:.6f}')


    '''    
    save model and data
    '''

    model = model.to('cpu').module
    torch.save(model.state_dict(), training_config['model_path_dst'])

    #  save the loss of the steps
    with open(training_config['step_losses_pth'], 'w') as file:
        json.dump(step_losses, file)
        file.close()

    with open(training_config['train_losses_pth'], 'w') as file:
        json.dump(train_losses, file)
        file.close()
    
    with open(training_config['test_losses_pth'], 'w') as file:
        json.dump(test_losses, file)
        file.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_of_epochs"    , type=int   , help="number of epochs"                                  , default=5)
    parser.add_argument("--batch_size"       , type=int   , help="batch size"                                        , default=512)
    parser.add_argument("--learning_rate"    , type=float , help="learning rate"                                     , default=1e-3)
    parser.add_argument("--weight_decay"     , type=float , help="weight_decay"                                      , default=1e-4)
    parser.add_argument("--vocab_size"       , type=int   , help="vocab size"                                        , default=2979)
    parser.add_argument("--embedding_dim"    , type=int   , help="embedding dimmention"                              , default=512)
    parser.add_argument("--LSTM_hidden_size" , type=int   , help="hidden_size of the BiLSTM model"                   , default=256)
    parser.add_argument("--num_labels"       , type=int   , help="types of labels"                                   , default=6)
    parser.add_argument("--sequence_length"  , type=int   , help="sequence_length"                                   , default=128)
    parser.add_argument("--model_path_dst"   , type=str   , help="the directory to save model"                       , default='./saved_models/saved_dict.pth')
    parser.add_argument("--model_path_src"   , type=str   , help="the directory to load model"                       , default='./saved_models/saved_dict.pth')
    parser.add_argument("--step_losses_pth"  , type=str   , help="the path of the json file that saves step losses"  , default='./trails/step_losses.json')
    parser.add_argument("--train_losses_pth" , type=str   , help="the path of the json file that saves train losses" , default='./trails/train_losses.json')
    parser.add_argument("--test_losses_pth"  , type=str   , help="the path of the json file that saves test losses"  , default='./trails/test_losses.json')

    
    args = parser.parse_args()

    training_config = dict()
    for arg in vars(args):
        training_config[arg] = getattr(args, arg)

    train(training_config)

