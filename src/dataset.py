import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import BertTokenizer
from collections import Counter
import csv
import re
import os
import copy


'''
read the data from the corpus file
and extract the (text, label) pairs

labels:  S:0  B:1  E:2  M:3 P:4, [PAD]:5,
'''
def get_text_labels(data_path, max_length):
    buffer = 16
    text_list  = list()
    label_list = list()

    with open(data_path, 'r') as file:
        line = file.readline()

        text  = str()
        label = list()
        sentence       = str()
        sentence_label = list()

        while line:
            pattern_characters = r'\['
            line = re.sub(pattern_characters, '', line)
            word_pairs = line.split()
            if len(word_pairs) > 0:
                text = ""
                label.clear()
                sentence = ""
                sentence_label.clear()
                for word_pair in word_pairs[1:]:
                    word_pair = word_pair.split("/")
                    sentence = sentence + word_pair[0]

                    if word_pair[1] == 'w':
                    #  append the sentence to text
                        sentence_label = sentence_label + [4] * len(word_pair[0])

                    elif len(word_pair[0]) == 1:
                        sentence_label.append(0)
                    else:
                        sentence_label = sentence_label + [1] + [3] * (len(word_pair[0]) - 2) + [2]
                        

                    if word_pair[1] == 'w' or len(sentence) > max_length - buffer:
                        if len(sentence) + len(text) > max_length - 2:
                            text_list.append(text)
                            label_list.append(label.copy())
                            text  = ""
                            label.clear()
                        text  = text  + sentence
                        label = label + sentence_label
                        sentence       = ""
                        sentence_label.clear()

                
                if len(sentence) + len(text) > max_length - 2:
                    text_list.append(text)
                    label_list.append(label.copy())
                    text_list.append(sentence)
                    label_list.append(sentence_label.copy())
                else:
                    text  = text  + sentence
                    label = label + sentence_label
                    text_list.append(text)
                    label_list.append(label.copy())

            #  read the next line
            line = file.readline()
        file.close()
    return text_list, label_list


def save_dict(src_dict, table_path):
    with open(table_path, 'w') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in src_dict.items():
            writer.writerow([key,value])
        csv_file.close()


def load_dict(table_path):
    trg_dict = dict()
    with open(table_path, 'r') as f:
        csv_reader = csv.reader(f)
        
        for row in csv_reader:
            key   = int(row[0])
            value = int(row[1])
            trg_dict[key] = value

        f.close()

    return trg_dict


'''
shrink the vocab_size
'''
def shuffle_encoding(text_list, shuffle_dict_path):

    shuffle_dict = dict()
    if os.path.exists(shuffle_dict_path):
        shuffle_dict = load_dict(shuffle_dict_path)
    else:
        token_counter = Counter()
        for text in text_list:
            token_counter.update(text)

        
        i = 0
        for token, counter in token_counter.most_common():
            if counter < 10:
                break
            shuffle_dict[token] = i
            i += 1

        save_dict(shuffle_dict, shuffle_dict_path)

    text_list = [[shuffle_dict.get(token, shuffle_dict[100]) for token in text] for text in text_list]

    return text_list




class PKCorpus(Dataset):
    def __init__(self, data_path, max_length):
        '''
        extract the text and label each symbol from the txt file
        '''
        text_list, label_list = get_text_labels(data_path, max_length)

        length_list = [len(lst) for lst in label_list]

        #  pad the labels
        label_list = [label + [5] * (max_length - len(label)) for label in label_list]


        tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
        encoding = tokenizer(text_list, padding = 'max_length', truncation=True, add_special_tokens=False, max_length=max_length)

        text_list   = encoding['input_ids']
        mask_list   = encoding['attention_mask']

        #  reassign the coding  of the tokens
        text_list = shuffle_encoding(text_list, 'data/shuffle_dict.csv')


        self.text_list   = text_list
        self.mask_list   = mask_list
        self.label_list  = label_list
        self.length_list = length_list
        





    def __len__(self):
        return len(self.text_list)


    def __getitem__(self, idx):
        text  = self.text_list[idx]
        label = self.label_list[idx]
        mask  = self.mask_list[idx]
        length = self.length_list[idx]

        #  tensorlize
        text_tensor   = torch.tensor(text   , dtype = torch.long)
        label_tensor  = torch.tensor(label  , dtype = torch.long)
        mask_tensor   = torch.tensor(mask   , dtype = torch.bool)
        length_tensor = torch.tensor(length , dtype = torch.long)

        
        return text_tensor, label_tensor, mask_tensor, length_tensor
