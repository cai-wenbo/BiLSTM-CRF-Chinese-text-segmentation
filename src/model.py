import torch
import torch.nn as nn
from torch.nn.modules.fold import F
from torch.nn.modules.loss import _Reduction
from torchcrf import CRF
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence



'''
BiLSTM followed by a Conditional random field for text segmentation
'''
class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, embedding_dim, LSTM_hidden_size, LSTM_num_layers, num_labels, sequence_length):
        super(BiLSTM_CRF, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm      = nn.LSTM(
                input_size    = embedding_dim,
                hidden_size   = LSTM_hidden_size,
                num_layers    = LSTM_num_layers,
                bidirectional = True,
                batch_first   = True
                )

        self.fc = nn.Linear(LSTM_hidden_size * 2, num_labels)
        self.softmax = nn.Softmax(dim = -1)
        self.crf = CRF(
                num_tags    = num_labels,
                batch_first = True
                )
        self.sequence_length = sequence_length

    
    '''
    you can skip the manual model parameters initialization 
    because pytorch has already done it for  us, using 
    Kaiming He initialization
    '''


    '''
    return the most likely tag sequence
    '''
    #  batched_text = (batch_size, sequence_length)
    def forward(self, batched_text, batched_mask=None, lengths = None):
        #  embedded_inputs shape = (batch_size, sequence_length, embedding_dim)
        embedded_text = self.embedding(batched_text)

        #  LSTM_out shape = (batch_size, sequence_length, 2 * LSTM_hidden_size)
        if lengths is not None:
            packed_text = pack_padded_sequence(embedded_text, lengths = lengths.cpu(), batch_first=True, enforce_sorted=False)
            packed_LSTM_out, _ = self.lstm(packed_text)
            LSTM_out, _ = pad_packed_sequence(packed_LSTM_out, batch_first=True, total_length=self.sequence_length)
        else:
            LSTM_out, _ = self.lstm(embedded_text)

        # probs shape = ( batch_size, sequence_length, num_labels)
        logits = self.fc(LSTM_out)
        probs = self.softmax(logits)



        #  get the most likely tag sequence
        if batched_mask is not None:
            predicts = self.crf.decode(probs, batched_mask)
            predicts = [predict + [5] * (self.sequence_length - len(predict)) for predict in predicts]
        else:
            predicts = self.crf.decode(probs)

        
        predicts = torch.tensor(predicts, dtype = torch.long).to(self.device)

        return predicts


    '''
    return the negative log likelihood
    '''
    def get_log_likelihood(self, batched_text, batched_label, batched_mask, lengths):
        #  embedded_inputs shape = (batch_size, sequence_length, embedding_dim)
        embedded_text = self.embedding(batched_text)

        #  LSTM_out shape = (batch_size, sequence_length, 2 * LSTM_hidden_size)
        packed_text = pack_padded_sequence(embedded_text, lengths=lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_LSTM_out, _ = self.lstm(packed_text)
        LSTM_out, _ = pad_packed_sequence(packed_LSTM_out, batch_first=True, total_length=self.sequence_length)

        # probs shape = ( batch_size, sequence_length, num_labels)
        logits = self.fc(LSTM_out)
        probs = self.softmax(logits)


        #  scores shape = (batch_size, sequence_length, vocab_size + 2)
        batched_log_likelihood = self.crf(probs, batched_label, batched_mask, reduction = 'none')

        return batched_log_likelihood
