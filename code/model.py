import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils


class LSTM_ASR(nn.Module):
    def __init__(self,
                 input_size=64, 
                 hidden_size=256, 
                 num_layers=2,
                 output_size=28
                 ): 
        
        super().__init__()
      
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(num_embeddings=300,embedding_dim=input_size)
        self.lstm = torch.nn.LSTM(input_size = input_size,
                                    hidden_size = hidden_size,
                                    num_layers = num_layers,
                                    bidirectional = True,
                                    batch_first = True,
                                    dropout = 0.5)
        
        self.fc = nn.Linear(hidden_size, output_size)  
        
    def forward(self, input_features,input_lengths):

        batch_input_features = self.embed(input_features)        # (N, T, C) N:Batch size, T:sequence length C:feature dimension
        
        packed_input = rnn_utils.pack_padded_sequence(batch_input_features, input_lengths, batch_first = True, enforce_sorted = False)
        packed_output, (_, _) = self.lstm(packed_input)
        # print(f"packed_output.shape:\t {packed_output}")
        output, _ = rnn_utils.pad_packed_sequence(packed_output, batch_first=True)

        H = output[:,:,:self.hidden_size] + output[:,:,self.hidden_size:]
    
        logits = F.log_softmax(self.fc(H),dim=-1)
        
        logits = logits.transpose(0,1)
        return logits
    
    # def _init_weights(self, m):
    #     # 仅对线性层和卷积层进行xavier初始化
    #     if type(m) == nn.Linear:
    #         nn.init.xavier_normal_(m.weight)
            
            
class LSTM_ASR_MFCC(nn.Module):
    def __init__(self,
                 input_size=40,#39  48 
                 hidden_size=256, 
                 num_layers=2,
                 output_size=28,
                 init = None):  # sourcery skip: remove-redundant-if
        super().__init__()
        self.hidden_size = hidden_size

        self.lstm = torch.nn.LSTM(input_size = input_size,
                                    hidden_size = hidden_size,
                                    num_layers = num_layers,
                                    bidirectional = True,
                                    batch_first = True,
                                    dropout = 0.5)

        self.fc = nn.Linear(hidden_size, output_size)
        
        # if init:
        #     self.apply(self._init_weights)    
        
    def forward(self, input_features,input_lengths):
        """forward _summary_
        Args:
            input_features: (N, T) N:Batch size, T:sequence length
            input_lengths: _description_

        Returns:
            _type_: _description_
        """
        # batch_input_features = self.embed(input_features)        # (N, T, C=48) N:Batch size, T:sequence length C:feature dimension
        
        packed_input = rnn_utils.pack_padded_sequence(input_features, input_lengths, batch_first = True, enforce_sorted = False)
        packed_output, (_, _) = self.lstm(packed_input)
        # print(f"packed_output.shape:\t {packed_output}")
        output, _ = rnn_utils.pad_packed_sequence(packed_output, batch_first=True)

        H = output[:,:,:self.hidden_size] + output[:,:,self.hidden_size:]

        logits = F.log_softmax(self.fc(H),dim=-1)
        
        logits = logits.transpose(0,1)
        return logits
    
    # def _init_weights(self, m):
    #     # 仅对线性层和卷积层进行xavier初始化
    #     if type(m) == nn.Linear:
    #         nn.init.xavier_normal_(m.weight)