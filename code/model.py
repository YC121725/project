import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils


class LSTM_ASR(nn.Module):
    def __init__(self,
                 model_type = 'discrete',
                 input_size=64, 
                 hidden_size=256, 
                 num_layers=2,
                 output_size=29,
                 dropout = 0.5
                 ): 
        
        super().__init__()

        self.model_type = model_type
        assert self.model_type in ['discrete', 'mfcc']
        
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(num_embeddings=300,embedding_dim=input_size)
        
        self.lstm = torch.nn.LSTM(input_size = input_size,
                                    hidden_size = hidden_size,
                                    num_layers = num_layers,
                                    bidirectional = True,
                                    batch_first = True,
                                    dropout = dropout)
        
        self.fc = nn.Linear(hidden_size, output_size)  
        
    def forward(self, input_features,input_lengths):
        
        # (N, T, C) N:Batch size, T:sequence length C:feature dimension
        if self.model_type == 'discrete':
            input_features = self.embed(input_features)
        
        packed_input = rnn_utils.pack_padded_sequence(input_features, input_lengths, batch_first = True, enforce_sorted = False)
        packed_output, (_, _) = self.lstm(packed_input)

        output, _ = rnn_utils.pad_packed_sequence(packed_output, batch_first=True)

        H = output[:,:,:self.hidden_size] + output[:,:,self.hidden_size:]
        
        logits = F.log_softmax(self.fc(H),dim=2)
        logits = logits.transpose(0,1)
        return logits
    
    
    #         
    # def _init_weights(self, m):
    #     # 仅对线性层和卷积层进行xavier初始化
    #     if type(m) == nn.Linear:
    #         nn.init.xavier_normal_(m.weight)

# class Predicter(nn.Module):
#     def __init__(self, *args, **kwargs) -> None:
#         super().__init__(*args, **kwargs)
        
        
#     def forward(self,x):
        