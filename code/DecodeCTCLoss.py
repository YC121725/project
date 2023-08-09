
import torch.nn as nn

# # TODO: 加入负样本，联合训练
class DecodeCTCLoss(nn.Module):
    def __init__(self):
        super(DecodeCTCLoss, self).__init__()
    
    def forward(self, model_out,target,Input):
        pass
        # 时域损失
    #     LT = torch.mean(torch.abs(torch.sub(model_out,target)))
        
    #     # out与label频域损失
    #     CTCLoss = nn.CTCLoss()  
    #     # 整体频域损失
    #     LPCM = CTCLoss + ResMagLoss
        
    #     LT_PCM  = 0.6*LT + 0.4*LPCM
    #     return LT_PCM
    
    # @staticmethod
    # def decode():