
import numpy as np

import torch.nn as nn

import tensorflow as tf
import matplotlib.pyplot as plt

from dataset import CHARS


# TODO: 加入负样本，联合训练

class FedCTCLoss(nn.Module):
    def __init__(self,
                 alpha = 0.3
                 ):
        super().__init__()
        self.alpha = alpha
        self.pos_loss = nn.CTCLoss(blank=0,zero_infinity=True)
        self.neg_loss = nn.CTCLoss(blank=0,zero_infinity=True)

    def forward(self, 
                model_out,
                model_length, 
                pos_label, 
                pos_length,       
                neg_label,  
                neg_length, 
                ):

    
        return self.pos_loss(model_out, pos_label,model_length,pos_length) - self.alpha*self.neg_loss(model_out, neg_label,model_length,neg_length)
    
        
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


# TODO: tf.nn.ctc_beam_search_decoder
def beam_search_decoder(probs,seq_length):
    """beam_search_decoder _summary_

    Args:
        probs  (float Tensor): size [max_time, batch_size, num_classes]
        seq_length (int32): sequence_length: 1-D int32 vector containing sequence lengths, having size
    [batch_size].

    Returns:
        _type_: _description_
    """
    decoded,_ = tf.nn.ctc_beam_search_decoder(probs,seq_length)

    for result in decoded:
        value = result.values.numpy()
        text = ''.join(chr(idx + ord('a') - 2) for idx in value)

    return text


def greedy_decode(probs, blank_idx=0, space_idx=28):
    
    probs = probs.transpose(0,1)
    probs = np.array(probs.detach().cpu().numpy())

    all_text = []
    number2letter = dict(enumerate(CHARS))
    for batch in range(probs.shape[0]):
        # for a batch

        # 初始化解码结果和前一个字符
        decoded_text = []
        prev_char = None

        # 遍历时间步
        for t in range(probs.shape[1]):
            # 获取当前时间步的最高概率字符的索引
            max_prob_idx = np.argmax(probs[batch][t])

            if max_prob_idx not in [blank_idx, prev_char,space_idx]:  # 0 1 per_char
                # 将字符索引映射为实际字符，并添加到解码结果中
                decoded_text.append(number2letter[max_prob_idx]) 

            # 更新前一个字符
            prev_char = max_prob_idx

        # 将解码结果连接成最终文本
        final_text = ''.join(decoded_text)
        final_text = final_text.replace(' ','').replace('_','')
        all_text.append(final_text)
        
    return all_text


def decode_label(coded_label):
    """decode_label 
    
    Args:
        coded_label (tensor): _description_

    Returns:
        _type_: decoded label 
    """

    number2letter = dict(enumerate(CHARS))
    coded_label = np.array(coded_label.cpu())
    all_text = []

    # 遍历时间步
    for t in range(coded_label.shape[0]):
        word = [
            number2letter[coded_label[t][j]]
            for j in range(coded_label.shape[1])
            if coded_label[t][j] not in [28, 1, 0]
        ]
        word = ''.join(word)
        all_text.append(word)

    return all_text

def compute_acc(text,label):
    return sum(
        1
        for i in range(len(text))
        if text[i].replace('_', '').replace(' ', '').strip()
        == label[i].replace('_', '').replace(' ', '').strip()
    )

def _draw(epochs,train_loss,val_loss,train_acc,val_acc,save):

    # 创建图像
    _, ax1= plt.subplots(figsize=(10, 6))

    # 绘制训练损失曲线
    color1 = 'tab:blue'
    color2 = 'orange'
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.plot(epochs, train_loss, color=color1,linestyle='-', linewidth=2,label='Training Loss')
    ax1.plot(epochs, val_loss, color=color2,linestyle='-', linewidth=2,label='Val  Loss')
    ax1.tick_params(axis='y')

    # 创建第二个y轴
    ax2 = ax1.twinx()

    # 绘制准确率曲线
    color1 = 'tab:green'
    color2 = 'red'
    ax2.set_ylabel('Accuracy')
    ax2.plot(epochs, train_acc, color=color1,linestyle='--', linewidth=2,label='Training Accuracy')
    ax2.plot(epochs, val_acc, color=color2,linestyle='--', linewidth=2,label='Val Accuracy')
    ax2.tick_params(axis='y')

    # 添加图例
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper left')
    # 设置标题
    ax1.set_title('Loss and Accuracy')
    
    # 显示图像
    plt.tight_layout()
    plt.savefig(save)
