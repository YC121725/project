import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

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


def greedy_decode(probs, blank_idx, space_idx):
    """decode function
    # TODO: 根据实际输出确定参数类型，完善代码
    # NOTE: 
    
    Args:
        probs (_type_): shape:(N, T, C)
        blank_idx (_type_): _description_
        space_idx (_type_): _description_
    
    # 假设我们有一个CTC输出概率矩阵probs，空白符的索引是0，空格符的索引是4
    # 
    probs = [
       [0.1, 0.2, 0.05, 0.1, 0.2],   # t=0
       [0.2, 0.05, 0.1, 0.1, 0.3],   # t=1
       [0.1, 0.3, 0.2, 0.1, 0.1],    # t=2
       [0.15, 0.1, 0.3, 0.2, 0.05],  # t=3
       [0.05, 0.2, 0.1, 0.2, 0.1],   # t=4
    ]
    >>>
    blank_idx = 0
    space_idx = 4
    >>>
    decoded_result = greedy_decode(probs, blank_idx, space_idx)
    print(decoded_result)  # 输出： "abcd"
    >>>
    
    Returns:
        _type_: _description_
    """
    probs = probs.transpose(0,1)
    probs = np.array(probs.cpu())
    # 将概率矩阵转换为NumPy数组
    all_text = []
    for batch in range(probs.shape[0]):
        # for a batch

        # 初始化解码结果和前一个字符
        decoded_text = []
        prev_char = None

        # 遍历时间步
        for t in range(probs.shape[1]):
            # 获取当前时间步的最高概率字符的索引
            max_prob_idx = np.argmax(probs[batch][t])

            # 如果当前字符是空白符或与前一个字符相同（重复字符），跳过
            if max_prob_idx in [blank_idx, prev_char]:
                continue

            # 如果当前字符是空格，将其映射为" "
            if max_prob_idx == space_idx:
                decoded_text.append(" ")
            else:
                # 将字符索引映射为实际字符，并添加到解码结果中
                decoded_text.append(chr(max_prob_idx + ord('a') - 2))

            # 更新前一个字符
            prev_char = max_prob_idx

        # 将解码结果连接成最终文本
        final_text = ''.join(decoded_text)
        all_text.append(final_text)

    return all_text


def decode_label(coded_label):
    """decode_label 
    
    Args:
        coded_label (tensor): _description_

    Returns:
        _type_: decoded label 
    """
    CHARS = [" ","<sil>","a","b","c","d","e","f","g",
                    "h","i","j","k","l","m","n",
                    "o","p","q","r","s","t",
                    "u","v","w","x","y","z",]
    number2letter = dict(enumerate(CHARS))
    coded_label = np.array(coded_label.cpu())
    all_text = []

    # 遍历时间步
    for t in range(coded_label.shape[0]):
        word = [
            number2letter[coded_label[t][j]]
            for j in range(coded_label.shape[1])
            if coded_label[t][j] not in [1, 0]
        ]
        word = ''.join(word)
        all_text.append(word)

    return all_text


def compute_acc(text,label):
    """compute_acc 

    Args:
        text (list): _description_
        label (list): _description_

    Returns:
        acc_num: the acc_num between the label and the text
    """
    
    return sum(text[i].strip() == label[i].strip() for i in range(len(text)))

def _draw(epochs,train_loss,val_loss,train_acc,val_acc,save):
    plt.plot(epochs,train_loss)
    plt.plot(epochs,val_loss)
    plt.plot(epochs,train_acc)
    plt.plot(epochs,val_acc)
    
    # 添加标题和标签
    plt.title(r'Training Loss Curve', fontsize=16)
    plt.xlabel(r'Epoch', fontsize=14)
    plt.ylabel(r'Loss', fontsize=14)

    # # 调整坐标轴刻度
    # plt.xticks(epochs, fontsize=12)
    # plt.yticks(np.arange(min(total_loss)-0.1, max(total_loss)+0.2, 0.2), fontsize=12)

    # 添加图例，调整图例位置
    plt.legend(fontsize=12, loc='upper right')

    # 调整图形边距
    plt.tight_layout()

    plt.savefig(save)