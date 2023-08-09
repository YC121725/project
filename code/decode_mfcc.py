
import os
import argparse
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.cuda.amp import autocast,GradScaler
from dataset import AsrDataset
from model import LSTM_ASR_MFCC
import tensorflow as tf


def make_parser():
    test_parser = argparse.ArgumentParser()
    
    test_parser.add_argument('--batch_size',type=int,default=4,
                              help='batch size')
    test_parser.add_argument('--test_path', type=str, required= True,
                              help = 'path to test file')
    test_parser.add_argument('--waveform_file', required=True, type=str,
                              help='path to training data file')
    test_parser.add_argument('--model_path', type=str, required= True,
                              help = 'path to model')
    return test_parser


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
    >>> blank_idx = 0
    >>> space_idx = 4
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
            # if max_prob_idx in [blank_idx, prev_char]:
            #     continue

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
        all_text.append(final_text.strip())

    return all_text


def decode_label(coded_label):
    """decode_label +
    
    
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


def collate_fn(batch):
    """
    This function will be passed to your dataloader.
    It pads word_spelling (and features) in the same batch to have equal length.with 0.
    :param batch: batch of input samples
    :return: (recommended) padded_word_spellings, 
                           padded_features,
                           list_of_unpadded_word_spelling_length (for CTCLoss), target_length
                           list_of_unpadded_feature_length (for CTCLoss), input_length
    """
    
    batch_size = len(batch)
    data  = [torch.tensor(batch[i][0],dtype=torch.float) for i in range(batch_size)]
    label = [torch.tensor(batch[i][1],dtype=torch.float) for i in range(batch_size)]
    
    unpadded_word_spelling_length = torch.tensor(np.array([len(i) for i in label]), dtype=torch.int)
    unpadded_feature_length = torch.tensor(np.array([len(i) for i in data]), dtype=torch.int)
    
    padded_features = pad_sequence(data,batch_first=True,padding_value=0)
    padded_label = pad_sequence(label,batch_first=True,padding_value=0)

    return padded_label, padded_features,unpadded_word_spelling_length,unpadded_feature_length


def decode(args):

    if "code" in os.getcwd():
        root = os.path.abspath(os.path.pardir)
    else:
        root = os.getcwd()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    model = LSTM_ASR_MFCC().to(device)
    model.load_state_dict(torch.load(args.model_path))

    # print(f"root:\t{root}")
    val_file = os.path.join(root, args.test_path)
    # print(val_file)
    waveform = os.path.join(root, args.waveform_file)
    # val_file = os.path.join(root, "data/val")
    print(os.path.join(val_file, "clsp.trnwav"))
    val_set = AsrDataset(
                feature_type='mfcc',
                scr_file = os.path.join(val_file, "clsp.trnscr"),
                wav_scp = os.path.join(val_file, "clsp.trnwav"),
                wav_dir=waveform
                )
            


    val_dataloader = DataLoader(dataset = val_set,
                                batch_size = args.batch_size,
                                shuffle=True,
                                collate_fn=collate_fn
                                )

    acc_num = 0
    total_num = 0
    with torch.no_grad():
        
        for batch in val_dataloader:
            padded_label = batch[0].to(device)
            padded_features = batch[1].to(device)
            unpadded_word_spelling_length = batch[2]
            unpadded_feature_length = batch[3]
            
            output = model(padded_features,unpadded_feature_length)
            print(output.shape)
            # text = beam_search_decoder(output.cpu(),unpadded_word_spelling_length)
            text = greedy_decode(output,0,1)
            label = decode_label(padded_label)

            print(text)
            print(label)
            break
            # for i in range(len(text)):
            #     total_num+=1
            #     if text[i].strip() == label[i].strip():
            #         acc_num +=1
        # print(acc_num)       
        # print(len(val_set))  #174
        # print(f"The Accuracy:{acc_num/len(val_set)}")
            
if __name__ == '__main__':
    parser = make_parser()
    args = parser.parse_args()
    decode(args)