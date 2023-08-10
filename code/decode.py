
import os
import argparse
import numpy as np
from tqdm import tqdm

# torch
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import tensorflow as tf

# user
from dataset import AsrDataset,collate_fn
from model import LSTM_ASR, LSTM_ASR_MFCC
from utils import *

def make_parser():
    test_parser = argparse.ArgumentParser()
    
    test_parser.add_argument('--batch_size',type=int,default=4,
                              help='batch size')
    test_parser.add_argument('--test_path', type=str, required= True,
                              help = 'path to test file')
    test_parser.add_argument('--waveform_file', default='data/waveforms', type=str,
                              help='path to training data file')
    test_parser.add_argument('--model_path', type=str, required= True,
                              help = 'path to model')
    return test_parser


def decode(args):

    if "code" in os.getcwd():
        root = os.path.abspath(os.path.pardir)
    else:
        root = os.getcwd()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    
    if args.model_type == 'discrete':
        model = LSTM_ASR().to(device)
    if args.model_type == 'mfcc':
        model = LSTM_ASR_MFCC().to(device)
    
    model.load_state_dict(torch.load(args.model_path))

    val_file = os.path.join(root, args.test_path)
    waveform = os.path.join(root, args.waveform_file)
    
    val_set = AsrDataset(
        scr_file = os.path.join(val_file, "clsp.trnscr"),
        feature_type = args.model_type,
        feature_file = os.path.join(val_file, "clsp.trnlbls"),
        feature_label_file = os.path.join(val_file, "clsp.lblnames"),
        wav_scp = os.path.join(val_file, "clsp.trnwav"),
        wav_dir = waveform,
        type='train',
        mfcc = args.mfcc_dim
        )

    val_dataloader = DataLoader(dataset = val_set,
                                batch_size = args.batch_size,
                                shuffle=True,
                                collate_fn=collate_fn
                                )

    with torch.no_grad():
        
        for batch in val_dataloader:
            padded_label = batch[0].to(device)
            padded_features = batch[1].to(device)
            unpadded_word_spelling_length = batch[2]
            unpadded_feature_length = batch[3]
            
            output = model(padded_features,unpadded_feature_length)
            
            text = greedy_decode(output,0,1)
            label = decode_label(padded_label)
            
        acc_num = compute_acc(text,label)
        print(f"The Accuracy:{acc_num}/{len(val_set)}'\t'{acc_num/len(val_set)}")
            
if __name__ == '__main__':
    parser = make_parser()
    args = parser.parse_args()
    decode(args)