
import os
import json
import argparse
import numpy as np
from tqdm import tqdm

# torch
import torch
from torch.utils.data import DataLoader
import tensorflow as tf

# user
from dataset import AsrDataset,collate_fn
from model import LSTM_ASR
from utils import *


def decode(config):

    if "code" in os.getcwd():
        root = os.path.abspath(os.path.pardir)
    else:
        root = os.getcwd()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    
    
    model = LSTM_ASR(
                config['model_type'],
                config['input_size'], 
                config['hidden_size'], 
                config['num_layers'],
                config['output_size'],
                config['dropout']).to(device)
    
    model.load_state_dict(torch.load(config['save_model']))

    val_file = os.path.join(root, config['validation_file'])
    waveform = os.path.join(root, config['waveform_file'])
    
    val_set = AsrDataset(
        scr_file = os.path.join(val_file, "clsp.trnscr"),
        feature_type = config['model_type'],
        feature_file = os.path.join(val_file, "clsp.trnlbls"),
        feature_label_file = os.path.join(val_file, "clsp.lblnames"),
        wav_scp = os.path.join(val_file, "clsp.trnwav"),
        wav_dir = waveform,
        type='train',
        mfcc = config['mfcc_dim']
        )

    val_dataloader = DataLoader(dataset = val_set,
                                batch_size = config['batch_size'],
                                shuffle=True,
                                collate_fn=collate_fn
                                )
    val_total_loss = []
    val_total_acc = []
    
    ctc_loss = torch.nn.CTCLoss(blank=0, zero_infinity=True)
    
    onepochloss = 0
    onepochacc = 0
    val_bar = tqdm(val_dataloader, total=len(val_dataloader),colour='green')
    with torch.no_grad():
        for batch in val_dataloader:
            padded_label = batch[0].to(device)
            padded_features = batch[1].to(device)
            unpadded_word_spelling_length = batch[2].to(device)
            unpadded_feature_length = batch[3]
            
            output = model(padded_features,unpadded_feature_length)
            
            loss = ctc_loss(output, padded_label,unpadded_feature_length,unpadded_word_spelling_length)
            
            decoded_output = greedy_decode(output)
            decoded_label  = decode_label(padded_label)
            print(decoded_output)
            print(decoded_label)
            onepochacc += compute_acc(decoded_output,decoded_label)
            
            onepochloss+=loss.item()
            val_bar.set_description("acc:{}/{}({:.6f})   loss:{:.6f}".format(onepochacc,len(val_set), onepochacc/len(val_set), loss.item()))
            val_bar.update(1)
        
        val_total_loss.append(onepochloss/len(val_dataloader))
        val_total_acc.append(onepochacc/len(val_set))
        
def make_parser():  
    decode_parser = argparse.ArgumentParser(description="Model Training")
    decode_parser.add_argument("--config", type=str, default="../config/lstm_decode.json", 
                              help="Path to config.json file")

    return decode_parser

if __name__ == '__main__':
    
    parser = make_parser()
    args = parser.parse_args()
    with open(args.config, "r") as config_file:
        config = json.load(config_file)
    decode(config)


            