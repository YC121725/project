#!/usr/bin/env python
# basic
import argparse
import json
import os 
import numpy as np
from tqdm import tqdm
# torch
import torch
from torch.utils.data import DataLoader
# user
from dataset import AsrDataset,collate_fn
from model import LSTM_ASR
from utils import _draw,greedy_decode,decode_label,compute_acc


def train(config):   # sourcery skip: low-code-quality
    
    if "code" in os.getcwd():
        root = os.path.abspath(os.path.pardir)
    else:
        root = os.getcwd()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    train_file = os.path.join(root, config['training_file'])  
    waveform = os.path.join(root, config['waveform_file'])
    
    training_set = AsrDataset(
                scr_file = os.path.join(train_file, "clsp.trnscr"),
                feature_type = config['model_type'],
                feature_file = os.path.join(train_file, "clsp.trnlbls"),
                feature_label_file = os.path.join(train_file, "clsp.lblnames"),
                wav_scp = os.path.join(train_file, "clsp.trnwav"),
                wav_dir = waveform,
                mfcc = config['mfcc_dim']
                )

    train_dataloader = DataLoader(dataset = training_set,
                                  batch_size = config['batch_size'],
                                  shuffle=True,
                                  collate_fn=collate_fn
                                  )

    
    if config['validation_file'] != None:

        val_file = os.path.join(root, config['validation_file']) # "data/val"
        
        val_set = AsrDataset(
                scr_file = os.path.join(val_file, "clsp.trnscr"),
                feature_type = config['model_type'],
                feature_file = os.path.join(val_file, "clsp.trnlbls"),
                feature_label_file = os.path.join(val_file, "clsp.lblnames"),
                wav_scp = os.path.join(val_file, "clsp.trnwav"),
                wav_dir = waveform,
                mfcc = config['mfcc_dim']
                )
        
        val_dataloader = DataLoader(dataset = val_set,
                                    batch_size = config['batch_size'],
                                    shuffle=True,
                                    collate_fn=collate_fn
                                    )
        
    # print(f"Training Set Length: %d" % len(training_set))
    # print(f"validation Set Length: %d" % len(val_set))
    model =  LSTM_ASR(
                config['model_type'],
                config['input_size'], 
                config['hidden_size'], 
                config['num_layers'],
                config['output_size'],
                config['dropout']).to(device)
    
    # loss, optimizer
    ctc_loss = torch.nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    # optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'],momentum=0.9)
    # torch.optim.AdamW()
    
    # Train
    train_total_loss = []
    val_total_loss = []
    train_total_acc = []
    val_total_acc = []
    
    for epoch in range(config['epochs']):
        # Start
        onepochloss = 0
        onepochacc  = 0
        
        train_bar = tqdm(train_dataloader, total=len(train_dataloader),colour='red')
        
        if config['validation_file']:
            val_bar = tqdm(val_dataloader, total=len(val_dataloader),colour='green')
        for data in train_dataloader:
            
            padded_label = data[0].to(device)
            padded_features = data[1].to(device)
            unpadded_word_spelling_length = data[2].to(device)
            unpadded_feature_length = data[3]
            
            output = model(padded_features,unpadded_feature_length)
            

            loss = ctc_loss(output, padded_label,unpadded_feature_length,unpadded_word_spelling_length)
            loss.backward()
            
            decoded_output = greedy_decode(output)
            decoded_label  = decode_label(padded_label)

            # print(decoded_output)
            # print(decoded_label)
            onepochacc += compute_acc(decoded_output,decoded_label)
            
            optimizer.step()
            optimizer.zero_grad()
            
            onepochloss+=loss.item()
            
            train_bar.set_description("train epcoh:{}  acc:{}/{}({:.6f})  loss:{:.6f}".format(epoch,onepochacc,len(training_set),onepochacc/len(training_set), loss.item()))
            train_bar.update(1)
            
        train_total_loss.append(onepochloss/len(train_dataloader))
        train_total_acc.append(onepochacc/len(training_set))
        
        # 验证集       
        if config['validation_file']:     
            onepochloss = 0
            onepochacc = 0
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
                    onepochacc += compute_acc(decoded_output,decoded_label)
                    
                    onepochloss+=loss.item()
                    val_bar.set_description("val epcoh:{}  acc:{}/{}({:.6f})   loss:{:.6f}".format(epoch,onepochacc,len(val_set), onepochacc/len(val_set), loss.item()))
                    val_bar.update(1)
                
                val_total_loss.append(onepochloss/len(val_dataloader))
                val_total_acc.append(onepochacc/len(val_set))
        # End
    if config['save_model']:
        torch.save(model.state_dict(), config['save_model']) # '../save_model/asr.ckpt'

    if config['save_pic']:
        epochs = np.arange(1,config['epochs']+1)
        _draw(epochs,train_total_loss,val_total_loss,train_total_acc,val_total_acc,config['save_pic'])


def make_parser():  
    train_parser = argparse.ArgumentParser(description="Model Training")
    train_parser.add_argument("--config", type=str, default="../config/lstm_config.json", 
                              help="Path to config.json file")

    return train_parser


if __name__ == '__main__':
    
    parser = make_parser()
    args = parser.parse_args()
    with open(args.config, "r") as config_file:
        config = json.load(config_file)
    train(config)
