#!/usr/bin/env python

import argparse
import os 
import torch
from torch.utils.data import DataLoader
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from dataset import AsrDataset
from model import LSTM_ASR_MFCC
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
import seaborn as sns
  
def make_parser():  
    train_parser = argparse.ArgumentParser()
    train_parser.add_argument('--batch_size',type=int,default=4,
                              help='batch size')
    train_parser.add_argument('-e','--epochs',type=int,default=20,
                              help='epochs')
    train_parser.add_argument('-lr','--learning_rate',type=float,default=0.005,
                              help='learning rate')
    train_parser.add_argument('--training_file', required=True, type=str,
                              help='path to training data file')
    train_parser.add_argument('--waveform_file', required=True, type=str,
                              help='path to training data file')
    train_parser.add_argument('--validation_file', default=None,
                              help='path to validation_file')
    train_parser.add_argument('--save_path', default=None,
                              help='path to save the model')
    train_parser.add_argument('--save_train_loss',default=None,
                              help='path to save the training loss')
    train_parser.add_argument('--save_val_loss',default=None,
                              help='path to save the validation loss')
    return train_parser

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

def train(args): 
    
    if "code" in os.getcwd():
        root = os.path.abspath(os.path.pardir)
    else:
        root = os.getcwd()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    train_file = os.path.join(root, args.training_file)  # "data/train"
    waveform = os.path.join(root, args.waveform_file)

    training_set = AsrDataset(
        feature_type='mfcc',
        scr_file = os.path.join(train_file, "clsp.trnscr"),
        wav_scp = os.path.join(train_file, "clsp.trnwav"),
        wav_dir=waveform
        )
    train_dataloader = DataLoader(dataset = training_set,
                                  batch_size = args.batch_size,
                                  shuffle=True,
                                  collate_fn=collate_fn
                                  )
    
    if args.validation_file != None:

        val_file = os.path.join(root, args.validation_file) # "data/val"
        
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
        
    model =  LSTM_ASR_MFCC().to(device)
    ctc_loss = torch.nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    train_total_loss = []
    val_total_loss = []
    for epoch in range(args.epochs):
        onepochloss = 0
        train_bar = tqdm(train_dataloader, total=len(train_dataloader),colour='red')
        if args.validation_file:
            val_bar = tqdm(val_dataloader, total=len(val_dataloader),colour='green')
        for data in train_dataloader:
                
            padded_label = data[0].to(device)
            padded_features = data[1].to(device)
            unpadded_word_spelling_length = data[2].to(device)
            unpadded_feature_length = data[3]

            output = model(padded_features,unpadded_feature_length)
            loss = ctc_loss(output, padded_label,unpadded_feature_length,unpadded_word_spelling_length)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            onepochloss+=loss.item()
            
            train_bar.set_description("train epcoh:{}  loss:{:.6f}".format(epoch, loss.item()))
            train_bar.update(1)
        train_total_loss.append(onepochloss/len(train_dataloader))
        # 验证集       
        if args.validation_file:     
            onepochloss = 0
            with torch.no_grad():
                for batch in val_dataloader:
                    padded_label = batch[0].to(device)
                    padded_features = batch[1].to(device)
                    unpadded_word_spelling_length = batch[2].to(device)
                    unpadded_feature_length = batch[3]
                    
                    output = model(padded_features,unpadded_feature_length)
                    loss = ctc_loss(output, padded_label,unpadded_feature_length,unpadded_word_spelling_length)
                    onepochloss+=loss.item()
                    val_bar.set_description("val epcoh:{}  loss:{:.6f}".format(epoch, loss.item()))
                    val_bar.update(1)
                val_total_loss.append(onepochloss/len(val_dataloader))
    if args.save_path:
        torch.save(model.state_dict(), args.save_path) # '../save_model/asr.ckpt'
        
    if args.save_train_loss:
        epochs = np.arange(1,args.epochs+1)
        _draw_training_loss(epochs,train_total_loss,args.save_loss)
        _draw_training_loss(epochs,val_total_loss,args.save_loss)
        
        
def _draw_training_loss(epochs,total_loss,save_loss):
    # 设置 Seaborn 风格
    sns.set(style="whitegrid")
    sns.set_palette("colorblind")  # 使用色盲友好的调色板
    
    # 创建一个绘图窗口
    plt.figure(figsize=(10, 6))

    # 使用 Seaborn 绘制损失曲线
    sns.lineplot(x=epochs, y=total_loss, marker='o', color='b', label='Loss', linewidth=2)

    # 添加数据标签
    for i, txt in enumerate(total_loss):
        plt.annotate(f"{txt:.2f}", (epochs[i], total_loss[i]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=10, color='black')

    # 添加标题和标签
    plt.title(r'Training Loss Curve', fontsize=16)
    plt.xlabel(r'Epoch', fontsize=14)
    plt.ylabel(r'Loss', fontsize=14)

    # 添加网格线，调整线条风格
    plt.grid(True, alpha=0.3, linestyle='--')

    # 调整坐标轴刻度
    plt.xticks(epochs, fontsize=12)
    plt.yticks(np.arange(min(total_loss)-0.1, max(total_loss)+0.2, 0.2), fontsize=12)

    # 添加图例，调整图例位置
    plt.legend(fontsize=12, loc='upper right')

    # 调整图形边距
    plt.tight_layout()

    plt.savefig(save_loss,dpi=1000)
    # 显示图形
    # plt.show()

if __name__ == '__main__':
    parser = make_parser()
    args = parser.parse_args()
    train(args)