import librosa
import os
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

import tensorflow as tf
import matplotlib.pyplot as plt

CHARS = [" ","_","a","b","c","d","e","f","g",
                        "h","i","j","k","l","m","n",
                        "o","p","q","r","s","t",
                        "u","v","w","x","y","z","<sil>"]
class AsrDataset(Dataset):
    def __init__(
        self,
        scr_file,
        feature_type="discrete",
        feature_file=None,
        feature_label_file=None,
        wav_scp=None,
        wav_dir=None,
        type= 'train',
        mfcc = 40,
        neg_sample = False,
    ):

        self.feature_type = feature_type
        assert self.feature_type in ["discrete", "mfcc"]
        
        self.type = type
        assert self.type in ['train','test']
        
        self.mfcc = mfcc
        
        if self.feature_type == "discrete":
            self.input_feature= self._extract_discrete(feature_label_file,feature_file)
            self.output_feature = self._extract_label(scr_file)

        if self.feature_type == "mfcc":
            self.input_feature = self._extract_mfcc(wav_scp,wav_dir,self.mfcc)
            self.output_feature = self._extract_label(scr_file)
        
        
    def __len__(self):
        
        return len(self.input_feature)

    def __getitem__(self, idx):
        if self.type=='train':
            return self.input_feature[idx], self.output_feature[idx]
        
        if self.type=='test':
            return self.input_feature[idx]
            

    # Here are some functions
    
    def _extract_discrete(self,feature_label_file=None,feature_file=None):
        
        # read the feature_label_file
        with open(feature_label_file) as f:
            self.feature_dict = {'<pad>': 0}
            
            for i, feat in enumerate(f.readlines()):
                if feat.strip() != '':
                    self.feature_dict[feat.strip()] = i+1

        # read discrete feature
        with open(feature_file) as f:
            input_feature = []
            for i, feat in enumerate(f.readlines()):
                if i==0:
                    continue
                feat = feat.strip().split()
                input_feature.append([])
                for index in feat:
                    if index in self.feature_dict:
                        input_feature[i - 1].append(self.feature_dict[index])

            return input_feature
    
    def get_feature_dict(self):
        return self.feature_dict
        
    @staticmethod
    def _extract_non_zero_features(audio):
        for i,data in enumerate(audio):
            if data != 0:
                Start = i
                break
            
        for j,data in enumerate(reversed(audio)):
            if data != 0:
                End = j
                break    
        return audio[Start:len(audio)-End]
    
    def _extract_mfcc(self,wav_scp, wav_dir,mfcc_dim,non_zero=True):
        """
        Compute MFCC acoustic features (dim=40) for each wav file.
        
        Args:
            wav_scp (_type_): _description_
            wav_dir (_type_): _description_
            mfcc_dim (_type_): _description_
            non_zero (bool, optional): _description_. Defaults to True.
            
        
        """
        features = []
        with open(wav_scp, "r") as f:
            for wavfile in f:
                wavfile = wavfile.strip()
                if wavfile == "clsp.trnwav":  # skip header
                    continue
                
                wav, sr = librosa.load(os.path.join(wav_dir, wavfile), sr=None)
                if non_zero:
                    wav = self._extract_non_zero_features(audio=wav)
                
                feats = librosa.feature.mfcc(
                    y=wav, sr=16000, n_mfcc=mfcc_dim, hop_length=160, win_length=400
                ).transpose()
                
                features.append(feats)

        return features
    
    @staticmethod
    def letter2num():
        return {letter: i for i, letter in enumerate(CHARS)}
    
    @staticmethod
    def num2letter():
        return dict(enumerate(CHARS))
    
    
    def _extract_label(self, scr_file,sil=True):

        self.letter_dict = self.letter2num()
        result = []
        with open(scr_file) as f:
            for i, line in enumerate(f.readlines()):
                if i == 0:  
                    continue
                # 生成每个单词的字母列表 ['e', 'a', 'c', 'h']
                each_word = list(line.strip())
                
                # 在重复单词中添加<sil>
                word = []

                per_char = None
                for i in each_word:
                    if i== per_char:
                        word.append("_")
                    word.append(i)
                    per_char = i
                
                letter = ['<sil>'] + word + ['<sil>'] if sil else word
                result.append([self.letter_dict[j] for j in letter])
        return result
    

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
    data  = [torch.tensor(batch[i][0]) for i in range(batch_size)]
    label = [torch.tensor(batch[i][1]) for i in range(batch_size)]
    
    unpadded_word_spelling_length = torch.tensor(np.array([len(i) for i in label]))
    unpadded_feature_length = torch.tensor(np.array([len(i) for i in data]))
    
    padded_features = pad_sequence(data,batch_first=True,padding_value=0)
    padded_label = pad_sequence(label,batch_first=True,padding_value=0)

    return padded_label, padded_features,unpadded_word_spelling_length,unpadded_feature_length

