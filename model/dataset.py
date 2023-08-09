import librosa
from utils import mfcc
import os
import pandas as pd
from torch.utils.data import Dataset
import numpy as np

class AsrDataset(Dataset):
    def __init__(
        self,
        scr_file,
        feature_type="discrete",
        feature_file=None,
        feature_label_file=None,
        wav_scp=None,
        wav_dir=None,
        type= 'train'
    ):
        """__init__ _summary_

        Args:
            scr_file (str):  clsp.trnscr
            feature_type (str, optional):  "quantized" or "mfcc". Defaults to "discrete".
            feature_file (str, optional): clsp.trainlbls or clsp.devlbls. Defaults to None.
            feature_label_file (str, optional):  clsp.lblnames. Defaults to None.
            wav_scp (str, optional): clsp.trnwav or clsp.devwav. Defaults to None.
            wav_dir (str, optional): wavforms/. Defaults to None.
            type (str, optional): if . Defaults to 'train'.
        
        if feature_type is 'discrete',
        if feature_file is 'mfcc', you should give wav_scp, wav_dir, and scr_file
        """

        self.feature_type = feature_type
        assert self.feature_type in ["discrete", "mfcc"]
        
        self.type = type
        assert self.type in ['train','test']
        
        if self.feature_type == "discrete":
            self.input_feature, self.output_feature = self.read_train(
                feature_label_file=feature_label_file,
                feature_file=feature_file,
                scr_file=scr_file
            )

        if self.feature_type == "mfcc":

            self.input_feature = self.compute_mfcc(wav_scp,wav_dir)
            with open(scr_file) as f:
                self.output_feature = self._extracted_from_read_train(f)
        
        
    def __len__(self):
        
        return len(self.input_feature)

    def __getitem__(self, idx):

        return self.input_feature[idx], self.output_feature[idx]

    def read_train(
        self,
        feature_label_file=None,
        feature_file=None,
        scr_file=None,
    ):
        # 1. read the feature_label_file
        # feature_dict
        # {'AA': 0, 'AB': 1, 'AC': 2, 'AD': 3,...}
        with open(feature_label_file) as f:
            self.feature_dict = {}
            for i, feat in enumerate(f.readlines()):
                if feat.strip() == "clsp.lblnames":
                    self.feature_dict['<blank>'] = i
                    continue
                self.feature_dict[feat.strip()] = i

        with open(feature_file) as f:
            input_feature = []
            for i, feat in enumerate(f.readlines()):
                if feat.strip() == "clsp.trnlbls":
                    continue

                feat = feat.strip().split()  # 去掉\n 和空格

                input_feature.append([])
                for index in feat:
                    if index in self.feature_dict:
                        input_feature[i - 1].append(self.feature_dict[index])

        # 3. read scr_file

        if scr_file != None:                

            with open(scr_file) as f:
                letter_label = self._extracted_from_read_train(f)
                        # print(total_word)

            return input_feature, letter_label

    def _extracted_from_read_train(self, f):
        # create `letter_dict`
        CHARS = [" ","<sil>","a","b","c","d","e","f","g",
                     "h","i","j","k","l","m","n",
                     "o","p","q","r","s","t",
                     "u","v","w","x","y","z",]

        self.letter_dict = {letter: i for i, letter in enumerate(CHARS)}
        # self.word = []
        self.total_word = []

        result = []

        for i, line in enumerate(f.readlines()):

            if i == 0:  
                continue

            each_word = line.strip()

            # 生成每个单词的字母列表 ['e', 'a', 'c', 'h', '_', 'w', 'o', 'r', 'd']
            letter = ['<sil>'] + list(each_word) + ['<sil>']
            # self.word.append(letter)  
            #  
            result.append([self.letter_dict[j] for j in letter])
            
            if each_word not in self.total_word:
                self.total_word.append(each_word)

        return result

    def get_word(self):
        return self.word
    
    def get_total_word(self):
        return self.total_word
    
    # This function is provided
    def compute_mfcc(self, wav_scp, wav_dir):
        """
        Compute MFCC acoustic features (dim=40) for each wav file.
        :param wav_scp:
        :param wav_dir:
        :return: features: List[np.ndarray, ...]
        """
        features = []
        with open(wav_scp, "r") as f:
            for wavfile in f:
                wavfile = wavfile.strip()
                if wavfile == "clsp.trnwav":  # skip header
                    continue
                
                wav, sr = librosa.load(os.path.join(wav_dir, wavfile), sr=None)
                wav = self._extract_non_zero_features(audio=wav)
                feats = librosa.feature.mfcc(
                    y=wav, sr=16000, n_mfcc=13, hop_length=160, win_length=400
                )
                feats = mfcc.Dynamic_Feature(feats).transpose()
                # feats = self.compute(os.path.join(wav_dir, wavfile))
                # print(feats)
                features.append(feats)

        return features
    
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
    
    # @staticmethod
    # def compute(audio):
    #     fs, sig = mfcc.readAudio(audio)
    #     # print(sig)
    #     # '''--------预处理--------'''
    #     # '''(1)预加重'''
    #     alpha = 0.97
    #     sig = np.append(sig[0], sig[1:] - alpha * sig[:-1])
        
    #     '''(2)分帧'''
    #     frame_len = 400     #25ms
    #     frame_shift = 160   #10ms
    #     N = 256
    #     frame_sig, pad_num = mfcc.enframe(sig,frame_len, frame_shift, fs)
        
    #     # '''(3)加窗'''
    #     window = mfcc.Window(frame_len,'hamming')
    #     frame_sig_win = window * frame_sig

    #     # '''--------stft--------'''
    #     # N = 512
    #     # print(frame_sig_win.shape)
    #     frame_pow = mfcc.stft(frame_sig_win, N ,fs)

    #     # # '''--------Mel 滤波器组--------'''
    #     # '''Filter Bank 特征和MFCC特征提取'''
    #     n_filter = 15   # mel滤波器个数
        
    #     filter_banks,mfcc_bank,_ = mfcc.mel_filter(frame_pow, fs, n_filter, N, mfcc_Dimen=13)

    #     # # # '''去均值'''
    #     # filter_banks -= (np.mean(filter_banks, axis=1)[:,np.newaxis] + np.finfo(float).eps)

    #     # # # '''动态特征提取'''
    #     # mfcc_final = mfcc.Dynamic_Feature(mfcc_bank)
    #     return mfcc_bank