import librosa
import os
import torch
from torch.utils.data import Dataset
import numpy as np
from torch.nn.utils.rnn import pad_sequence


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
    ):
        """
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
            self.feature_dict = {}
            for i, feat in enumerate(f.readlines()):
                if feat.strip() == "clsp.lblnames":
                    self.feature_dict['<blank>'] = i
                    continue
                self.feature_dict[feat.strip()] = i
        
        # read discrete feature
        with open(feature_file) as f:
            input_feature = []
            for i, feat in enumerate(f.readlines()):
                if feat.strip() == "clsp.trnlbls":
                    continue

                feat = feat.strip().split()
                input_feature.append([])
                for index in feat:
                    if index in self.feature_dict:
                        input_feature[i - 1].append(self.feature_dict[index])

            return input_feature
    
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
    
    def _extract_label(self, scr_file):
        # create `letter_dict`
        CHARS = [" ","<sil>","a","b","c","d","e","f","g",
                     "h","i","j","k","l","m","n",
                     "o","p","q","r","s","t",
                     "u","v","w","x","y","z",]

        self.letter_dict = {letter: i for i, letter in enumerate(CHARS)}
        result = []
        with open(scr_file) as f:
            for i, line in enumerate(f.readlines()):
                if i == 0:  
                    continue
                each_word = line.strip()
                # 生成每个单词的字母列表 ['e', 'a', 'c', 'h', '_', 'w', 'o', 'r', 'd']
                letter = ['<sil>'] + list(each_word) + ['<sil>']
                # letter = list(each_word)
                result.append([self.letter_dict[j] for j in letter])
                
        return result
    

    
    
    
    # def compute(self,audio):
        
    #     fs, sig = mfcc.readAudio(audio)
    #     sig = self._extract_non_zero_features(audio=sig)
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

    #     filter_banks,mfcc_bank,_ = mfcc.mel_filter(frame_pow, fs, n_filter, N, mfcc_Dimen=40)

    #     return mfcc.Dynamic_Feature(mfcc_bank,ischafen=True)
    
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