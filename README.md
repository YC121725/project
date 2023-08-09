#  Isolated-word Speech Recognizer 

​		这个项目的目标是为48个单词的词汇表构建一个孤立词语音识别器。

```python
D:.
│  Project.pdf
│  README.md
│  split.zip
│  txt.md
│  
├─.vscode
│      launch.json
│      
├─code
│  │  dataset.py
│  │  decode.py
│  │  DecodeCTCLoss.py
│  │  demo.py
│  │  model.py
│  │  test.py
│  │  train.py
│  │  
│  └─__pycache__
│          dataset.cpython-310.pyc
│          model.cpython-310.pyc
│          train.cpython-310.pyc
│          
├─data
│  │  .gitignore
│  │  
│  ├─test
│  │      clsp.devlbls
│  │      clsp.devwav
│  │      clsp.lblnames
│  │      
│  ├─train
│  │      clsp.lblnames
│  │      clsp.trnlbls.kept
│  │      clsp.trnscr.kept
│  │      clsp.trnwav.kept
│  │      
│  ├─val
│  │      clsp.lblnames
│  │      clsp.trnlbls.held
│  │      clsp.trnscr.held
│  │      clsp.trnwav.held
│  │      
│  └─waveforms
└─save_model
       asr.ckpt
```



## Introduction

- **Input**

```python
# 原始数据
seq = [HH HH HH BH JE IC HG AI AI EL HT JK AE AE CO AI IC JK JK IC CM HP HT HP CM HP JE JE IC BG BG HT AE AE AI IC IC IC DG IC CM JH EJ GV GV BP IA IA IA IA FR FR GN GN GN GN GN GN BE BR DA BR CD HL HL FD CU BW IN BB GC GC GC GC GC BK IZ AE EM IS JE JE HP HT AE CO EU EL CO AE EU CO HG HP HP CM HT CM BG BG JK CO AE EU HT EU BG HT HG AE IC CM BG GQ IC BG JA HV ]
```

​		1）将clsp.lblnames构建成词典，然后将seq进行编码。

```python
# 词典
{'AA': 1, 'AB': 2, 'AC': 3, 'AD': 4,...}、
# 编码
seq = [16 25 48 9 3 65 ...  0 0]
batch_seq = [[16 25 48 9 3 65 ... 0 0 0 ]
             [16 25 48 9 3 ... 0 0 0 0 0 ]]
```

* **Target**

```python
# 字符集大小为28，包括26个字母和空白符,以及<sil>字符
character_set = [' ','<sil>','a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'] 

eg.
target = ['<sil>','m','o','n','e','y','<sil>']
target = [1, 14, 16, 15, 6, 26, 1]
```



## model

该项目实现LSTM模型，采用CTCLoss

## Usage

* **train**

```python
cd ./code
python train.py --training_file [your training file]

eg.
cd ./code
python train.py --training_file data/train
python train.py --training_file data/train --validation_file data/val --save_path ../save_model/asr.ckpt
```

* **test**

```python
cd ./code
python test.py
```



## Save Model

```python
# asr.ckpt
## Model
Model(feature_type="discrete", 
    input_size=64, 
     hidden_size=256, 
     num_layers=2,
     output_size=28)

self.embed = nn.Embedding(num_embeddings=300,embedding_dim=input_size)
self.lstm = torch.nn.LSTM(input_size = input_size,
                            hidden_size = hidden_size,
                            num_layers = num_layers,
                            bidirectional = True,
                            batch_first = True,
                            dropout = 0.5)
self.fc = nn.Linear(hidden_size, output_size)

## training
batch_size=4,
epochs=20,
learning_rate=0.005
```



## Training Loss

![asr](D:\Study\Project\project\pic\asr.png)



















