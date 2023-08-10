#  Isolated-Word Speech Recognizer 

* **项目目标**

​		为48个单词的词汇表构建一个孤立词语音识别器。



* **项目要求**

- [x] 修改文件dataset.py中提供的代码以加载数据集。使用每个单词的拼写作为其语音的输出标签序列。选择<sil>符号填充每一面的拼写，让网络为沉默帧输出合适的标签。

- [x] 修改文件model.py中提供的代码，创建PyTorch模型。建议使用LSTM模型，但您可以使用cnn或cnn和LSTM的组合进行实验。模型输出是 Batch × InputLength × NumLetters。<!--这里只使用了LSTM模型进行测试-->

  > `TODO: CRNN模型的性能`

- [x] 使用CTC准则训练PyTorch模型，在训练期间计算一些保留(即验证)集上的CTC损失。

- [x] 创建一个识别器，使用这些概率从48个单词的词汇表中生成最可能的单词。

  > - [x] (a)采用greedy decode方法将每帧中最可能出现的字母作为输出符号，然后“压缩”重复的符号来拼写输出单词。
  >
  >   ```TODO: 1) 采用数据增强```
  >
  >   ​			 ```2) 字节对编码```
  >
  >   ​	 		```3) 采用负样本训练联合损失```
  >
  > - [ ] (b) 采用beam search 方法计算可能的单词。
  >
  >   `正在完善代码部分`
  >
  > - [ ] (c)为每个测试样例计算每个沉默添加词假设的内置CTCLoss，并选择损失最小的词作为该样例的输出。

- [ ] 检查模型在训练数据本身上的准确性，以确保它正确地训练。此外，请为提供的393个测试话语中的每一个提交你的单词假设。

  > 正在调参，改进模型性能中



* **提交内容**

- [x] 构建一个对比系统来探索主要系统的替代方案

> ​		使用MFCC矢量而不是提供的256个离散特性，计算40维mfcc，分析窗口为25ms，步长为10ms(即相邻窗口之间重叠15ms)。然而，它可能有助于调整这些超参数，看看什么最有效。
>
> ​		```# TODO: 调参验证```

- [x] 神经网络训练损失随迭代次数的函数图，以及验证集损失。
- [ ] 对于测试数据中的每个话语，最可能的单词和置信度。
- [x] 源代码，以及关于运行每个模块确切需要哪些文件的大量文档，以及用于运行培训和测试模块的命令行(用法)[Usage](#Usage)。
- [ ] 运行在最新版本GNU/Linux和PyTorch的x86 64机器上。
- [x] README.md

## Content

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



## <span id="rnn.pack_padded_sequence 和 pad_packed_sequence">Usage</span>

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
python decode.py --test_path data/test --waveform_file data/waveforms --model_path ../save_model/asr.ckpt 
```



## Introduction

### 1. 基于LSTM + CTC模型

#### **Input**

```python
# 原始数据
seq = [HH HH HH BH JE IC HG AI AI EL HT JK AE AE CO AI IC JK JK IC CM HP HT HP CM HP JE JE IC BG BG HT AE AE AI IC IC IC DG IC CM JH EJ GV GV BP IA IA IA IA FR FR GN GN GN GN GN GN BE BR DA BR CD HL HL FD CU BW IN BB GC GC GC GC GC BK IZ AE EM IS JE JE HP HT AE CO EU EL CO AE EU CO HG HP HP CM HT CM BG BG JK CO AE EU HT EU BG HT HG AE IC CM BG GQ IC BG JA HV ]

# 将clsp.lblnames构建成词典，然后将seq进行编码
# 词典
{'AA': 1, 'AB': 2, 'AC': 3, 'AD': 4,...}

# 编码
seq = [16 25 48 9 3 65 ...]

# Padding
batch_seq = [[16 25 48 9 3 65 ... 0 0 0 ]
             [16 25 48 9 3 ... 0 0 0 0 0 ]]
```

#### **Target**

```python
# 字符集大小为28，包括26个字母和空白符,以及<sil>字符
character_set = [' ','<sil>','a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'] 

eg.
target = ['<sil>','m','o','n','e','y','<sil>']
target = [1, 14, 16, 15, 6, 26, 1]
```

#### **Model**

​		采用两层双向LSTM，参数均是代码原始给定。采用pack_padded_sequence函数和pad_packed_sequence函数将数据进行压缩和解压缩，消除pad对模型的影响，具体详细说明见[rnn.pack_padded_sequence 和 pad_packed_sequence](#rnn.pack_padded_sequence 和 pad_packed_sequence)。

```python
class LSTM_ASR(nn.Module):
    def __init__(self,
                 input_size=64, 
                 hidden_size=256, 
                 num_layers=2,
                 output_size=28,
                 init = None):
        super().__init__()
      
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(num_embeddings=300,embedding_dim=input_size)
        self.lstm = torch.nn.LSTM(input_size = input_size,
                                    hidden_size = hidden_size,
                                    num_layers = num_layers,
                                    bidirectional = True,
                                    batch_first = True,
                                    dropout = 0.5)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, input_features,input_lengths):
        """forward 
        Args:
            input_features: (N, T) N:Batch size, T:sequence length
            input_lengths: unpaded sequence length

        """
        batch_input_features = self.embed(input_features)        
        # (N, T, C) N:Batch size, T:sequence length C:feature dimension
        
        packed_input = rnn_utils.pack_padded_sequence(
            batch_input_features, input_lengths, batch_first = True, enforce_sorted = False)
        
        packed_output, (_, _) = self.lstm(packed_input)
        # print(f"packed_output.shape:\t {packed_output}")
        output, _ = rnn_utils.pad_packed_sequence(packed_output, batch_first=True)

        H = output[:,:,:self.hidden_size] + output[:,:,self.hidden_size:]
    
        logits = F.log_softmax(self.fc(H),dim=-1)
        
        logits = logits.transpose(0,1)
        return logits
```

#### Save Model

* **asr.ckpt**

```python
## Model
Model(input_size=64, 
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

## Training Para
batch_size=4,
epochs=20,
learning_rate=0.005
```

#### TODO

> 1)  采用不同的解码方案，对比分析
> 2)  调试batch size，学习率，Epoch等参数。



### 2. 基于MFCC + LSTM + CTC 模型

#### Input

​		1） 读入音频，将音频的非静音段作为MFCC的输入。

​		2） MFCC 维度设为13， 将一阶差分和二阶差分与MFCC进行拼接，维度共为39维，再进行去均值和归一化，作为最终的训练集，维度为`[Batch, Seq, MFCC]`。

#### Target

​		去掉<sil>标签，以字母作为标签。

```python
# 字符集大小为28，包括26个字母和空白符,以及<sil>字符
character_set = [' ','<sil>','a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'] 

eg.
target = ['m','o','n','e','y']
target = [14, 16, 15, 6, 26]
```

#### Model

​		采用LSTM，和模型1结构相同，input_size 为 39，hidden_size = 100。

```python
class LSTM_ASR_MFCC(nn.Module):
    def __init__(self,
                 input_size=39,# 48 
                 hidden_size=100, 
                 num_layers=2,
                 output_size=28,
                 init = None):  # sourcery skip: remove-redundant-if
        super().__init__()
        self.hidden_size = hidden_size
        # self.embed = nn.Embedding(num_embeddings=300,embedding_dim=input_size)
        self.lstm = torch.nn.LSTM(input_size = input_size,
                                    hidden_size = hidden_size,
                                    num_layers = num_layers,
                                    bidirectional = True,
                                    batch_first = True,
                                    dropout = 0.5)

        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, input_features,input_lengths):
        """forward _summary_
        Args:
            input_features: (N, T) N:Batch size, T:sequence length
            input_lengths: 
        """
        packed_input = rnn_utils.pack_padded_sequence(input_features, input_lengths, batch_first = True, enforce_sorted = False)
        packed_output, (_, _) = self.lstm(packed_input)
        # print(f"packed_output.shape:\t {packed_output}")
        output, _ = rnn_utils.pad_packed_sequence(packed_output, batch_first=True)

        H = output[:,:,:self.hidden_size] + output[:,:,self.hidden_size:]

        logits = F.log_softmax(self.fc(H),dim=-1)
        
        logits = logits.transpose(0,1)
        return logits
```

#### Save Model

* **asr_mfcc_no_sil.ckpt**

```python
## Model
Model(input_size=35,
     hidden_size=256, 
     num_layers=3,
     output_size=28,)
self.lstm
self.fc

"""
# [seq_len, Batch, mfcc(39)]

## Training Para
batch_size=4,
epochs=20,
learning_rate=0.005
mfcc = 13 采用一阶差分，二阶差分，拼接而成39维度，然后进行取均值归一化。
```

#### TODO

> 1)  【问题】训练不收敛
> 2)  同上
> 3)  测试MFCC维度的影响



## Loss

* **LSTM + CTC Loss**

![asr](D:\Study\Project\project\pic\asr_training.png)



* **MFCC + LSTM + CTC Loss**



```python
# train
> python train_mfcc.py --training_file data/train --waveform_file data/waveforms  --validation_file data/val --save_path ../save_model/asr_mfcc_no_sil.ckpt --save_train_loss ../pic/asr_mfcc_no_sil_training.png --save_val_loss ../pic/asr_mfcc_no_sil_validation.png --epochs 30

# decode
> python decode_mfcc.py --test_path data/val --waveform_file data/waveforms --model_path ../save_model/asr_mfcc_no_sil.ckpt
```

![asr_mfcc_no_sil_training](D:\Study\Project\project\pic\asr_mfcc_no_sil_training.png)

![asr_mfcc_no_sil_validation](D:\Study\Project\project\pic\asr_mfcc_no_sil_validation.png)



## Decode

### **Greedy Decode**

​		贪婪解码（Greedy Decode）是一种序列解码方法，通常用于序列生成任务，例如文本生成、语音识别和机器翻译。在贪婪解码中，每个时间步都选择概率最高的标记（字符、单词等）作为输出，从而构建整个序列。这意味着在每个时间步，模型都选择当前概率最高的标记作为输出，而不考虑后续时间步的概率。

​		步骤：

1. 对于每个时间步，从模型的输出概率分布中选择概率最高的标记。

2. 将所选的标记添加到输出序列中。

3. 在下一个时间步中，重复步骤 1 和 2。

4. 重复直到到达序列的结束或达到最大允许长度

   代码如下

   ```	python
   def greedy_decode(probs, blank_idx, space_idx):
       """decode function
       # TODO: 根据实际输出确定参数类型，完善代码
       # NOTE: 
       
       Args:
           probs (ternsor): shape:(N, T, C)
           blank_idx (int): 
           space_idx (int): 
       
       # 假设我们有一个CTC输出概率矩阵probs，空白符的索引是0，空格符的索引是4
       # 
       >>> probs = [
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
       >>> decoded_result = greedy_decode(probs, blank_idx, space_idx)
       >>> print(decoded_result)  # 输出： "abcd"
       >>>
       
       Returns:
           text: decoded text
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
   ```

   

   #### 【**实验结果**】

   * 基于LSTM+CTC

     Epoch: **40**  Training Acc:  Validation Acc: **38%**

   * 基于MFCC + LSTM + CTC 

     Epoch: **40**   Training Acc:  Validation Acc:

### **Beam Search**

```python
# TODO: Beam Search
```



### 基于CTCLoss 选择最小的损失

```python
# TODO: CTCLoss
```



## *Function Usage*

### <span id="rnn.pack_padded_sequence 和 pad_packed_sequence">rnn.pack_padded_sequence 和 pad_packed_sequence</span>

> `torch.nn.utils.rnn.pack_padded_sequence` 是 PyTorch 中用于处理变长序列的函数，通常用于序列填充（padding）和序列压缩（packing）操作，以在循环神经网络（RNN）中有效地处理变长输入序列。这个函数主要在处理序列数据时非常有用，比如自然语言处理中的文本序列。
>
> 以下是如何使用 `torch.nn.utils.rnn.pack_padded_sequence` 的基本步骤：
>
> 1. 导入 PyTorch：
> ```python
> import torch
> import torch.nn as nn
> ```
>
> 2. 准备数据：
> 假设你有一个变长序列的批量数据，每个序列都需要进行填充以匹配最长序列的长度。你还需要准备一个描述每个序列的实际长度的张量。
>
> ```python
> # 假设 batch_size 为批量大小，seq_lengths 为每个序列的实际长度
> batch_size = 4
> seq_lengths = [10, 8, 6, 5]  # 实际长度从长到短排列
> max_seq_length = max(seq_lengths)
> 
> # 假设 input_data 为模拟的批量输入数据，形状为 (batch_size, max_seq_length, input_size)
> input_data = torch.randn(batch_size, max_seq_length, input_size)
> ```
>
> 3. 创建 Pack 对象：
> ```python
> # 创建一个 Pack 对象，对输入数据进行打包
> packed_input = nn.utils.rnn.pack_padded_sequence(input_data, seq_lengths, batch_first=True)
> ```
>
> - `input_data`：输入数据的张量，形状为 `(batch_size, max_seq_length, input_size)`。
> - `seq_lengths`：一个包含每个序列实际长度的列表或张量。
> - `batch_first`：如果为 `True`，则输入数据的形状应为 `(batch_size, seq_length, input_size)`；如果为 `False`，则形状应为 `(seq_length, batch_size, input_size)`。
>
> 4. 将打包的数据传递给 RNN 模型：
> 将打包的数据传递给 RNN 模型，例如 LSTM。
>
> ```python
> # 假设定义了一个 LSTM 模型
> rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
> 
> # 将打包的数据传递给 LSTM 模型
> packed_output, (h_n, c_n) = rnn(packed_input)
> ```
>
> 5. 解包输出：
> 如果你需要将压缩后的输出解包成正常的张量，可以使用 `torch.nn.utils.rnn.pad_packed_sequence` 函数。
>
> ```python
> # 解包输出，返回解包后的数据和长度
> output_data, unpacked_seq_lengths = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
> ```
>
> `output_data` 的形状将为 `(batch_size, max_seq_length, hidden_size)`。
>
> 通过使用 `pack_padded_sequence` 和相关的函数，你可以在处理变长序列数据时有效地使用循环神经网络。这对于自然语言处理和其他序列数据的任务非常有用。









