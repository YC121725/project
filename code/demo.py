import torch
import torch.nn as nn
from model import LSTM_ASR
import numpy as np
import torch.nn.functional as F
# BATCH_SIZE = 16
# SEQUENCE_SIZE = 128
# X = torch.randint(1,257,(BATCH_SIZE,SEQUENCE_SIZE)) 
# print(X.shape)
# input_lengths = torch.randint(low = 20,high=100,size=(BATCH_SIZE,))

# model = LSTM_ASR()
# out = model(X,input_lengths)
# a = [1,2,3,4,5,6,7,8,9,10]
# # [3 , 6]
# print([0]*3 + a[3:6-1]+ [0]*(10-6+1))

# batch = 4
# padded_features = torch.zeros((4,10))
# x = [[1,2,3],[4,5,6,7],[7,9,10,2,5,],[7,8]]
# for i in range(batch):
#     _list = torch.from_numpy(np.array(x[i]))
#     pad_length = torch.zeros(10-len(_list))
#     _list = torch.hstack((_list,pad_length))
#     padded_features[i] = torch.add(padded_features[i],_list)

# print(padded_features)
# padded_features = torch.tensor(np.zeros((batch,10)))
'''------------- chatgpt -----------------'''

## NOTE: CTCLoss 解释

# # 假设你的数据集中有5个类别，包括一个用于表示CTC空白标记
# C = 5
# blank_index = C - 1

# # 0 1 2 3 4(空白) 

# # 假设你有一批训练数据和对应的目标序列
# N = 4
# T = 50  # 序列长度       不等长需要<PAD>
# # input_size = 40   # 输入特征维度   量化维度256



# # 假设模型输出为(T, N, C)，模型的输出序列长度应与目标序列长度一致
# model_outputs = torch.randn(T, N, C).log_softmax(2).detach().requires_grad_()
# # input_lengths = torch.full((N,),T,dtype=torch.long)
# input_lengths = torch.tensor([26,22,34,24])
# # target_lengths = torch.randint(low=10, high=20, size=(N,), dtype=torch.long)  # 每个序列的目标长度
# target_lengths = torch.tensor([13, 11, 17, 12])


# print(input_lengths)
# # print(sum(target_lengths))
# print(target_lengths)
# # 随机生成目标序列，长度为T，范围为[0, C-2]，最后一个位置用于CTC空白标记

# targets = torch.randint(low=0, high=C, size=(N,max(target_lengths)))

# print(targets)


# # blank_index = torch.tensor(blank_index)

# # # # 实例化CTC损失函数
# ctc_loss = nn.CTCLoss()

## NOTE: 官方关于CTCLoss 解释

# # # # 计算CTC损失
# loss = ctc_loss(model_outputs, targets, input_lengths=input_lengths, target_lengths=target_lengths)
# print("CTC损失:", loss.item())


# ctc_loss = nn.CTCLoss()   # T N C
# log_probs = torch.randn(50, 4, 20).log_softmax(2).detach().requires_grad_()

# input_lengths = torch.full((4,), 50, dtype=torch.long)

# target_lengths = torch.randint(10,30,(4,), dtype=torch.long) 
# print(target_lengths)

# targets = torch.randint(1, 20, (sum(target_lengths),), dtype=torch.long)  # 
# print(targets)

# loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)
# # loss.backward()
# print(loss.item())

## NOTE: ChatGPT 关于word recognizer 的解释
""" 
在CTC（Connectionist Temporal Classification）模型中，解码部分的"word recognizer"通常是指将CTC模型的输出序列映射为最终文本序列的组件。
这个组件负责对CTC输出序列进行解码，去除重复字符、空白符，并还原出原始的文本内容。

在解码部分，常见的"word recognizer"算法是基于贪婪算法或束搜索（Beam Search）。
贪婪算法简单高效，但可能会产生一些错误，因为它在每个时间步仅选择概率最高的字符。
束搜索则在一定程度上弥补了这个问题，允许保留多个候选路径，从中选择最终最优解。

以下是一个简化的Python代码示例，用贪婪算法来实现CTC解码过程：
"""
# 假设CTC模型输出为probs，是一个T x C的概率矩阵，T为时间步数，C为字符集大小
# 字符集中包括了所有的字符，以及空白符
# import numpy as np

# def greedy_decode(probs, blank_idx, space_idx):
#     # 将概率矩阵转换为NumPy数组
#     probs = np.array(probs)

#     # 初始化解码结果和前一个字符
#     decoded_text = []
#     prev_char = None

#     # 遍历时间步
#     for t in range(probs.shape[0]):
#         # 获取当前时间步的最高概率字符的索引
#         max_prob_idx = np.argmax(probs[t])
#         print(f"max_prob_idx: {max_prob_idx}")
#         # 如果当前字符是空白符或与前一个字符相同（重复字符），跳过
#         if max_prob_idx == blank_idx or max_prob_idx == prev_char:
#             continue

#         # 如果当前字符是空格，将其映射为" "
#         if max_prob_idx == space_idx:
#             decoded_text.append(" ")
#         else:
#             # 将字符索引映射为实际字符，并添加到解码结果中
#             decoded_text.append(chr(max_prob_idx + ord('a')))
#             print(decoded_text)

#         # 更新前一个字符
#         prev_char = max_prob_idx

#     # 将解码结果连接成最终文本
#     final_text = ''.join(decoded_text)

#     return final_text

# # 假设我们有一个CTC输出概率矩阵probs，空白符的索引是0，空格符的索引是4
#         # blank a b c space
# probs = [  
#     [0.1, 0.2, 0.05, 0.1, 0.2],   # t=0  
#     [0.2, 0.05, 0.1, 0.1, 0.3],   # t=1
#     [0.1, 0.3, 0.2, 0.1, 0.1],    # t=2
#     [0.15, 0.1, 0.3, 0.2, 0.05],  # t=3
#     [0.05, 0.2, 0.1, 0.2, 0.1],   # t=4
# ]

# blank_idx = 0
# space_idx = 4

# decoded_result = greedy_decode(probs, blank_idx, space_idx)
# print(decoded_result)  # 输出： "abcde"
# import torch
# import torch.nn as nn
# import numpy as np
# def ctc_beam_search(logits, beam_width=10):
#     T, C = logits.shape

#     # Beam Search算法初始化
#     beam_probs = [0.0] * beam_width
#     beam_results = [[],] * beam_width

#     beam_probs[0] = logits[0, 0]
#     beam_results[0] = [0]  # 0表示blank标签

#     for t in range(1, T):
#         new_beam_probs = [0.0] * beam_width
#         new_beam_results = [[] for _ in range(beam_width)]

#         # Beam Search算法的扩展步骤
#         for b in range(beam_width):
#             prev_prob = beam_probs[b]
#             prev_result = beam_results[b]

#             choices = [
#                 (prev_prob, prev_result + [0]),  # 保持blank
#             ]

#             for c in range(1, C):  # 添加新标签或扩展已有标签
#                 prob = logits[t, c]
#                 if c == prev_result[-1]:  # 连续相同字符
#                     new_prob = prev_prob + prob
#                 else:
#                     new_prob = prev_prob + prob
#                     choices.append((new_prob, prev_result + [c]))

#             # Beam宽度剪枝，保留概率最高的beam_width个结果
#             choices.sort(key=lambda x: x[0], reverse=True)
#             for i in range(min(beam_width, len(choices))):
#                 new_beam_probs[i] = choices[i][0]
#                 new_beam_results[i] = choices[i][1]

#         beam_probs = new_beam_probs
#         beam_results = new_beam_results

#     # 返回Beam Search结果，选择概率最高的序列
#     best_result_idx = np.argmax(beam_probs)
#     best_result = beam_results[best_result_idx][1:]  # 去除开头的blank标签

#     return best_result


# # 虚拟的CTC模型
# class CTCModel(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super(CTCModel, self).__init__()
#         self.rnn = nn.RNN(input_size, hidden_size, bidirectional=True, batch_first=True)
#         self.fc = nn.Linear(2 * hidden_size, output_size)

#     def forward(self, x):
#         out, _ = self.rnn(x)
#         out = self.fc(out)
#         return out

# # 虚拟的输入数据
# seq_length = 10
# num_classes = 5
# input_size = 20
# hidden_size = 32
# output_size = num_classes + 1  # 加1是为了包含blank标签

# inputs = torch.randn(1, seq_length, input_size)  # 假设batch size为1

# # 初始化虚拟的CTC模型
# model = CTCModel(input_size, hidden_size, output_size)

# # 假设模型输出为logits，形状为(seq_length, num_classes)
# logits = model(inputs)

# # 使用Beam Search解码
# decoded_output = ctc_beam_search(logits[0])  # 解码结果是一个列表，包含标签序列

# print("模型输出的Logits:\n", logits)
# print("解码后的输出结果:\n", decoded_output)

# NOTE: guanyu RNN工具的说明
# import torch
# import torch.nn as nn
# import torch.nn.utils.rnn as rnn_utils

# # 假设有一个batch的序列数据
# seq1 = torch.tensor([1, 2, 3])
# seq2 = torch.tensor([4, 5])
# seq3 = torch.tensor([6, 7, 8, 9])
# sequences = [seq1, seq2, seq3]

# # 使用pad_sequence将其填充为等长序列
# padded_sequences = rnn_utils.pad_sequence(sequences, batch_first=True)
# print("Padded Sequences:\n", padded_sequences)

# # 使用pack_padded_sequence将填充后的序列压缩为压缩序列
# lengths = [len(seq) for seq in sequences]
# packed_sequences = rnn_utils.pack_padded_sequence(padded_sequences, lengths, batch_first=True,enforce_sorted=False)
# print("Packed Sequences:\n", packed_sequences)

# # 使用pad_packed_sequence将压缩序列解压缩为填充后的序列
# unpacked_sequences, _ = rnn_utils.pad_packed_sequence(packed_sequences, batch_first=True)
# print("Unpacked Sequences:\n", unpacked_sequences)


## NOTE: 实际应用中

# import torch
# import torch.nn as nn
# import torch.nn.utils.rnn as rnn_utils

# class LSTM_ASR(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim):
#         super(LSTM_ASR, self).__init__()
#         self.rnn = nn.LSTM(input_dim, hidden_dim, batch_first=True)
#         self.fc = nn.Linear(hidden_dim, output_dim)

#     def forward(self, input_features, input_lengths):
#         packed_input = rnn_utils.pack_padded_sequence(input_features, input_lengths, batch_first=True)
#         packed_output, (hidden, cell) = self.rnn(packed_input)
#         output, output_lengths = rnn_utils.pad_packed_sequence(packed_output, batch_first=True)
#         logits = F.log_softmax(self.fc(output),dim=-1)
#         logits = logits.transpose(0,1)
#         return logits

# 假设语音信号特征的维度是40，字符集大小为28（包括26个字母和空白符）
# input_dim = 40
# hidden_dim = 256
# output_dim = 28

# # 假设我们有一个批次的语音信号特征表示，其中有两个样本，特征序列长度分别是100和80
# seq1 = torch.randint(1,200,(100,))
# seq2 = torch.randint(1,200,(80,))

# batch_input_features = rnn_utils.pad_sequence([seq1, seq2], batch_first=True)
# # print(batch_input_features.shape)
# # 假设我们有对应的文本序列标签，长度分别是30和20（字符数）
# target = torch.tensor([[15,  9, 12, 25, 0, 0],
#                        [2,  5,  6, 15, 18,  5]])  # 文本序列标签
# target = torch.tensor([15,  9, 12, 25, 2,  5,  6, 15, 18,  5])  # 文本序列标签


# # # 计算每个样本的实际长度
# input_lengths = torch.tensor([100, 80])
# target_lengths = torch.tensor([4, 6])

# # 初始化语音识别模型
# model = LSTM_ASR(input_size=input_dim,hidden_size=hidden_dim,  output_size=output_dim)

# # 计算模型的输出
# logits = model(batch_input_features, input_lengths)

# print(logits.shape)
# # # # 计算CTC损失
# ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)

# loss = ctc_loss(logits, target, input_lengths, target_lengths)
# print(loss)
"""-------------绘图---------------"""
# import matplotlib
# import seaborn as sns
# import matplotlib.pyplot as plt
# import numpy as np

# 生成虚拟的损失值数据
# epochs = np.arange(1, 9)
# loss_values = [0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01]

# # 设置 Seaborn 风格
# sns.set(style="whitegrid")
# sns.set_palette("colorblind")  # 使用色盲友好的调色板

# # plt.rc('text',usetex=True)
# # 创建一个绘图窗口
# plt.figure(figsize=(10, 6))

# # 使用 Seaborn 绘制损失曲线
# sns.lineplot(x=epochs, y=loss_values, marker='o', color='b', label='Loss', linewidth=2)

# # 添加数据标签
# for i, txt in enumerate(loss_values):
#     plt.annotate(f"{txt:.2f}", (epochs[i], loss_values[i]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=10, color='black')

# # 添加标题和标签，包括使用 LaTeX 数学字体
# # plt.title(r'$\textbf{Training Loss Curve}$', fontsize=16)
# # plt.xlabel(r'$\textbf{Epoch}$', fontsize=14)
# # plt.ylabel(r'$\textbf{Loss}$', fontsize=14)

# # 添加网格线，调整线条风格
# plt.grid(True, alpha=0.3, linestyle='--')

# # 调整坐标轴刻度
# plt.xticks(epochs, fontsize=12)
# plt.yticks(np.arange(0, 0.6, 0.1), fontsize=12)

# # 添加图例，调整图例位置
# plt.legend(fontsize=12, loc='upper right')

# # 调整图形边距
# plt.tight_layout()

# # 添加背景色
# ax = plt.gca()
# ax.set_facecolor((0.95, 0.95, 0.95))

# # 添加边框和阴影
# for spine in ax.spines.values():
#     spine.set_visible(False)
# ax.spines['left'].set_visible(True)
# ax.spines['bottom'].set_visible(True)
# ax.spines['left'].set_color('black')
# ax.spines['bottom'].set_color('black')
# ax.spines['left'].set_linewidth(0.5)
# ax.spines['bottom'].set_linewidth(0.5)

# # 去掉顶部和右侧刻度
# ax.xaxis.tick_bottom()
# ax.yaxis.tick_left()

# # 显示图形
# plt.show()

# import matplotlib.pyplot as plt
# import numpy as np

# # 示例数据，替换成你的实际训练损失和准确率数据
# epochs = np.arange(1, 11)
# train_loss = np.array([0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1, 0.08, 0.07, 0.06])
# accuracy = np.array([0.75, 0.8, 0.85, 0.87, 0.9, 0.92, 0.94, 0.95, 0.96, 0.97])

# # 创建图像
# fig, ax1 = plt.subplots(figsize=(10, 6))

# # 绘制训练损失曲线
# color = 'tab:blue'
# ax1.set_xlabel('Epochs')
# ax1.set_ylabel('Loss', color=color)
# ax1.plot(epochs, train_loss, color=color,label='Training Loss')
# ax1.tick_params(axis='y', labelcolor=color)

# # 创建第二个y轴
# ax2 = ax1.twinx()

# # 绘制准确率曲线
# color = 'tab:green'
# ax2.set_ylabel('Accuracy', color=color)
# ax2.plot(epochs, accuracy, color=color, label='Accuracy')
# ax2.tick_params(axis='y', labelcolor=color)

# # 添加图例
# lines, labels = ax1.get_legend_handles_labels()
# lines2, labels2 = ax2.get_legend_handles_labels()
# ax2.legend(lines + lines2, labels + labels2, loc='upper left')

# # 设置标题
# ax1.set_title('Training Loss and Accuracy')

# # 显示图像
# plt.show()

