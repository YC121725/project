import itertools
import numpy as np
from scipy.io import wavfile
from scipy.fftpack import dct
import matplotlib.pyplot as plt


'''-----------预处理-------------'''
def readAudio(wav_file,iscut = False):
    '''    
    预处理——读取音频
    -----------------
    `wav_file`:   音频名称
    `iscut`：     如果需要截取音频输入时长，如果不需要截取音频，输入False

    '''
    fs, sig = wavfile.read(wav_file)
    #fs = 8000Hz
    sig = sig[:int(iscut * fs)] if iscut else sig
    return fs,sig

'''-----------特征提取-------------'''
def enframe(sig=np.array([]),frame_len=2048, frame_shift=1024, fs=8000):
    '''
    分帧
    ----------------
    主要是计算对应下标,并重组
    >>> enframe(sig,0.025,0.01,8000)
    >>>     return frame_sig
    
    Parameters
    ----------------
    `sig`:            信号
    `frame_len_s`:    帧长，s 一般25ms
    `frame_shift_s`:  帧移，s 一般10ms
    `fs`:             采样率，hz

    return 
    ----------------
    二维list，一个元素为一帧信号
    '''
    # np.round 四舍五入
    # np.ceil 向上取整
    sig_n = len(sig)

    frame_len_n, frame_shift_n = int(round(frame_len)), int(round(frame_shift))
    # print('frame_len_n:',frame_len_n,'\t','frame_shift_n:',frame_shift_n)

    num_frame = 0
    while num_frame * frame_shift_n + frame_len_n <= sig_n:
        num_frame+=1
    if (num_frame - 1) * frame_shift_n + frame_len_n != sig_n:
        num_frame+=1
    # print('num_frame',num_frame)
    # print('length of Sig:',sig_n,'\t','num_frame:',num_frame)

    pad_num = frame_shift_n * (num_frame-1) + frame_len_n - sig_n   # 待补0的个数
    pad_zero = np.zeros(int(pad_num))    # 补0
    pad_sig = np.append(sig, pad_zero)   

    index = np.arange(0,frame_len_n)                           # 一行    
    frame_index = np.tile(index,(num_frame,1))                 # frame num个行
    '''
    print(frame_index)
    [[  0   1   2 ... 197 198 199]
    [  0   1   2 ... 197 198 199]
    [  0   1   2 ... 197 198 199]
    ...
    [  0   1   2 ... 197 198 199]
    [  0   1   2 ... 197 198 199]
    [  0   1   2 ... 197 198 199]]
    '''
    frame_inner_index = np.arange(0,num_frame)*frame_shift_n   # frame num 个值
    frame_inner_index_expand = np.expand_dims(frame_inner_index,1)
    '''
    print(frame_inner_index_expand)
    [[    0]
    [   80]
    ...
    [27760]]
    '''
    each_frame_index = frame_inner_index_expand + frame_index
    each_frame_index = each_frame_index.astype(int,copy=False)
    '''
    print(each_frame_index)
    [[    0     1     2 ...   197   198   199]
    [   80    81    82 ...   277   278   279]
    [  160   161   162 ...   357   358   359]
    ...
    [27600 27601 27602 ... 27797 27798 27799]
    [27680 27681 27682 ... 27877 27878 27879]
    [27760 27761 27762 ... 27957 27958 27959]]
    '''
    frame_sig = pad_sig[each_frame_index]
    # print(frame_sig.shape)
    return frame_sig,pad_num

def enframe2(sig=np.array([]),frame_len_s=0.025, frame_shift_s=0.01, fs=8000):
    '''
    分帧
    ----------------
    主要是计算对应下标,并重组
    >>> enframe(sig,0.025,0.01,8000)
    >>>     return frame_sig
    
    Parameters
    ----------------
    `sig`:            信号
    `frame_len_s`:    帧长，s 一般25ms
    `frame_shift_s`:  帧移，s 一般10ms
    `fs`:             采样率，hz

    return 
    ----------------
    二维list，一个元素为一帧信号
    '''
    # np.round 四舍五入
    # np.ceil 向上取整
    sig_n = len(sig)

    frame_len_n, frame_shift_n = int(round(fs * frame_len_s)), int(round(fs * frame_shift_s))
    # print('frame_len_n:',frame_len_n,'\t','frame_shift_n:',frame_shift_n)

    num_frame = 0
    while num_frame * frame_shift_n + frame_len_n <= sig_n:
        num_frame+=1
    if (num_frame - 1) * frame_shift_n + frame_len_n != sig_n:
        num_frame+=1
    # print('num_frame',num_frame)
    # print('length of Sig:',sig_n,'\t','num_frame:',num_frame)

    pad_num = frame_shift_n * (num_frame-1) + frame_len_n - sig_n   # 待补0的个数
    pad_zero = np.zeros(int(pad_num))    # 补0
    pad_sig = np.append(sig, pad_zero)   

    index = np.arange(0,frame_len_n)                           # 一行    
    frame_index = np.tile(index,(num_frame,1))                 # frame num个行
    '''
    print(frame_index)
    [[  0   1   2 ... 197 198 199]
    [  0   1   2 ... 197 198 199]
    [  0   1   2 ... 197 198 199]
    ...
    [  0   1   2 ... 197 198 199]
    [  0   1   2 ... 197 198 199]
    [  0   1   2 ... 197 198 199]]
    '''
    frame_inner_index = np.arange(0,num_frame)*frame_shift_n   # frame num 个值
    frame_inner_index_expand = np.expand_dims(frame_inner_index,1)
    '''
    print(frame_inner_index_expand)
    [[    0]
    [   80]
    ...
    [27760]]
    '''
    each_frame_index = frame_inner_index_expand + frame_index
    each_frame_index = each_frame_index.astype(int,copy=False)
    '''
    print(each_frame_index)
    [[    0     1     2 ...   197   198   199]
    [   80    81    82 ...   277   278   279]
    [  160   161   162 ...   357   358   359]
    ...
    [27600 27601 27602 ... 27797 27798 27799]
    [27680 27681 27682 ... 27877 27878 27879]
    [27760 27761 27762 ... 27957 27958 27959]]
    '''
    return pad_sig[each_frame_index]


def deframe(frame=np.array([]),pad_num = 0,frame_len_s=0.025, frame_shift_s=0.01, fs=8000):
    '''
    将分帧后的信号转化为原始信号
    将enframe直接截取拼接
    '''
    X = np.array([])
    for i in range(frame.shape[0]):
        if i < frame.shape[0]-1:
            X = np.hstack((X,frame[i][:int(frame_shift_s*fs)]))
        if i == frame.shape[0]-1:
            X = np.hstack((X,frame[i][:int(frame_len_s*fs-pad_num)]))
    return X
    
def deframe2(frame:np.ndarray,pad_num:int = 0,frame_len:int = 200, frame_shift:int = 10):

    '''
    将分帧后的信号转化为原始信号,只针对50% overlap的数据，重叠部分进行求和取平均
    '''
    '''
    a = np.array([[1,1,1,1,1,1,1,1],
              [2,2,2,2,2,2,2,2],
              [3,3,3,3,3,3,0,0]])
    '''
    X = np.zeros((frame.shape[0],int(frame_shift*(frame.shape[0]+1))))
    # print(X.shape)
    for i in range(frame.shape[0]):
        X[i][int(frame_shift*i):int(frame_shift*i+frame_len)] = frame[i]
    '''
    [[1. 1. 1. 1. 1. 1. 1. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
    [0. 0. 0. 0. 2. 2. 2. 2. 2. 2. 2. 2. 0. 0. 0. 0.]
    [0. 0. 0. 0. 0. 0. 0. 0. 3. 3. 3. 3. 3. 3. 0. 0.]]
    '''
    # 列求和
    # print('---------Concat----------')
    # print(X)
    # print('-------------------------')
    sumX = np.sum(X,axis=0)

    one = np.ones(frame_shift)
    two = np.ones(int(frame_shift*(frame.shape[0]-1)))
    two = np.multiply(two,2)
    divX = np.hstack((one,two,one))
    # 除以2
    div = np.divide(sumX,divX)
    # [1.  1.  1.  1.  1.5 1.5 1.5 1.5 2.5 2.5 2.5 2.5 3.  3.  0.  0. ]
    #去掉pad_num
    # [1.  1.  1.  1.  1.5 1.5 1.5 1.5 2.5 2.5 2.5 2.5 3.  3. ]
    return div if pad_num == 0 else div[:-pad_num]
    
def Window(frame_len,Window_Type:str,isshow_fig=False):
    '''
    三种窗函数
    -------
    `hamming`
    `hanning`
    `blackman`
    '''
    if Window_Type == 'blackman':
        window = np.blackman(int(round(frame_len)))
    elif Window_Type == 'hamming':
        window = np.hamming(int(round(frame_len)))
    elif Window_Type == 'hanning':
        window = np.hanning(int(round(frame_len)))
    if isshow_fig: 
        _extracted_from_Window_16(window, Window_Type)
    return window


# TODO Rename this here and in `Window`
def _extracted_from_Window_16(window, Window_Type):
    plt.figure(figsize=(10, 5))
    plt.plot(window)
    plt.grid(True)
    plt.xlim(0, 200)
    plt.ylim(0, 1)
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    plt.title(Window_Type)
    plt.show()

def stft(frame_sig, nfft=512 ,fs=8000,isshow_fig = False):
    """
    短时傅里叶变换
    ---------------
    >>> stft(frame_sig,512,8000)
    >>> return frame_pow
    
    Parameters
    ---------------
    `frame_sig`: 分帧后的信号
    `nfft`: fft点数
    `fs`: 采样率
    `isshow_fig`:显示第一帧的stft图像
    
    说明
    ---------------
    >>> np.fft.fft vs np.fft.rfft
    >>> np.fft.fft
    >>>     return nfft
    >>> np.fft.rfft
    >>>     return nfft/2 + 1
    """
    frame_spec = np.fft.rfft(frame_sig, nfft)
    # 幅度谱
    fhz = np.linspace(0,frame_spec.shape[1],frame_spec.shape[1])
    fhz  = fhz*fs/nfft

    frame_mag = np.abs(frame_spec)
    # 功率谱
    frame_pow = (frame_mag ** 2) * 1.0 / nfft
    if isshow_fig:
        _extracted_from_stft(fhz, frame_pow)
    return frame_pow


# TODO Rename this here and in `stft`
def _extracted_from_stft(fhz, frame_pow):
    plt.figure(figsize=(10, 5))
    plt.rcParams['font.sans-serif']=['SimHei']
    plt.rcParams['axes.unicode_minus']=False
    plt.plot(fhz,frame_pow[0])
    plt.grid(True)
    plt.xlabel('F/Hz')
    plt.ylabel('功率谱')
    plt.title('短时傅里叶变换')
    plt.show()

def mel_filter(frame_pow, fs, n_filter, nfft, mfcc_Dimen = 12,isshow_fig = False):
    '''
    mel滤波器系数计算
    ----------------
    >>> mel_filter(frame_pow,fs=8000,n_filter=15,nfft=512,mfcc_Dimen = 13)
    >>>     return filter_bank,mfcc,Mel_Filters
    
    Parameters
    ------------
    `frame_pow`:  分帧信号功率谱
    `fs`:         采样率 hz
    `n_filter`:   滤波器个数
    `nfft`:       fft点数,通常为512
    `mfcc_Dimenson`: 取多少DCT系数，通常为12-13

    '''
    mel_min = 0     # 最低mel值
    mel_max = 2595 * np.log10(1 + fs / 2.0 / 700)               # 最高mel值，最大信号频率为 fs/2
    mel_points = np.linspace(mel_min, mel_max, n_filter + 2)    # n_filter个mel值均匀分布与最低与最高mel值之间
    fhz = 700 * (10 ** (mel_points / 2595.0) - 1)               # mel值对应回频率点，频率间隔指数化
    k = np.floor(fhz * (nfft + 1) / fs)                         # 对应到fft的点数比例上
    freq = np.linspace(0,int(nfft/2),int(nfft/2+1),)*fs/nfft
    # 求mel滤波器系数
    fbank = np.zeros((n_filter, int(nfft / 2 + 1)))
    for m in range(1, 1 + n_filter):
        f_left = int(k[m - 1])     # 左边界点
        f_center = int(k[m])       # 中心点
        f_right = int(k[m + 1])    # 右边界点
        for j in range(nfft):
            if f_left <= j <= f_center:
                fbank[m-1,j] = (j - f_left) / (f_center - f_left)
            elif f_center <= j <= f_right:
                fbank[m-1,j] = (f_right - j) / (f_right - f_center)

    if isshow_fig:
        plt.figure(figsize=(10, 5))
        for i in range(n_filter):
            _extracted_from_mel_filter(freq, fbank, i)
        plt.show()   

    # mel 滤波
    filter_banks = np.dot(frame_pow, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
    # 取对数
    filter_banks = 20 * np.log10(filter_banks)  # dB
    # 求取MFCC特征
    mfcc = dct(filter_banks, type=2,axis=1, norm='ortho')[:, 1:mfcc_Dimen+1]
    return filter_banks.T,mfcc.T,fbank


# TODO Rename this here and in `mel_filter`
def _extracted_from_mel_filter(freq, fbank, i):
    plt.plot(freq,fbank[i,:])
    plt.rcParams['font.sans-serif']=['SimHei']
    plt.rcParams['axes.unicode_minus']=False
    plt.xlabel('f/Hz')
    plt.ylabel('幅值')
    plt.title('Mel滤波器组')

def Dynamic_Feature(mfcc,cutframe = True):
    '''
    动态特征提取
    -----------
    包含升倒谱系数;三阶差分运算

    (1)提升倒谱系数可以提高在噪声环境下的识别效果
    >>> Dynamic_Feature(mfcc)  # mfcc [] 
    >>>     return mfcc_final 
    
    (2)将经过升倒谱后的MFCC进行一阶、二阶差分运算、三阶差分运算，构成动态特征
    自动切除前后两帧
    
    Parameters
    -----------
    `mfcc`：通常为[MFCC,feature_num]的行向量
    `cutframe`：是否切除前后各两帧
    ''' 
    J = mfcc.T
    K = [1+ 22/2*np.sin(np.pi*i/22) for i in range(J.shape[1])]
    K /= np.max(K)
    feat = np.zeros((J.shape[0],J.shape[1]))

    for i, j in itertools.product(range(J.shape[0]), range(J.shape[1])):
        feat[i][j] = J[i][j]*K[j]

    '''--------3阶差分运算--------
        一阶差分
    '''
    dtfeat = np.zeros(feat.shape)
    for i in range(2,dtfeat.shape[0]-2):
        dtfeat[i,:] = -2*feat[i-2,:]-feat[i-1,:]+feat[i+1,:]+2*feat[i+2,:]
    dtfeat = dtfeat/10

    '''二阶差分'''
    dttfeat = np.zeros(feat.shape)
    for i in range(2,dttfeat.shape[0]-2):
        dttfeat[i,:] = -2*dtfeat[i-2,:]-dtfeat[i-1,:]+dtfeat[i+1,:]+2*dtfeat[i+2,:]
    dttfeat = dttfeat/10

    '''三阶差分'''
    dtttfeat = np.zeros(feat.shape)
    for i in range(2,dtttfeat.shape[0]-2):
        dtttfeat[i,:] = -2*dttfeat[i-2,:]-dttfeat[i-1,:]+dttfeat[i+1,:]+2*dttfeat[i+2,:]
    dtttfeat = dtttfeat/10   

    '''拼接'''
    mfcc_final = np.concatenate((feat.T,dtfeat.T,dttfeat.T,dtttfeat.T))

    '''是否去掉前后各两帧'''
    if cutframe:
        mfcc_final = mfcc_final[2:-2,:]
    return mfcc_final

def CMVN(feature):
    '''
    倒谱均值方差归一化
    ''' 
    return (feature - np.mean(feature, axis=1)[:, np.newaxis]) / (
        np.std(feature, axis=1) + np.finfo(float).eps
    )[:, np.newaxis]

'''-----------绘图----------'''
def plot_time(sig, fs,title):
    '''
    绘制时间图
    -----------
    '''
    time = np.arange(0, len(sig)) * (1.0 / fs)
    plt.figure(figsize=(10, 5))
    plt.plot(time, sig)
    plt.rcParams['font.sans-serif']=['SimHei']
    plt.rcParams['axes.unicode_minus']=False
    plt.title(title)
    plt.xlabel('Time(s)')
    plt.ylabel('Amplitude')
    plt.grid()
    plt.show()
    
def plot_freq(sig, sample_rate,title='频域图', nfft=512):
    '''
    绘制频率图
    ------------
    '''
    xf = np.fft.rfft(sig, nfft)
    print('Number of fft:',(len(xf)-1)*2)      # 257个点
    xfp = -20 * np.log10(np.abs(xf))           # np.clip(np.abs(xf), 1e-20, 1e100)
    freqs = np.linspace(0, int(sample_rate/2), int(nfft/2) + 1)
    plt.figure(figsize=(10, 5))
    plt.plot(freqs, xfp)
    plt.rcParams['font.sans-serif']=['SimHei']
    plt.rcParams['axes.unicode_minus']=False
    plt.title(title)
    plt.xlabel('Freq(hz)')
    plt.ylabel('dB')
    plt.grid()
    plt.show()

def plot_spectrogram(ylabel,title,*args, **kwargs):
    '''
    绘制语谱图
    ------------
    
    example
    -----------
    >>> plot_spectrogram(ylabel,title, [2,2], spec1,spec2,...)
    '''
    
    fig = plt.figure(figsize=(10, 5))
    # heatmap = plt.pcolor(spec)
    # fig.colorbar(mappable=heatmap)
    index = 1
    for _ in range(args[0][0]):
        for _ in range(args[0][1]):
            # print(int(str(args[0][0])+str(args[0][1])+str(index)))
            plt.subplot(int(str(args[0][0])+str(args[0][1])+str(index)))
            fig.colorbar(plt.pcolor(args[index]))
            plt.rcParams['font.sans-serif']=['SimHei']
            plt.rcParams['axes.unicode_minus']=False
            plt.xlabel('Num of Frame')
            plt.ylabel(ylabel)
            plt.title(title)
            index+=1
    index = 0
    # plt.show()
    
def calculate_filter(sig,fs):
    '''
    计算 filter_banks,mfcc
    '''
    alpha = 0.97
    sig = np.append(sig[0], sig[1:] - alpha * sig[:-1])

    '''(2)分帧'''
    frame_len_s = 0.025    #25ms
    frame_shift_s = 0.01   #10ms
    frame_sig,padnum = enframe(sig,frame_len_s, frame_shift_s, fs)

    '''(3)加窗'''
    window = Window(frame_len_s,fs,'hamming')
    frame_sig_win = window * frame_sig

    '''--------stft--------'''
    N = 512
    frame_pow = stft(frame_sig_win, N ,fs)

    n_filter = 15   # mel滤波器个数
    filter_banks,mfcc,_ = mel_filter(frame_pow, fs, n_filter, N,mfcc_Dimen=13)
    return filter_banks,mfcc
    # plot_spectrogram(filter_banks,'Filter Banks','Filter Banks特征')

def tocomplex(real,imag):
    '''
    real和imag矩阵是相等大小的
    
    '''
    cplex = np.zeros_like(real,dtype=complex)
    for i, j in itertools.product(range(real.shape[0]), range(real.shape[1])):
        cplex[i][j] = complex(real[i][j],imag[i][j])
    return cplex
    