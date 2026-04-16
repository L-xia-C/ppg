import os
import pandas as pd
import numpy as np
import pycwt,pywt
import cv2
import matplotlib.pyplot as plt
from scipy import signal
from skimage import io, transform
from scipy.signal import savgol_filter,butter, lfilter
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder
import argparse
from tqdm import tqdm
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a
def smooth_signal(data, sample_rate, window_length=None, polyorder=3):
    '''smooths given signal using savitzky-golay filter

    Function that smooths data using savitzky-golay filter using default settings.

    Functionality requested by Eirik Svendsen. Added since 1.2.4

    Parameters
    ----------
    data : 1d array or list
        array or list containing the data to be filtered

    sample_rate : int or float
        the sample rate with which data is sampled

    window_length : int or None
        window length parameter for savitzky-golay filter, see Scipy.signal.savgol_filter docs.
        Must be odd, if an even int is given, one will be added to make it uneven.
        default : 0.1  * sample_rate

    polyorder : int
        the order of the polynomial fitted to the signal. See scipy.signal.savgol_filter docs.
        default : 3

    Returns
    -------
    smoothed : 1d array
        array containing the smoothed data

    Examples
    --------
    Given a fictional signal, a smoothed signal can be obtained by smooth_signal():

    >>> x = [1, 3, 4, 5, 6, 7, 5, 3, 1, 1]
    >>> smoothed = smooth_signal(x, sample_rate = 2, window_length=4, polyorder=2)
    >>> np.around(smoothed[0:4], 3)
    array([1.114, 2.743, 4.086, 5.   ])

    If you don't specify the window_length, it is computed to be 10% of the
    sample rate (+1 if needed to make odd)
    >>> import heartpy as hp
    >>> data, timer = hp.load_exampledata(0)
    >>> smoothed = smooth_signal(data, sample_rate = 100)

    '''

    if window_length == None:
        window_length = sample_rate // 10

    if window_length % 2 == 0 or window_length == 0: window_length += 1
    if window_length <= polyorder:
        polyorder = polyorder - 1
    smoothed = savgol_filter(data, window_length=window_length,
                             polyorder=polyorder)

    return smoothed
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y
def wavelet_filter_ccnu(data_int, fs, freq_l, freq_h):
    dt = 1 / fs
    sig = data_int
    nn = len(data_int)
    dj = 1 / 32  # dj = 1 / 12
    s0 = 2 * dt  # 20 * dt  # -1
    J = np.log2(nn * dt / s0) / dj  # 10/dj#-1-1  #
    wvn = pycwt.Morlet(6)  # 'morlet'
    normalize = True

    t = np.arange(0, dt * nn, dt)
    x = (sig - sig.mean()) / sig.std()
    cwtm, scales, freqs, coi, fft, _ = pycwt.cwt(x, dt, dj, s0, J, wvn)

    freq_l = freq_l  # Hz
    freq_h = freq_h  # Hz
    freq_indin = np.where((freqs >= freq_l) & (freqs <= freq_h))[0]

    icwt1 = pycwt.icwt(cwtm[freq_indin], scales[freq_indin], dt, dj, wvn)
    return icwt1
def get_roi_frequency_range(heart_rate):
    low_freq = (heart_rate - 10) / 60  # Lower frequency (Hz)
    high_freq = (heart_rate + 10) / 60  # Higher frequency (Hz)
    return low_freq, high_freq

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--number', type=int, default=1)
    return parser.parse_args()

args = get_args()
subject_number = args.number
plt.rcParams['figure.dpi'] = 112
plt.rcParams['savefig.dpi'] = 112

path = os.getcwd()

Input_file = rf'D:\XZX\PPG\PPG\ppg_dalida\subject_{subject_number}.csv'

s_dir_output_specturm = r"D:\XZX\PPG\PPG\cwt_picture\ppg_dalia/"
folder = os.path.exists(s_dir_output_specturm)
if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
    os.makedirs(s_dir_output_specturm)  # makedirs 创建文件时如果路径不存在会创建这个路径
    print("---  new folder...  ---")
else:
    print("---  There is this folder!  ---")
df_mmWave = pd.read_csv(Input_file, index_col=False)
x = df_mmWave.iloc[:, 2:].values
y = df_mmWave.loc[:, 'label'].values
y = np.reshape(y, (y.shape[0], 1))
y = np.array(y)
Nx_row = x.shape[0]
Nx_col = x.shape[1]

fs             = 32
T              = 8
f_l            = 0.6
f_h            = 2.8

N_pixels       = 224
s_spectrum_flag="cwt"
flag_scalar = 1  # 归一化
scalar = MinMaxScaler(feature_range=(0, 1))  # 加载函数

if s_spectrum_flag == "cwt":
    fs_bandwidth_plot = [f_l, f_h]
    fs_bw = round(fs_bandwidth_plot[1] - fs_bandwidth_plot[0],2)
    t = np.arange(0, T, 1.0 / fs)  # time

    # wavelet
    wavename = 'cgau8'
    totalscal = 1024 * 4
    fs_resolution = fs / (totalscal * 2)  # freqency resolution
    ind_fs_h = int(np.ceil(fs_bandwidth_plot[1] / fs_resolution))
    ind_fs_l = int(np.ceil(fs_bandwidth_plot[0] / fs_resolution))
    ind_fs = np.arange(ind_fs_l, ind_fs_h, 1, "int")  # index
    N_ind_fs = len(ind_fs)
    fc = pywt.central_frequency(wavename)
    cparam = 2 * fc * totalscal
    scales = cparam / np.arange(totalscal, 1, -1)

    t_downsampling = np.arange(0, T, T / N_pixels)

    frequencies_downsampling = np.arange(fs_bandwidth_plot[0], fs_bandwidth_plot[1], fs_bw / N_pixels)
    for i in tqdm(range(Nx_row),desc=f"subject_{subject_number}:"):
        # print(f"Processing row {i+1}/{Nx_row}")

        data = x[i, :]
        [cwtmatr, frequencies] = pywt.cwt(data, scales, wavename, 1.0 / fs)

        full_region  = abs(cwtmatr[-ind_fs, :])
        if flag_scalar == 1:
            norm_full_region  = scalar.fit_transform(full_region)  # 归一化,按时间,行
        else:
            norm_full_region  = full_region

        # 下采样：原始归一化图像和提亮后的ROI图像均下采样到N_pixels x N_pixels
        temp_2 = transform.resize(norm_full_region, (N_pixels, N_pixels))
        # 绘制并保存图像（保存的是包含提亮ROI区域的完整图像）
        plt.contourf(t_downsampling, frequencies_downsampling, temp_2)
        plt.axis('off')
        ss = s_dir_output_specturm + str(df_mmWave.iloc[i, 0]) + f"[{y[i].item():.2f}]" + ".jpg"
        plt.savefig(ss, bbox_inches='tight', pad_inches=0)
        plt.close()
