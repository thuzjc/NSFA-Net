# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 15:32:59 2021

@author: phantom
"""
import numpy as np
# import re
# import stft
# import seaborn as sns
from scipy.signal import butter, lfilter
from scipy import signal
import mne
import matplotlib.pyplot as plt
import sys
# import pywt

sys.path.append("..")


def butter_bandstop_filter(data, lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq

    i, u = butter(order, [low, high], btype='bandstop')
    y = lfilter(i, u, data)
    return y


def butter_highpass_filter(data, cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    y = lfilter(b, a, data)
    return y


def getSpectral_STFT(data, fs=256):
    # (7680, 23)
    #   print("stft input {}".format(data.shape))
    # 	fs=256
    lowcut = 117
    highcut = 123

    y = butter_bandstop_filter(data.T, lowcut, highcut, fs, order=6)
    lowcut = 57
    highcut = 63
    y = butter_bandstop_filter(y, lowcut, highcut, fs, order=6)

    cutoff = 1
    y = butter_highpass_filter(y, cutoff, fs, order=6)

    Pxx = signal.spectrogram(y, nfft=256, fs=256, return_onesided=True, noverlap=192)[2]  # (23, 129, 34)
    Pxx = np.transpose(Pxx, (0, 2, 1))  # (23, 34, 129)
    Pxx = np.concatenate((Pxx[:, :, 1:57],
                          Pxx[:, :, 64:117],
                          Pxx[:, :, 124:]), axis=-1)  # (23, 34, 114)

    # 归一化(33, 129, 23)
    stft_data = (10 * np.log10(Pxx) - (10 * np.log10(Pxx)).min()) / (10 * np.log10(Pxx)).ptp()
    # print("stft result {}".format(stft_data.shape))  # (1, 95, 114)
    return stft_data  # (23, 59, 114)


# def getSpectral_Morlet(data, fs=256):
#     if len(data.shape) > 1:
#         data = data.squeeze()
#     data = data.transpose()
#     # print("data=", data.shape)
#     # print("len(data)=", len(data))
#     cwtm_list = []
#     cwtm_list_all = []
#     # for i in range(len(data)):
#     #     w = 6.
#     #     freq = np.linspace(1, fs / 2, 100)  # (100,)
#     #     widths = w * fs / (2 * freq * np.pi)  # (100,)
#     #     cwtm = signal.cwt(data[i], signal.morlet2, widths, w=w)
#     #     cwtm_list.append(cwtm)
#     # cwtm_list = np.array(cwtm_list)
#     # print("cwtm_list=", cwtm_list.shape)

#     for i in range(len(data)):
#         wavename = "morl"  # 小波函数 cgau8
#         totalscal = 2     # totalscal是对信号进行小波变换时所用尺度序列的长度(通常需要预先设定好)
#         fc = pywt.central_frequency(wavename)  # 计算小波函数的中心频率
#         cparam = 2 * fc * totalscal  # 常数c  cparam/np.arange(totalscal, 1, -1)
#         scales = cparam/np.arange(totalscal, 1, -1)  # 为使转换后的频率序列是一等差序列，尺度序列必须取为这一形式（也即小波尺度）
#         [cwtmatr, frequencies] = pywt.cwt(data[i], scales, wavename, 1.0/fs)  # 连续小波变换模块
#         # print("cwtmatr_shape=", cwtmatr.shape)
#         # print("cwtmatr=", cwtmatr)
#         # print("frequencies=", frequencies)
#         # print("shape1=", len([cwtmatr, frequencies]))
#         # print("shape2=", len([cwtmatr, frequencies][0]))
#         cwtm_list.append(cwtmatr.squeeze().reshape(30, 256))
#     cwtm_list = np.array(cwtm_list)
#     print("cwtm_list=", cwtm_list.shape)

#     # t, dt = np.linspace(0, 1, 200, retstep=True)#t.shape(200,) dt=0.005025125628140704
#     # fs = 1/dt #199
#     # w = 6.
#     # sig = np.cos(2*np.pi*(50 + 10*t)*t) + np.sin(40*np.pi*t)#(200,)
#     # freq = np.linspace(1, fs / 2, 100)  # (100,)
#     # widths = w * fs / (2 * freq * np.pi)  # (100,)
#     # cwtm = signal.cwt(data, signal.morlet2, widths, w=w)  # (100, 200) (freq, time)
#     # plt.pcolormesh(t, freq, np.abs(cwtm), cmap='viridis', shading='gouraud')
#     # plt.show()
#     return cwtm_list
