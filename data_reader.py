import math
import librosa
import numpy as np
import soundfile as sf
import librosa
import os
import pandas as pd
import os
import numpy as np
import torch
import torch.utils.data as data
import torch.utils.data.distributed
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import pandas as pd
import librosa
import math
import random


class SRDataset(data.Dataset):

    def __init__(self,dir,dir_tp,dir_fp):
        super(SRDataset, self).__init__()
        self.path = dir
        # self.files = os.listdir(self.path)
        # for i in range(len(self.files)):
        #     if '.flac' not in self.files:
        #         del self.files[i]
        self.tp = np.array(pd.read_csv(dir_tp))
        self.fp = np.array(pd.read_csv(dir_fp))
        self.all=np.concatenate((self.tp,self.fp),axis=0)
        # self.all=list(self.tp[:,0])+list(self.fp[:,0])

    def __getitem__(self, index):
        win=3
        y, sr = librosa.load(os.path.join(self.path, self.all[index][0]+'.flac'),sr=None)
        start=self.all[index][3]
        length=(self.all[index][5]-self.all[index][3])
        species=self.all[index][1]
        r=random.uniform(0,abs(length-win))
        if length>win:
            slice=y[int(start+r)*sr:int(start+r+win)*sr]
        else:
            if start-r>=0 and start-r+win<len(y)//sr:
                slice=y[int(start-r)*sr:int(start-r+win)*sr]
            elif start-r<0:
                slice=y[:int(win*sr)]
            elif start-r+win>=len(y)//sr:
                slice=y[-int(win*sr):]

        # slice=slice[:len(slice)//2]
        # print(slice.shape)
        mel_spec = librosa.feature.melspectrogram(slice, n_fft=2048, hop_length=1128, sr=sr,
                                                  power=1.5)
        mel_spec=mel_spec.reshape(1,mel_spec.shape[0],mel_spec.shape[1])
        return mel_spec,species

        # S = librosa.feature.melspectrogram(y=y, sr=sr)

    def __len__(self):
        return len(self.all)