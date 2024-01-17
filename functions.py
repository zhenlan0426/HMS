import pandas as pd, numpy as np
from torch.utils.data import Dataset,DataLoader
import torch


""" Data """
class eegData(Dataset):
    def __init__(self, df, dataPath):
        self.df = df
        self._load_eegs(dataPath)
        
    def _load_eegs(self,dataPath):
        self.eegs = [pd.read_parquet(dataPath+'train_eegs/'+str(id)+'.parquet').values\
                      for id in self.df.eeg_id.tolist()]
        
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self,idx):
        out = self.df.iloc[idx,1:].tolist()
        offset,target = self.sample_time(out)
        eeg = self.eegs[idx][offset*200:(offset+50)*200]
        return torch.tensor(eeg,dtype=torch.float32),target
    
    @staticmethod
    def sample_time(out):
        time = np.random.randint(0,len(out[0]))
        offset, *target = [o[time] for o in out]
        return int(offset),torch.tensor(target,dtype=torch.float32)