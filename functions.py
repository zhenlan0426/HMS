import pandas as pd, numpy as np
from torch.utils.data import Dataset,DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from mamba_ssm.models.mixer_seq_simple import create_block,_init_weights
from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn


""" Data """
class eegData(Dataset):
    def __init__(self, df, dataPath, dataFolder, seqLen=10000):
        # dataFolder save local vs global normalized eeg input
        self.df = df
        self.seqLen = seqLen
        self._load_eegs(dataPath,dataFolder)
        
    def _load_eegs(self,dataPath,dataFolder):
        self.eegs = [np.load(dataPath+dataFolder+'/'+str(id)+'.npy')\
                      for id in self.df.eeg_id.tolist()]
        
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self,idx):
        out = self.df.iloc[idx,1:].tolist()
        offset,target = self.sample_time(out)
        eeg = self.eegs[idx][offset*200:offset*200+self.seqLen]
        return torch.tensor(eeg,dtype=torch.float32),target
    
    @staticmethod
    def sample_time(out):
        time = np.random.randint(0,len(out[0]))
        offset, *target = [o[time] for o in out]
        return int(offset),torch.tensor(target,dtype=torch.float32)

class eegDataXonly(Dataset):
    # random start time rather than centering around target
    def __init__(self, df, dataPath, dataFolder, seqLen=10000):
        # dataFolder save local vs global normalized eeg input
        self.df = df
        self.seqLen = seqLen
        self._load_eegs(dataPath,dataFolder)
        
    def _load_eegs(self,dataPath,dataFolder):
        self.eegs = [np.load(dataPath+dataFolder+'/'+str(id)+'.npy')\
                      for id in self.df.eeg_id.tolist()]
        self.eegs_len = [x.shape[0] for x in self.eegs]
        
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self,idx):
        start = np.random.randint(0,max(self.eegs_len[idx]-self.seqLen-1,1))
        eeg = self.eegs[idx][start:start+self.seqLen]
        return torch.tensor(eeg,dtype=torch.float32)
    


""" Model """
class Config(object):
    def __init__(self, data_dict):
        for key, value in data_dict.items():
            setattr(self, key, value)

    def to_dict(self):
        return vars(self)

    def __str__(self):
        attributes = vars(self)
        return ','.join(f"{key}-{value}" for key, value in attributes.items())

fixed_config = {
    "ssm_cfg": {},
    "residual_in_fp32": True,
    "fused_add_norm": True,
    "n_class":6,
}

class seq2seqModel(nn.Module):
    def __init__(
        self,
        config,
        initializer_cfg=None,
        device=None,
        dtype=None,
    ) -> None:
        
        self.config = config
        d_model = config.d_model
        n_layer = config.n_layer
        ssm_cfg = config.ssm_cfg
        rms_norm = config.rms_norm
        residual_in_fp32 = config.residual_in_fp32
        fused_add_norm = config.fused_add_norm
        normOverChannel = config.normOverChannel
        in_channels = config.in_channels
        seqLen = config.seqLen
        n_class = config.n_class
        factory_kwargs = {"device": device, "dtype": dtype}
        
        super().__init__()

        self.norm = (nn.LayerNorm if not rms_norm else RMSNorm)(in_channels if normOverChannel else seqLen)
        self.input2model = nn.Linear(in_channels,d_model)
        self.model2input = nn.Linear(d_model,in_channels)
        self.classHead = nn.Linear(d_model,n_class)
        self.act = nn.SiLU()
        self.layers = nn.ModuleList(
            [
                create_block(
                    d_model,
                    ssm_cfg=ssm_cfg,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    **factory_kwargs,
                )
                for i in range(n_layer)
            ]
        )
        
        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(d_model)
        self.autoLoss = nn.L1Loss()
        self.classLoss = nn.KLDivLoss(reduction="batchmean")
        self.readOut = lambda x: x[:,-1]
        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )
        
    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.backbone.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)

    def forward(self, input_ids, target=None, IsTrain=True, inference_params=None):
        # input_ids has shape b,l,d
        # if target is None, in predict mode, return logit
        # else return autoregressive loss, target_loss
        transpose = partial(torch.transpose, dim0=1, dim1=2)
        if self.config.normOverChannel:
            hidden_states = self.norm(input_ids)
        else:
            hidden_states = transpose(self.norm(transpose(input_ids)))
        hidden_states = self.act(self.input2model(hidden_states))
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params
            )
            
        # Set prenorm=False here since we don't need the residual
        fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
        hidden_states = fused_add_norm_fn(
            hidden_states,
            self.norm_f.weight,
            self.norm_f.bias,
            eps=self.norm_f.eps,
            residual=residual,
            prenorm=False,
            residual_in_fp32=self.config.residual_in_fp32,
        )
        
        if IsTrain:
            if target is None:
                input_predict = self.model2input(hidden_states) # b,l,d
                autoL = self.autoLoss(input_predict[:,:-1],input_ids[:,1:])
                return autoL
            else:
                classlog = F.log_softmax(self.classHead(self.readOut(hidden_states)),-1) # b, k
                classL = self.classLoss(classlog, target)
                return classL
        else:            
            classlog = F.softmax(self.classHead(self.readOut(hidden_states)),-1) # b, k
            return classlog