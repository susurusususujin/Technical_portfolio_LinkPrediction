# helper.py
import torch
from torch.nn import Parameter
from torch.nn.init import xavier_normal_
from torch_scatter import scatter_add  # CompGCNConv.compute_norm 에서 사용

def get_param(shape):
    p = Parameter(torch.empty(*shape))
    xavier_normal_(p.data)
    return p

# PyTorch 2.x에서 안전한 FFT 버전
def cconv(a, b):
    n = a.shape[-1]
    return torch.fft.irfft(torch.fft.rfft(a, n=n) * torch.fft.rfft(b, n=n), n=n)

def ccorr(a, b):
    n = a.shape[-1]
    return torch.fft.irfft(torch.conj(torch.fft.rfft(a, n=n)) * torch.fft.rfft(b, n=n), n=n)
