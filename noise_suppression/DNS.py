import torch
from torch import nn
from torch.utils.data import Dataset

from lib.component import Components, DoNothing
from lib.dataset import GpuMultiTFDataset, mlt_store_to, mlt_load_from

class DNSnoise(nn.Module):
    """Meta Denoiser (dns64)\n
    pip install denoiser==0.1.5
    """
    def __init__(self):
        super().__init__()
        from denoiser import pretrained
        self.model = pretrained.dns64(pretrained=True)
        self.model.eval()

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            y = self.model(x)
        y = y.squeeze_(dim=0)
        return y
    
def DNS_cache_set(dataset:Dataset, device:str, sample_rate:int, cache_path:str) -> Dataset:
    assert sample_rate == 16000
    gpu_set = GpuMultiTFDataset(
        dataset=dataset, 
        tfs=[Components(transforms=[
            DNSnoise()
        ])],
        device=device
    )
    index_file_name = 'metaInfo.csv'
    mlt_store_to(
        dataset=gpu_set, root_path=cache_path, index_file_name=index_file_name,
        data_tfs=[Components(transforms=[DoNothing()])]
    )

    return mlt_load_from(root_path=cache_path, index_file_name=index_file_name)