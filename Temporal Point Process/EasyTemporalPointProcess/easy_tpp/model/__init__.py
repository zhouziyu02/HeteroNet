# Modified: register only the required TPP model.
from easy_tpp.model.torch_model.torch_basemodel import TorchBaseModel
from easy_tpp.model.torch_model.torch_heteronet import HeteroNet as TorchHeteroNet

__all__ = ['TorchBaseModel', 'TorchHeteroNet']
