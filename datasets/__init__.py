from .blender import BlenderDataset
from .llff import LLFFDataset
from .pyredner import PyRednerDataset

dataset_dict = {'blender': BlenderDataset,
                'llff': LLFFDataset, 
                'pyredner': PyRednerDataset, 
                }