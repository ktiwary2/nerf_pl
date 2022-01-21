from .blender import BlenderDataset
from .llff import LLFFDataset
from .pyredner import PyRednerDataset
from .blender_shadows import BlenderDatasetShadows

dataset_dict = {'blender': BlenderDataset,
                'llff': LLFFDataset, 
                'pyredner': PyRednerDataset, 
                'shadows': BlenderDatasetShadows,
                }