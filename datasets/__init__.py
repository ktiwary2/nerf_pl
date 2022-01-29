from .blender import BlenderDataset
from .llff import LLFFDataset
from .pyredner import PyRednerDataset
from .blender_shadows import BlenderDatasetShadows
from .blender_efficient_sm import BlenderEfficientShadows
from .pyredner2 import PyRednerShadowsDataset

dataset_dict = {'blender': BlenderDataset,
                'llff': LLFFDataset, 
                'pyredner': PyRednerDataset, 
                'shadows': BlenderDatasetShadows,
                'efficient_sm': BlenderEfficientShadows,
                'pyredner2': PyRednerShadowsDataset,
                }