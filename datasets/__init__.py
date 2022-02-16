from .blender import BlenderDataset
from .llff import LLFFDataset
from .blender_shadows import BlenderVariableLightDataset

dataset_dict = {'blender': BlenderDataset,
                'llff': LLFFDataset, 
                'blender_light_sm': BlenderVariableLightDataset}