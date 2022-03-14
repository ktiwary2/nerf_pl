import os
from os.path import exists, join, basename

project_name = "pyrender"
if not exists(project_name):
  # clone and install
  !git clone -q https://github.com/mmatl/pyrender.git
  #requirements file gives the wrong pyglet
  #ERROR: pyrender 0.1.23 has requirement pyglet==1.4.0b1, but you'll have pyglet 1.4.0a1 which is incompatible.
  #!cd $project_name && pip install -q -r requirements.txt
  
import sys
sys.path.append(project_name)