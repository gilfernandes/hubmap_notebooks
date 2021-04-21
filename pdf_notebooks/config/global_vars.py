import multiprocessing
from pathlib import Path
import torch

SLICES_PATH = Path('/home/jupyter/ds_cache')
RESIZE = 2
TILE_SHAPE = 768
WINDOW = TILE_SHAPE * 2
OVERLAP = 32
THRESHOLD = 100
ENCODER_NAME = "efficientnet-b7"
# NUM_WORKERS = multiprocessing.cpu_count() - 2
NUM_WORKERS = 4
ENCODER_WEIGHTS = 'imagenet'
CLASSES = 2
CHANNELS = 3

LABEL_SMOOTH = 0.01
GRAD_ACCU_STEPS = 1
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'