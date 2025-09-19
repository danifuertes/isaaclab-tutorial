import sys
import skrl
import torch
import random
import numpy as np
from packaging import version
from isaaclab.app import AppLauncher

def parse_args(parser, train=False):
    
    # Append AppLauncher cli args
    AppLauncher.add_app_launcher_args(parser)
    
    # Parse arguments
    if train:
        opts, hydra_args = parser.parse_known_args()
        sys.argv = [sys.argv[0]] + hydra_args   # Clear out sys.argv for Hydra
    else:
        opts = parser.parse_args()
    
    # Always enable cameras to record video
    if opts.video:
        opts.enable_cameras = True
    assert not opts.video, "Video recording is not supported yet"
    
    # Launch omniverse app
    launcher = AppLauncher(opts)
    
    # Check skrl version
    skrl_version(opts)
    return opts, launcher

def skrl_version(opts):
    
    SKRL_VERSION = "1.4.2"
    if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
        skrl.logger.error(
            f"Unsupported skrl version: {skrl.__version__}. "
            f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
        )
        exit()

def set_seed(seed):
    """
    Set random seed for reproducibility.

    Args:
        seed (int): seed value.
    """
    random.seed(seed)  # Python random module.
    np.random.seed(seed)  # Numpy module.
    torch.manual_seed(seed)  # Torch CPU
    torch.cuda.manual_seed(seed)  # Torch GPU
    torch.cuda.manual_seed_all(seed)  # Torch multi-GPU.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
