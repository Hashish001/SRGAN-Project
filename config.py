import os

class Config:
    # Paths
    MODEL_PATH = "SRGAN Generator Model.pth"
    
    # Image processing
    SCALE_FACTOR = 4
    
    # Upload settings
    MAX_BATCH_SIZE = 10
    DEFAULT_COLOR_ENHANCE = 1.2
    DEFAULT_SHARPNESS_ENHANCE = 1.1
    DEFAULT_CONTRAST_ENHANCE = 1.05

config = Config()