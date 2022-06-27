import sys
sys.path.append('../uvtracking')

import numpy as np

from realtime_pipeline import Stage
import uvtracking

class DetectDyeStage(Stage):

    def __init__(self):
        super(DetectDyeStage, self).__init__()
        self.threshold = 20

    def consume(self, image):
        # mask = uvtracking.detect_dye(image, hue_range=(0.0, 0.65))
        mask = image[:, :, 0] > self.threshold
        mask = mask.astype(np.uint8) * 255
        self.emit(mask)
