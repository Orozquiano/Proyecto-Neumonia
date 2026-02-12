import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from image.preprocess_img import Preprocessor

def test_preprocess_output_shape():

    pre = Preprocessor()

    # Imagen falsa RGB 1024x1024
    fake_image = np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)

    result = pre.preprocess(fake_image)

    assert result.shape == (1, 512, 512, 1)
