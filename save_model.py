import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
import os

loaded_best_model = keras.models.load_model("/home/lenovo/cow_ml/cow_skin_check/model_07-0.97.h5")

MODEL_DIR = "/home/lenovo/cow_ml/cow_skin_check/model/" #tempfile.gettempdir()
version = 1
export_path = os.path.join(MODEL_DIR, str(version))
print('export_path = {}\n'.format(export_path))

keras.models.save_model(
    loaded_best_model,
    export_path,
    overwrite=True,
    include_optimizer=True,
    save_format=None,
    signatures=None,
    options=None
)