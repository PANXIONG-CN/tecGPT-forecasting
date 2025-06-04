import pickle
import numpy as np
from src.utils.util import DummyScaler

scaler_data = {"tec_scaler": DummyScaler(2911), "sw_scaler": DummyScaler(5)}

with open("./small_test_data/scaler.pkl", "wb") as f:
    pickle.dump(scaler_data, f)

print("Scaler created successfully using DummyScaler from utils")
