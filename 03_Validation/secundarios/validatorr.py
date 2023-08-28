from pycaret.datasets import get_data
from pycaret.classification import *
from pycaret.classification import ClassificationExperiment
import pandas as pd

data = pd.read_csv('abalone_dataset.csv')
# data = get_data('abalone_dataset.csv')

functional_api = setup(data, target = 'type', session_id = 123, use_gpu=True)

oop_api  = ClassificationExperiment()
oop_api .setup(data, target = 'Class variable', session_id = 123, use_gpu=True)

# functional API
best_functional_api = functional_api.compare_models()

# OOP API
best_oop_api = oop_api.compare_models()