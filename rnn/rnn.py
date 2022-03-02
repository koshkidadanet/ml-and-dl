import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import re

from tensorflow.keras.layers import Dense, SimpleRNN, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer

with open('text.txt','r',encoding='utf-8') as f:
    text = f.read()
    text = text.replace('\ufeff','')
    text = re.sub(r'^[А-я]', ' ',text)