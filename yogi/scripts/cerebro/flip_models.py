#!/usr/bin/env python3
from yogi.db import session
from yogi.models import Model


models_to_flip = [
    'cerebro-paired-mono-aug',
    'cerebro-aug',
    'cerebro-all-behaviors-aug',
    'cerebro-1-cam-1-light-aug',
    'cerebro-4-cam-1-light-aug'
]

for model_name in models_to_flip:
    flipped_model_name = model_name + '-flipped'
