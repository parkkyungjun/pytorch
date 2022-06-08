"""Predict
"""
from modules.utils import load_yaml
from modules.datasets import *

from modules.trainer import Trainer
from datetime import datetime, timezone, timedelta
import numpy as np
import random
import os
import torch
import pandas as pd
from models.utils import get_model, EMA
import warnings
warnings.filterwarnings('ignore')

# Config
PROJECT_DIR = os.path.dirname(__file__)
predict_config = load_yaml(os.path.join(PROJECT_DIR, 'config', 'predict_config.yml'))

# Serial
train_serial = predict_config['TRAIN']['train_serial']
kst = timezone(timedelta(hours=9))
predict_timestamp = datetime.now(tz=kst).strftime("%Y%m%d_%H%M%S")
predict_serial = train_serial + '_' + predict_timestamp

# Predict directory
PREDICT_DIR = os.path.join(PROJECT_DIR, 'results', 'predict', predict_serial)
os.makedirs(PREDICT_DIR, exist_ok=True)

# Train config
RECORDER_DIR = os.path.join(PROJECT_DIR, 'results', 'train', train_serial)
train_config = load_yaml(os.path.join(RECORDER_DIR, 'train_config.yml'))

# SEED
torch.manual_seed(predict_config['PREDICT']['seed'])
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(predict_config['PREDICT']['seed'])
random.seed(predict_config['PREDICT']['seed'])

# Gpu
os.environ['CUDA_VISIBLE_DEVICES'] = str(predict_config['PREDICT']['gpu'])

if __name__ == '__main__':

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data
    data_loader = BuildDataLoader(num_labels=train_config['MODEL']['num_labels'], dataset_path=train_config['DIRECTORY']['dataset'],
                                  batch_size=predict_config['PREDICT']['batch_size'])
    _, _, _, test_loader = data_loader.build(supervised=False)

    # Load model
    model = get_model(model_name=train_config['TRAINER']['model'], num_classes=train_config['MODEL']['num_labels'],
                      output_dim=train_config['MODEL']['output_dim']).to(device)
    checkpoint = torch.load(os.path.join(RECORDER_DIR, 'model.pt'))
    model.load_state_dict(checkpoint['model'])
    ema = EMA(model, 0.99)  # Mean teacher model

    # Trainer
    trainer = Trainer(model=model,
                      ema=ema,
                      data_loader=data_loader,
                      optimizer=None,
                      device=device,
                      logger=None,
                      config=train_config['TRAINER'],
                      interval=None)

    sample_submission_df = pd.read_csv(os.path.join(predict_config['DIRECTORY']['sample_submission'], 'sample_submission.csv'))
    trainer.inference(test_loader=test_loader, save_path=os.path.join(PREDICT_DIR, 'submission.csv'), sample_submission=sample_submission_df)
