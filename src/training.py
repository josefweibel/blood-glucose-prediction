import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from tqdm.notebook import tqdm
import os
import matplotlib.pyplot as plt
import yaml


DATA_TIME_INTERVAL = 5 # min


device = torch.device('cpu')

if torch.cuda.is_available():
  device = torch.device('cuda')
elif torch.backends.mps.is_available():
  device = torch.device('mps')


class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x, hidden_prev=None): # x.shape = (batch_size, 18)
        batch_size = x.shape[0]
        x = x.unsqueeze(-1) # x.shape = (batch_size, 18, 1)
        x, hidden_prev = self.rnn(x, hidden_prev) # x.shape = (batch_size, 18, hidden_size)
        hidden_size = x.shape[-1]
        x = x.reshape(-1, hidden_size) # x.shape = (batch_size * 18, hidden_size)
        x = self.linear(x) # x.shape = (batch_size * 18, 1)
        x = x.reshape(batch_size, -1) # x.shape = (batch_size, 18)

        return x, hidden_prev


def load_config(config_name):
    if not os.path.exists(f'./config/{config_name}.yaml'):
        raise KeyError(f'./configs/{config_name}.yaml does not exist')

    with open(f'./config/{config_name}.yaml') as stream:
        config = yaml.safe_load(stream)

    if 'architecture' not in config:
        raise KeyError('no architecture provided in ' + config_name)

    if 'type' not in config['architecture']:
        raise KeyError('no architecture type provided in ' + config_name)

    if 'horizons' not in config:
        raise KeyError('no horizons provided in ' + config_name)

    return config

def build_model(config):
    params = {**config['architecture']}
    del params['type']

    if config['architecture']['type'] == 'rnn':
        return RNNModel(**params, input_size=1)
    else:
       raise KeyError('unknwon architecture ' + config['architecture']['type'])

def get_criterion(config):
    if config['loss'] == 'mse':
        return nn.MSELoss()
    else:
       raise KeyError('unknwon loss function ' + config['loss'])


def train(config_name):
    print('👉 loading config')

    config = load_config(config_name)

    print('👉 loading data')
    train_data = pd.read_csv('./data/train_processed.csv').sort_values('5minute_intervals_timestamp')
    val_data = pd.read_csv('./data/val_processed.csv').sort_values('5minute_intervals_timestamp')

    n_train = int(config['horizons']['train'] / DATA_TIME_INTERVAL)
    n_pred = int(config['horizons']['pred'] / DATA_TIME_INTERVAL)

    model = build_model(config)
    model = model.to(device)

    print('👉 training model')
    model.train()

    optimizer = optim.Adam(model.parameters(), config['lr'], weight_decay=0.)
    criterion = get_criterion(config)

    losses = []
    for epoch in tqdm(range(config['epochs'])):
        X = []
        for i in range(config['samples_per_subject']):
            for subject in train_data['subject'].unique():
                subject_data = train_data[train_data['subject'] == subject]

                start_idx = np.random.choice(np.arange(len(subject_data) - n_train - n_pred + 1))

                x = subject_data['cbg'].iloc[start_idx:start_idx + n_train + n_pred].values
                X.append(torch.Tensor(x))

        X = torch.vstack(X).to(device)
        Y, _ = model(X)

        loss = criterion(Y.flatten(), X.flatten())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.detach().item())
        # TODO set loss

    print('👉 saving model')
    torch.save(model.state_dict(), f'./models/{config_name}.pt')
    np.save(f'./models/{config_name}_meta.npy', np.array(losses))

    return model

