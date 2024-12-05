import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm.notebook import tqdm
import os
import yaml
import json
from itertools import chain

DATA_TIME_INTERVAL = 5 # min


device = torch.device('cpu')

if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')

class RNNBaseModel(nn.Module):
    def __init__(self, type, input_size, hidden_size, num_layers=1, rnn_dropout=0., fc_layer_dropout=0.):
        super(RNNBaseModel, self).__init__()
        hidden_size = hidden_size if isinstance(hidden_size, list) else [hidden_size]
        fc_layer_dropout = fc_layer_dropout if isinstance(fc_layer_dropout, list) else [fc_layer_dropout] * len(hidden_size)

        rnn_component = self.__get_rnn_component(type)
        self.rnn = rnn_component(
            input_size=input_size,
            hidden_size=hidden_size[0],
            num_layers=num_layers,
            batch_first=True,
            dropout=rnn_dropout,
        )
        self.linear = nn.Sequential(
            *chain.from_iterable((nn.Linear(hidden_size[i - 1], hidden_size[i]), nn.ReLU(), nn.Dropout(fc_layer_dropout[i])) for i in range(1, len(hidden_size))),
            nn.Linear(hidden_size[-1], 1)
        )

    def __get_rnn_component(self, type):
        if type == 'rnn':
            return nn.RNN
        elif type == 'lstm':
            return nn.LSTM
        elif type == 'gru':
            return nn.GRU

        raise KeyError('unknown architecture type ' + type)

    def forward(self, x, hidden_prev=None): # x.shape = (batch_size, 18)
        batch_size = x.shape[0]
        x, hidden_prev = self.rnn(x, hidden_prev) # x.shape = (batch_size, 18, hidden_size)
        hidden_size = x.shape[-1]
        x = x.reshape(-1, hidden_size) # x.shape = (batch_size * 18, hidden_size)
        x = self.linear(x) # x.shape = (batch_size * 18, 1)
        x = x.reshape(batch_size, -1) # x.shape = (batch_size, 18)

        return x, hidden_prev


def load_config(config_name):
    if not os.path.exists(f'./config/{config_name}.yaml'):
        raise KeyError(f'./config/{config_name}.yaml does not exist')

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
    if config['architecture']['type'] in ['rnn', 'lstm', 'gru']:
        return RNNBaseModel(**config['architecture'], input_size=len(config['features']))
    else:
       raise KeyError('unknwon architecture ' + config['architecture']['type'])

def get_criterion(config):
    if config['loss'] == 'mse':
        return nn.MSELoss()
    else:
       raise KeyError('unknwon loss function ' + config['loss'])

def get_normalisation_statistics(config, train_data):
    if config['normalisation'] == 'none':
        return {'normalisation': 'none'}
    elif config['normalisation'] == 'z-score':
        return {'normalisation': 'z-score', 'mean': train_data.mean(axis=(0, 1)), 'std': train_data.std(axis=(0, 1))}
    elif config['normalisation'] == 'min-max':
        return {'normalisation': 'min-max', 'min': train_data.min(dim=1).values.min(dim=0).values, 'max': train_data.max(dim=1).values.max(dim=0).values}
    else:
        raise KeyError('unknwon normalisation ' + config['normalisation'])

def normalise(data, statistics):
    # print('statistics', statistics)
    if statistics['normalisation'] == 'none':
        return data
    elif statistics['normalisation'] == 'z-score':
        mean = statistics['mean'][0:data.shape[2]].to(data.device)
        std = statistics['std'][0:data.shape[2]].to(data.device)
        return (data - mean) / std
    elif statistics['normalisation'] == 'min-max':
        min = statistics['min'][0:data.shape[2]].to(data.device)
        max = statistics['max'][0:data.shape[2]].to(data.device)
        return (data - min) / (max - min)
    else:
        raise KeyError('unknwon normalisation ' + statistics['normalisation'])

def denormalise(data, statistics):
    if statistics['normalisation'] == 'none':
        return data
    elif statistics['normalisation'] == 'z-score':
        mean = statistics['mean'][0:data.shape[2]].to(data.device)
        std = statistics['std'][0:data.shape[2]].to(data.device)
        return data * std + mean
    elif statistics['normalisation'] == 'min-max':
        min = statistics['min'][0:data.shape[2]].to(data.device)
        max = statistics['max'][0:data.shape[2]].to(data.device)
        return data * (max - min) + min
    else:
        raise KeyError('unknwon normalisation ' + statistics['normalisation'])

def build_train_dataloader(config):
    train_data = pd.read_csv('./data/train_processed.csv').sort_values('5minute_intervals_timestamp')

    n_train = int(config['horizons']['train'] / DATA_TIME_INTERVAL)
    n_pred = int(config['horizons']['pred'] / DATA_TIME_INTERVAL)

    features = config['features']

    samples = []
    for subject in train_data['subject'].unique():
        subject_data = train_data[train_data['subject'] == subject]
        for start_idx in range(len(subject_data) - n_train - n_pred + 1):
            x_features = [subject_data[feature].iloc[start_idx:start_idx + n_train + n_pred].values for feature in features]

            # Stack features to get (sequence_length, n_features) array
            x_features = np.stack(x_features, axis=1)
            samples.append(x_features)

    print('   training samples:', len(samples))

    samples = torch.Tensor(np.array(samples))
    norm_statistics = get_normalisation_statistics(config, samples)
    samples = normalise(samples, norm_statistics)

    return DataLoader(
        TensorDataset(samples),
        batch_size=int(config['batch_size']),
        shuffle=True
    ), norm_statistics

def build_val_dataloader(config, norm_statistics):
    val_data = pd.read_csv('./data/val_processed.csv').sort_values('5minute_intervals_timestamp')

    n_train = int(config['horizons']['train'] / DATA_TIME_INTERVAL)
    n_pred = int(config['horizons']['pred'] / DATA_TIME_INTERVAL)

    features = config['features']

    X = []
    Y = []
    for subject in val_data['subject'].unique():
        subject_data = val_data[val_data['subject'] == subject]
        for start_idx in range(len(subject_data) - n_train - n_pred + 1):
            x_features = [subject_data[feature].iloc[start_idx:start_idx + n_train].values for feature in features]
            y = subject_data['cbg'].iloc[start_idx + n_train:start_idx + n_train + n_pred].values[:, np.newaxis]

            # Stack features to get (sequence_length, n_features) array
            x_features = np.stack(x_features, axis=1)
            X.append(x_features)

            Y.append(y)

    print('   val samples:', len(X))

    X = torch.Tensor(np.array(X))
    X = normalise(X, norm_statistics)

    Y = torch.Tensor(np.array(Y))
    Y = normalise(Y, norm_statistics)

    return DataLoader(
        TensorDataset(X, Y),
        batch_size=int(config['batch_size']),
        shuffle=False,
    )

def validate(model, val_dataloader, norm_statistics):
    model.eval()

    with torch.no_grad():
        Y_trues = []
        Y_preds = []
        for X, Y_true in val_dataloader:
            X = X.to(device)  # Shape: (batch_size, sequence_length, n_features)
            Y_true = Y_true.to(device)

            Y_pred = torch.zeros(Y_true.shape, device=device)
            last_hidden_state = None
            for i in range(Y_true.shape[1]):
                pred, last_hidden_state = model(X, last_hidden_state)
                Y_pred[:, i] = pred[:, -1].unsqueeze(-1)

                if (X.shape[2] >= 2):
                    #####
                    X_zeros = torch.zeros(X.shape[0], 1, X.shape[2] - 2, device=pred.device) # zeros for unpredicted features
                    X_pred = pred[:, -1].unsqueeze(-1).unsqueeze(-1) # cbg prediction
                    X_mean = torch.mean(X[:, : ,1], dim = 1).unsqueeze(-1).unsqueeze(-1) # mean for basal over validaiton X
                    X = torch.cat((X_pred, X_mean, X_zeros), dim =-1)
                    #####
                else:
                    X = pred[:, -1].unsqueeze(-1).unsqueeze(-1)

            Y_preds.append(denormalise(Y_pred, norm_statistics))
            Y_trues.append(denormalise(Y_true, norm_statistics))

        Y_preds = torch.cat(Y_preds)
        Y_trues = torch.cat(Y_trues)

    return Y_preds, Y_trues

def calculate_metrics(Y_preds, Y_trues):
    with torch.no_grad():
        Y_preds = Y_preds.flatten()
        Y_trues = Y_trues.flatten()

        mse = nn.functional.mse_loss(Y_preds, Y_trues)
        mape = torch.mean(torch.abs(Y_trues - Y_preds) / Y_trues.abs())
        gmse = calculate_gmse(Y_preds, Y_trues)
        return {
            'mse': mse.detach().item(),
            'rmse': torch.sqrt(mse).detach().item(),
            'mape': mape.detach().item(),
            'mae': nn.functional.l1_loss(Y_preds, Y_trues).detach().item(),
            'gmse': gmse.detach().item(),
            'grmse': torch.sqrt(gmse).detach().item()
        }

def calculate_gmse(Y_preds, Y_trues, lower=70, upper=180):
    weights = torch.ones_like(Y_trues)
    weights[(Y_trues <= lower) & (Y_preds > Y_trues)] = 2.5
    weights[(Y_trues >= upper) & (Y_preds < Y_trues)] = 2.0

    squared_errors = (Y_trues - Y_preds) ** 2
    weighted_squared_errors = weights * squared_errors

    return torch.mean(weighted_squared_errors)

def set_seed(seed):
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
      torch.cuda.manual_seed(seed)
      torch.cuda.manual_seed_all(seed)

  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False

def train(config_name):
    set_seed(1)

    print('👉 loading config')

    config = load_config(config_name)

    print('👉 loading data')
    train_dataloader, norm_stats = build_train_dataloader(config)
    val_dataloader = build_val_dataloader(config, norm_stats)

    model = build_model(config)
    model = model.to(device)

    print('👉 training model')

    optimizer = optim.Adam(model.parameters(), config['lr'], weight_decay=0.)
    criterion = get_criterion(config)

    features = config['features']

    losses = []
    val_scores = []
    best_epoch = -1
    best_epoch_score = None
    best_epche_state = None

    for epoch in (pbar := tqdm(range(config['epochs']))):
        pbar.set_description('Training')
        for i, (X,) in enumerate(train_dataloader):
            model.train()
            X = X.to(device)  # Shape: (batch_size, sequence_length, n_features)
            Y, _ = model(X[:, :-1])

            # Loss computation
            loss = criterion(Y.flatten(), X[:, 1:, 0].flatten())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.detach().item())
            if i % 50 == 0:
                if len(losses) >= 100:
                    pbar.set_postfix({'loss': np.mean(losses[-100])})

        pbar.set_description('Validating')

        y_preds, y_trues = validate(model, val_dataloader, norm_stats)
        epoch_scores = calculate_metrics(y_preds, y_trues)

        if best_epoch_score is None or epoch_scores[config['loss']] < best_epoch_score:
            best_epoch = epoch
            best_epoch_score = epoch_scores[config['loss']]
            best_epche_state = model.state_dict()

        val_scores.append(epoch_scores)


    print('👉 saving model')
    torch.save(best_epche_state, f'./models/{config_name}.pt')

    with open(f'./models/{config_name}_meta.json', '+w') as f:
        scores = {}
        for epoch_data in val_scores:
            for key, value in epoch_data.items():
                scores[key] = scores.get(key, [])
                scores[key].append(value)

        json.dump({
            'norm_stats': {k: v.tolist() if torch.is_tensor(v) else v for k, v in norm_stats.items()},
            'best_epoch': best_epoch,
            'loss': losses,
            'val_scores': scores
        }, f)

    return model



def investigate_predictions(config_name):
    print('👉 loading model')

    config = load_config(config_name)
    model = build_model(config).to(device)

    model.load_state_dict(torch.load(f'./models/{config_name}.pt', weights_only=True))

    print('👉 loading data')
    _, norm_stats = build_train_dataloader(config)
    val_dataloader = build_val_dataloader(config, norm_stats)

    print('👉 predicting values')
    Y_preds, Y_trues = validate(model, val_dataloader, norm_stats)

    X = denormalise(torch.vstack([X for X, _ in val_dataloader]), norm_stats)
    return X, Y_preds, Y_trues



#%%
