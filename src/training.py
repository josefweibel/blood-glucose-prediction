import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm.notebook import tqdm
import os
import yaml
import json


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
    params = {**config['architecture']}
    del params['type']

    if config['architecture']['type'] == 'rnn':
        return RNNModel(**params, input_size=len(config['features']))
    else:
       raise KeyError('unknwon architecture ' + config['architecture']['type'])

def get_criterion(config):
    if config['loss'] == 'mse':
        return nn.MSELoss()
    else:
       raise KeyError('unknwon loss function ' + config['loss'])


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

    return DataLoader(
        TensorDataset(torch.Tensor(np.array(samples))),
        batch_size=int(config['batch_size']),
        shuffle=True
    )

def build_val_dataloader(config):
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
            y = subject_data['cbg'].iloc[start_idx + n_train:start_idx + n_train + n_pred].values

            # Stack features to get (sequence_length, n_features) array
            x_features = np.stack(x_features, axis=1)
            X.append(x_features)

            Y.append(y)

    print('   val samples:', len(X))

    return DataLoader(
        TensorDataset(torch.Tensor(np.array(X)), torch.Tensor(np.array(Y))),
        batch_size=int(config['batch_size']),
        shuffle=False,
    )

def validate(model, val_dataloader):
    model.eval()

    with torch.no_grad():
        Y_trues = []
        Y_preds = []
        for X, Y_true in val_dataloader:
            X = X.to(device)  # Shape: (batch_size, sequence_length, n_features)
            Y_true = Y_true.to(device)

            Y_pred = torch.zeros(Y_true.shape)
            last_hidden_state = None
            for i in range(Y_true.shape[1]):
                pred, last_hidden_state = model(X, last_hidden_state)
                Y_pred[:, i] = pred[:, -1]

                if (X.shape[2] >= 2):
                    #####
                    X_zeros = torch.zeros(X.shape[0], 1, X.shape[2] - 2, device=pred.device) # zeros for unpredicted features
                    X_pred = pred[:, -1].unsqueeze(-1).unsqueeze(-1) # cbg prediction
                    X_mean = torch.mean(X[:, : ,1], dim = 1).unsqueeze(-1).unsqueeze(-1) # mean for basal over validaiton X
                    X = torch.cat((X_pred, X_mean, X_zeros), dim =-1)
                    #####
                else:
                    X = torch.Tensor(pred[:, -1].unsqueeze(-1).unsqueeze(-1))

            Y_preds.extend(Y_pred)
            Y_trues.extend(Y_true)

        Y_preds = torch.vstack(Y_preds)
        Y_trues = torch.vstack(Y_trues)

    return Y_preds, Y_trues

def calculate_metrics(Y_preds, Y_trues):
    Y_preds = Y_trues.flatten()
    Y_trues = Y_trues.flatten()

    mse = nn.functional.mse_loss(Y_preds, Y_trues)
    mape = torch.mean(torch.abs(Y_trues - Y_preds) / Y_trues.abs())
    return {
        'mse': mse.detach().item(),
        'rmse': torch.sqrt(mse).detach().item(),
        'mape': mape.detach().item(),
        'mae': nn.functional.l1_loss(Y_preds, Y_trues).detach().item()
    }


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
    train_dataloader = build_train_dataloader(config)
    val_dataloader = build_val_dataloader(config)

    model = build_model(config)
    model = model.to(device)

    print('👉 training model')

    optimizer = optim.Adam(model.parameters(), config['lr'], weight_decay=0.)
    criterion = get_criterion(config)

    features = config['features']
    nb_features = len(features)

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
            Y, _ = model(X)

            # Loss computation
            if nb_features == 1:
                loss = criterion(Y.flatten(), X.flatten())  # Compare with the same flattened tensor
            else:
                loss = criterion(Y.flatten(), X[:, :, 0].flatten())  # Adjust target as needed for the first feature

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.detach().item())
            if i % 50 == 0:
                if len(losses) >= 100:
                    pbar.set_postfix({'loss': round(np.mean(losses[-100]), 3)})

        pbar.set_description('Validating')

        y_trues, y_preds = validate(model, val_dataloader)
        epoch_scores = calculate_metrics(y_trues, y_preds)

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
    val_dataloader = build_val_dataloader(config)

    print('👉 predicting values')
    y_trues, y_preds = validate(model, val_dataloader)

    X = torch.vstack([X for X, _ in val_dataloader])
    return X, y_trues, y_preds



#%%
