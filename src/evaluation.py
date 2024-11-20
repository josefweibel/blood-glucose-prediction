import os
if os.getcwd().endswith('notebooks'):
    os.chdir('..')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json


def model_evaluation_plots(model_name):
    with open(f'./models/{model_name}_meta.json') as f:
        model_data = json.load(f)

    # MSE Training Loss plot
    plt.figure(figsize=(10, 5))
    sns.lineplot(model_data['loss'])
    plt.title('Loss on Training Set')
    plt.xlabel('iteration')
    plt.ylabel('MSE')
    plt.show()

    # Log MSE Training Loss plot
    plt.figure(figsize=(10, 5))
    sns.lineplot(model_data['loss'])
    plt.title('Loss on Training Set')
    plt.xlabel('iteration')
    plt.ylabel('log MSE')
    plt.yscale('log')
    plt.show()

    # MAPE on Validation Set plot
    metric = 'mape'

    plt.figure(figsize=(10, 5))
    sns.lineplot(model_data['val_scores'][metric])
    plt.title(metric.upper() + ' on Validation Set')
    plt.xlabel('epoch')
    plt.ylabel(metric.upper())
    plt.show()


def model_evaluation_numeric(model_name):
    with open(f'./models/{model_name}_meta.json') as f:
        model_data = json.load(f)

    # Extract relevant metrics
    best_epoch = model_data['best_epoch']
    training_losses = model_data['loss']
    val_scores = model_data['val_scores']

    # Get MAPE and loss at the best epoch
    best_mape = val_scores['mape'][best_epoch]
    best_loss = training_losses[best_epoch]

    # Print metrics for comparison
    print(f"Model: {model_name}")
    print(f"Best Epoch: {best_epoch}")
    print(f"Training Loss at Best Epoch: {best_loss:.4f}")
    print(f"Validation MAPE at Best Epoch: {best_mape:.4f}")

    # Aggregate Metrics for Comparison
    mean_mape = np.mean(val_scores['mape'])
    std_mape = np.std(val_scores['mape'])

    print(f"Mean Validation MAPE: {mean_mape:.4f}")
    print(f"Standard Deviation of Validation MAPE: {std_mape:.4f}")

    # Worst-case Performance
    max_mape = np.max(val_scores['mape'])
    print(f"Worst-case Validation MAPE: {max_mape:.4f}")

    return best_epoch, best_loss, best_mape


def model_comparison_table(model_names):
    models_metrics = []
    for model_name in model_names:
        best_epoch, best_loss, best_mape = model_evaluation_numeric(model_name)
        models_metrics.append({"model": model_name, "best_epoch": best_epoch, "best_loss": best_loss, "best_mape": best_mape})

    # Display as a pandas DataFrame
    df_metrics = pd.DataFrame(models_metrics)
    print(df_metrics)

    return df_metrics



