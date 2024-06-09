import json
from sklearn.metrics import mutual_info_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


def discretize_columns(data, columns_to_discretize, num_bins=20):
    df = pd.DataFrame(data)

    for idx in columns_to_discretize:
        column_name = df.columns[idx]
        df[column_name] = pd.cut(df[column_name], bins=num_bins, labels=False)

    discretized_data = df.values
    return discretized_data


def prepare_data_to_calculate_mi(input_file):
    with open(input_file, 'r') as file:
        X = []
        Y = []
        param_names = None
        for line in file:
            data = json.loads(line)
            if param_names is None:
                param_names = list(data.keys())
                param_names.remove('month_0_skip_count')
                param_names.remove('month_0_play_count')
                param_names.remove('month_0_like_count')
                param_names.remove('genres')
            value = list(data.values())
            Y.append(value[18]*3 + value[17])

            value.pop(18)
            value.pop(17)
            value.pop(16)
            value.pop(15)
            X.append(value)
    X = discretize_columns(X, [0, 1, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17])
    Y = discretize_columns(Y, [0])
    return X, Y, param_names


def shuffle_array(array):
    np.random.shuffle(array)
    return array


def calculate_mi(X, Y):
    X_shuffled = shuffle_array(X.copy())
    X_shuffled = X_shuffled.T
    X = X.T
    Y = Y.T
    MI_values = []
    MI_values_shuffled = []

    for i in range(len(X)):
        MI_value = mutual_info_score(X[i], Y[0])
        MI_values.append(round(MI_value, 4))

        MI_value_shuffled = mutual_info_score(X_shuffled[i], Y[0])
        MI_values_shuffled.append(round(MI_value_shuffled, 4))

    return (MI_values, MI_values_shuffled)


def plot_mi_values(mi_values, mi_values_shuffled, param_names):
    num_features = len(mi_values)
    bar_width = 0.35
    index = np.arange(num_features)

    plt.figure(figsize=(10, 6))
    plt.bar(index, mi_values, bar_width, color='blue', label='MI Values')
    plt.bar(index + bar_width, mi_values_shuffled, bar_width, color='orange', label='Shuffled MI Values')

    plt.xlabel('Parameter')
    plt.ylabel('Mutual Information (MI)')
    plt.title('Mutual Information (MI) for each parameter')
    plt.xticks(index + bar_width / 2, param_names, rotation=90, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_correlation_matrix(data, param_names):
    columns_to_keep = [col_idx for col_idx in range(data.shape[1]) if col_idx != 3]
    data_without_categorical = data[:, columns_to_keep]
    updated_param_names = [param_names[idx] for idx in range(len(param_names)) if idx != 3]

    correlation_matrix = np.zeros((len(updated_param_names), len(updated_param_names)))
    for i, param1 in enumerate(data_without_categorical.T):
        for j, param2 in enumerate(data_without_categorical.T):
            corr, _ = pearsonr(param1, param2)
            correlation_matrix[i, j] = corr

    plt.figure(figsize=(20, 25))
    plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='nearest')
    plt.colorbar()
    plt.title('Correlation Matrix', fontsize=20)
    plt.xticks(np.arange(len(updated_param_names)), updated_param_names, rotation=90, fontsize=20)
    plt.yticks(np.arange(len(updated_param_names)), updated_param_names, fontsize=20)
    plt.tight_layout()
    plt.show()


def draw_histograms(data, param_names):
    num_plots = len(param_names)
    fig, axs = plt.subplots(num_plots, figsize=(10, 6*num_plots))

    for i, subset in enumerate(param_names):
        axs[i].hist(data[:, i], bins=20, color='skyblue', edgecolor='black')
        axs[i].set_title(f'Histogram {param_names[i]}')
        axs[i].set_xlabel('Wartość')
        axs[i].set_ylabel('Liczebność')
        axs[i].grid(True)

    plt.tight_layout()
    plt.show()


X, Y, param_names = prepare_data_to_calculate_mi('data/tracks_filtered_with_genres_with_event_counts.jsonl')
MI_values, MI_values_shuffled = calculate_mi(X, Y)

print("MI values:                ", MI_values)
print("MI values after shuffling:", MI_values_shuffled)
plot_mi_values(MI_values, MI_values_shuffled, param_names)

draw_histograms(X, param_names)

plot_correlation_matrix(X, param_names)
