from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import  Ridge
from sklearn.metrics import mean_squared_error, r2_score
import json
import numpy as np
import pickle
import re


def save_data(data, file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(data, file)


def load_jsonl_to_numpy(filename):
    dataX = []
    dataY = []
    dataX_all_values = []
    with open(filename, 'r') as file:
        for line in file:
            entry = json.loads(line)
            dataX_all_values.append(list(entry.values()))
            dataY.append(entry['month_0_play_count']+3*entry['month_0_like_count'])
            entry.pop('month_0_play_count')
            entry.pop('month_0_skip_count')
            entry.pop('month_0_like_count')
            entry.pop('tempo')
            entry.pop('valence')
            entry.pop('liveness')
            entry.pop('instrumentalness')
            entry.pop('speechiness')
            entry.pop('key')
            entry.pop('energy')
            entry.pop('id')
            entry.pop('name')
            dataX.append(list(entry.values()))

    return np.array(dataX, dtype=object), np.array(dataY), np.array(dataX_all_values)


def one_hot_encode_genres(genres_list, unique_genres):
    encoding = np.zeros(len(unique_genres))
    for genre in genres_list:
        index = unique_genres.index(genre)
        encoding[index] = 1

    return encoding


def encode_genres_in_X(X, idx):
    all_genres = set()
    for row in X:
        all_genres.update(row[idx])

    unique_genres = sorted(all_genres)
    encoded_genres = []

    for row in X:
        genres_list = row[idx]
        encoding = one_hot_encode_genres(genres_list, unique_genres)
        encoded_genres.append(encoding)

    X = np.delete(X, np.s_[idx], axis=1)
    X = np.concatenate((X, np.array(encoded_genres)), axis=1)

    return X


X, y, X_all_values = load_jsonl_to_numpy('data/tracks_filtered_with_genres_with_event_counts.jsonl')
X = encode_genres_in_X(X, 8)

artist_ids = [record[3] for record in X]

ohe = OneHotEncoder(sparse=False)
onehot_artist_ids = ohe.fit_transform(np.array(artist_ids).reshape(-1, 1))

X = np.delete(X, np.s_[3], axis=1)
X = np.concatenate((X, onehot_artist_ids), axis=1)

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_val_scaled = scaler.transform(X_val)

print("Rozmiar zbioru treningowego:", X_train_scaled.shape[0])
print("Rozmiar zbioru testowego:", X_test_scaled.shape[0])
print("Rozmiar zbioru walidacyjnego:", X_val_scaled.shape[0])

ridge_model = Ridge()

param_grid = {
    'alpha': [0.57, 0.58, 0.59]
}

grid_search = GridSearchCV(estimator=ridge_model, param_grid=param_grid, cv=5)

grid_search.fit(X_train_scaled, y_train)

print("Najlepsze parametry:", grid_search.best_params_)
print("Najlepszy wynik:", grid_search.best_score_)

y_pred = grid_search.predict(X_test_scaled)
y_pred_non_negative = np.maximum(y_pred, 0)


mse = mean_squared_error(y_test, y_pred_non_negative)
r2 = r2_score(y_test, y_pred_non_negative)
mae = np.mean(np.abs(y_test - y_pred_non_negative))
print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("R^2 Score:", r2)

print(np.mean(y_test))
print(np.median(y_test))

X_scaled = scaler.transform(X)
y_all_pred = grid_search.predict(X_scaled)
y_all_pred_non_negative = np.maximum(y_all_pred, 0)

top_indices = np.argsort(y_all_pred_non_negative)[-300:][::-1]

top_predictions_by_genre = {
    'pop': [],
    'rock': [],
    'country': [],
    'rap': [],
    'hip hop': []
}

for idx in top_indices:
    genres = X_all_values[idx][17]

    if any(re.search(r'\bpop\b', subgenre.lower()) for subgenre in genres):
        top_predictions_by_genre['pop'].append({
            'song_id': X_all_values[idx][0],
            'song_name': X_all_values[idx][1]
        })
    if any(re.search(r'\brock\b', subgenre.lower()) for subgenre in genres):
        top_predictions_by_genre['rock'].append({
            'song_id': X_all_values[idx][0],
            'song_name': X_all_values[idx][1]
        })
    if any(re.search(r'\bcountry\b', subgenre.lower()) for subgenre in genres):
        top_predictions_by_genre['country'].append({
            'song_id': X_all_values[idx][0],
            'song_name': X_all_values[idx][1]
        })
    if any(re.search(r'\brap\b', subgenre.lower()) for subgenre in genres):
        top_predictions_by_genre['rap'].append({
            'song_id': X_all_values[idx][0],
            'song_name': X_all_values[idx][1]
        })
    if any(re.search(r'\bhip hop\b', subgenre.lower()) for subgenre in genres):
        top_predictions_by_genre['hip hop'].append({
            'song_id': X_all_values[idx][0],
            'song_name': X_all_values[idx][1]
        })

for genre, predictions in top_predictions_by_genre.items():
    top_data = []
    for pred in predictions[:10]:
        top_data.append({
            'song_id': pred['song_id'],
            'song_name': pred['song_name']
        })
    file_name = f'../../microservice/app/data/regression/top_10_{genre}_regression.pkl'
    save_data(top_data, file_name)
    print(f"Zapisano top 10 predykcji dla gatunku {genre} do pliku {file_name}")

top_indices = top_indices[:20]
top_X_values = X_all_values[top_indices]
top_data = []
for i, x_values in enumerate(top_X_values):
    top_data.append({
        'song_id': x_values[0],
        'song_name': x_values[1]
    })

save_data(top_data, '../../microservice/app/data/regression/top_20_regression.pkl')
