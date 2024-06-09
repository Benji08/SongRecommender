from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import json
import pickle
import re
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


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

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

print("Rozmiar zbioru treningowego:", X_train_scaled.shape[0])
print("Rozmiar zbioru walidacyjnego:", X_val_scaled.shape[0])

X_train = X_train.tolist()
y_train = y_train.tolist()
X_val = X_val.tolist()
y_val = y_val.tolist()

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

train_data = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
val_data = TensorDataset(X_val_tensor, y_val_tensor)
val_loader = DataLoader(val_data, batch_size=64)


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size_1, hidden_size_2):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size_1)
        self.fc2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.fc3 = nn.Linear(hidden_size_2, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


model = MLP(input_size=X.shape[1], hidden_size_1=256, hidden_size_2=128)


criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0002)

num_epochs = 40
for epoch in range(num_epochs):
    train_loss = 0.0
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets.view(-1, 1))
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * inputs.size(0)
    avg_train_loss = train_loss / len(train_loader.dataset)

    with torch.no_grad():
        val_loss = 0.0
        for inputs, targets in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets.view(-1, 1))
            val_loss += loss.item() * inputs.size(0)
        avg_val_loss = val_loss / len(val_loader.dataset)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

with torch.no_grad():
    y_val_pred = model(X_val_tensor).numpy()
    y_val_pred_non_negative = np.maximum(y_val_pred, 0)

    mse = mean_squared_error(y_val, y_val_pred_non_negative)
    r2 = r2_score(y_val, y_val_pred_non_negative)
    mae = np.mean(np.abs(y_val - y_val_pred_non_negative))

    print("Mean Absolute Error:", mae)
    print("Mean Squared Error:", mse)
    print("R^2 Score:", r2)


all_data_scaled = scaler.transform(X)

all_data_scaled = all_data_scaled.tolist()
all_data_tensor = torch.tensor(all_data_scaled, dtype=torch.float32)
all_data = TensorDataset(all_data_tensor)
all_data_loader = DataLoader(all_data, batch_size=64)

with torch.no_grad():
    y_all_pred = model(all_data_tensor)
y_all_pred_non_negative = np.maximum(y_all_pred, 0)

y_all_pred_non_negative = y_all_pred_non_negative.numpy().flatten()
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
    file_name = f'../../microservice/app/data/nn/top_10_{genre}_nn.pkl'
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

save_data(top_data, '../../microservice/app/data/nn/top_20_nn.pkl')
