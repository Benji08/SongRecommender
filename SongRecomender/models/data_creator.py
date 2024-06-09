import json
from datetime import datetime, timedelta


def filter_sessions_and_save(input_file, output_file):
    with open(input_file, 'r') as input_f, open(output_file, 'w') as output_f:
        for line in input_f:
            data = json.loads(line)
            if "session_id" in data:
                del data["session_id"]
            if "timestamp" in data:
                data["timestamp"] = data["timestamp"][:10]
            event_type = data.get("event_type", "advertisment")
            if event_type != "advertisment":
                output_f.write(json.dumps(data) + '\n')


def filter_tracks_and_save(input_file, output_file):
    with open(input_file, 'r') as f:
        with open(output_file, 'w') as fout:
            for line in f:
                track = json.loads(line)
                loudness = track.get('loudness', 0)
                tempo = track.get('tempo', 0)

                release_date = track.get('release_date')
                if release_date:
                    track['release_date'] = int(release_date.split('-')[0])

                if loudness <= 0.0 and tempo != 0.0:
                    fout.write(json.dumps(track) + '\n')


def add_genres_to_tracks(tracks_file, artists_file, output_file):
    artist_genres = {}
    with open(artists_file, 'r') as artists_f:
        for line in artists_f:
            artist_data = json.loads(line)
            artist_genres[artist_data['id']] = artist_data['genres']

    with open(tracks_file, 'r') as tracks_f, open(output_file, 'w') as output_f:
        for line in tracks_f:
            track_data = json.loads(line)
            if track_data['id_artist'] in artist_genres:
                track_data['genres'] = artist_genres[track_data['id_artist']]
            else:
                track_data['genres'] = []
            output_f.write(json.dumps(track_data) + '\n')


def count_events_per_song_in_months(file_path, target_date, num_months):
    target_date = datetime.strptime(target_date, '%Y-%m-%d')
    results = {}

    month_start_dates = {}
    current_year = target_date.year
    current_month = target_date.month
    for i in range(num_months):
        current_month = current_month - 1
        if current_month <= 0:
            current_year = current_year - 1
            current_month = 12 + current_month

        month_start_date = datetime(current_year, current_month, 1)
        month_end_date = month_start_date.replace(day=1, month=(month_start_date.month % 12) + 1, year=(
                month_start_date.year + month_start_date.month//12)) - timedelta(days=1)
        month_name = f'month_{i}'
        month_start_dates[month_name] = (month_start_date, month_end_date)

    with open(file_path, 'r') as file:
        for line in file:
            event = json.loads(line)
            event_date = datetime.strptime(event['timestamp'], '%Y-%m-%d')

            for month, (month_start, month_end) in month_start_dates.items():
                if month_start <= event_date <= month_end:
                    song_id = event['track_id']
                    if song_id not in results:
                        results[song_id] = {month: {'skip': 0, 'play': 0, 'like': 0} for month in
                                            month_start_dates.keys()}
                    results[song_id][month][event['event_type']] += 1
                    break

    return results


def count_events_per_song_in_months_and_append(file_path_read, file_path_write, target_date, num_months):
    event_counts_per_song = count_events_per_song_in_months(file_path_read, target_date, num_months)

    with open(file_path_write, 'r') as file:
        tracks_data = [json.loads(line) for line in file]

    for track_data in tracks_data:
        track_id = track_data['id']
        if track_id in event_counts_per_song:
            for month, event_counts in event_counts_per_song[track_id].items():
                for event_type, count in event_counts.items():
                    month_column_name = f'{month}_{event_type}_count'
                    track_data[month_column_name] = count
        else:
            for month in range(num_months):
                for event_type in ['skip', 'play', 'like']:
                    month_column_name = f'month_{month}_{event_type}_count'
                    track_data[month_column_name] = 0

    output_file_path = file_path_write.replace('.jsonl', '_with_event_counts.jsonl')
    with open(output_file_path, 'w') as output_file:
        for track_data in tracks_data:
            output_file.write(json.dumps(track_data) + '\n')


session_input_file = 'IUM_Z01_T75_V3/sessions.jsonl'
session_output_file = 'data/sessions_filtered.jsonl'
filter_sessions_and_save(session_input_file, session_output_file)

tracks_input_file = 'IUM_Z01_T75_V3/tracks.jsonl'
tracks_output_file = 'data/tracks_filtered.jsonl'
filter_tracks_and_save(tracks_input_file, tracks_output_file)

artists_input_file = 'IUM_Z01_T75_V3/artists.jsonl'
tracks_filtered_with_genres_output_file = 'data/tracks_filtered_with_genres.jsonl'
add_genres_to_tracks(tracks_output_file, artists_input_file, tracks_filtered_with_genres_output_file)


count_events_per_song_in_months_and_append(session_output_file, tracks_filtered_with_genres_output_file,
                                           '2024-04-9', 2)
