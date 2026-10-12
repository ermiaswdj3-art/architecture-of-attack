"""Fetch the open data used in the book from its owners (not redistributed with the code).
StatsBomb Open Data (credit required; read their user agreement): events + matches for competition 43 / season 3.
Metrica Sports sample data, Games 1-2 (credit required)."""
import json, os, urllib.request
SB='https://raw.githubusercontent.com/statsbomb/open-data/master/data'
MT='https://raw.githubusercontent.com/metrica-sports/sample-data/master/data'
def get(url,path):
    if os.path.exists(path): return
    os.makedirs(os.path.dirname(path) or '.',exist_ok=True); print('fetching',url); urllib.request.urlretrieve(url,path)
get(f'{SB}/matches/43/3.json','data/matches_43_3.json')
for m in json.load(open('data/matches_43_3.json')): get(f"{SB}/events/{m['match_id']}.json",f"data/{m['match_id']}.json")
for g in (1,2):
    for f in (f'Sample_Game_{g}_RawTrackingData_Home_Team.csv',f'Sample_Game_{g}_RawTrackingData_Away_Team.csv',f'Sample_Game_{g}_RawEventsData.csv'):
        get(f'{MT}/Sample_Game_{g}/{f}',f'tracking/{f}')
print('done')
