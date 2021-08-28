import pandas as pd
from data_helper import one_hot_encode, normalize_with_scaler
from neural_net import get_model

df = pd.read_csv('data/csgo_round_snapshots.csv')
one_hot_encode(df,"round_winner",['CT','T'])
one_hot_encode(df,"map",['de_dust2', 'de_mirage', 'de_nuke', 'de_inferno', 'de_overpass', 'de_vertigo', 'de_train', 'de_cache'])
vals = normalize_with_scaler(df)

model = get_model(vals.shape[-1])
model.summary()