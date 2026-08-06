import pandas as pd

# Load in CSV file as a dataframe
df = pd.read_csv("example_player_death.csv")

# Group by attacker_id and gets number of rows as kills as each attacker event is a kill
kills = df.groupby('attacker_id_fixed').size().rename("kills")

# Group by player_id and gets number of rows as deaths as each player_id event in this context is a death
deaths = df.groupby('player_id_fixed').size().rename("deaths")

# Concat kills and deaths together to get scoreboard. fillna(0) handles players with either no kill or deaths
scoreboard = pd.concat([kills, deaths], axis=1).fillna(0).astype(int)

# Gets index as its own column, name it player_id_fixed, and cast to int (source column is float due to NaN rows)
scoreboard = scoreboard.reset_index().rename(columns={'index':'player_id_fixed'})
scoreboard['player_id_fixed'] = scoreboard['player_id_fixed'].astype(int)

# Sort by kills, deaths, player_id desc
scoreboard = scoreboard.sort_values(by=['kills', 'deaths', 'player_id_fixed'], ascending=[False, False, False])

# Export the scoreboard to csv
scoreboard.to_csv("scoreboard.csv", index=False)