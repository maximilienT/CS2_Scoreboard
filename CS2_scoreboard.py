import pandas as pd

# Load in CSV file as a dataframe
df = pd.read_csv("example_player_death.csv")

# Kills are only credited for killing an enemy. notna() drops the planted_c4 deaths outright rather than relying on NaN comparing unequal to everything.
enemy_kills = df[(df['attacker_team_code'] != df['player_team_code'])
                & df['attacker_id_fixed'].notna()]

# If player_id and attacker_id match this is a suicide
suicides = df[df['player_id_fixed'] == df['attacker_id_fixed']]

# If player team and attacker team match + player id and attacker don't match (without counts suicides) count towards team kills
team_kills = df[(df['player_team_code'] == df['attacker_team_code'])
                & (df['player_id_fixed'] != df['attacker_id_fixed'])]

# Uses enemy kills to calculate each players total kills
kills = enemy_kills.groupby('attacker_id_fixed').size().rename("kills")

# Combine suicide and team kill dfs into one df
penalty_kills = pd.concat([suicides, team_kills], ignore_index=True)

# Calculates each players number of team kills and suicides
penalty_kills = penalty_kills.groupby('attacker_id_fixed').size()

# Subtracts team kills and suicides from players total kill counts. Use .sub since original dfs didnt have matching index (attacker_ids). Done because in game suicides and team kills should take away from your total kill counter.
final_kills = kills.sub(penalty_kills,fill_value=0).rename('kills')

# Group by player_id and gets number of rows as deaths as each player_id event in this context is a death
deaths = df.groupby('player_id_fixed').size().rename("deaths")

# Concat kills and deaths together to get scoreboard. fillna(0) handles players with either no kill or deaths
scoreboard = pd.concat([final_kills, deaths], axis=1).fillna(0).astype(int)

# Gets index as its own column, name it player_id_fixed, and cast to int (source column is float due to NaN rows)
scoreboard = scoreboard.reset_index().rename(columns={'index':'player_id_fixed'})
scoreboard['player_id_fixed'] = scoreboard['player_id_fixed'].astype(int)

# Sort by kills, deaths, player_id desc
scoreboard = scoreboard.sort_values(by=['kills', 'deaths', 'player_id_fixed'], ascending=[False, False, False])

# Export the scoreboard to csv
scoreboard.to_csv("scoreboard.csv", index=False)