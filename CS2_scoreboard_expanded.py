import pandas as pd
import math

# Load in CSV file as a dataframe
df = pd.read_csv("example_player_death.csv")



# Create X,Y,Z coords for player and attackers
df['player_xyz'] = df[['player_x_pos', 'player_y_pos', 'player_z_pos']].apply(tuple, axis=1)
df['attacker_xyz'] = df[['attacker_x_pos', 'attacker_y_pos', 'attacker_z_pos']].apply(tuple, axis=1)

# Calculate distance of each instance of a kill, handle any rows without a X,Y,Z due to environmental death
df['kill_dist'] = df.apply(
      lambda row: round(math.dist(row['player_xyz'], row['attacker_xyz']),1)
      if pd.notna(row['attacker_x_pos']) else None,
      axis=1
  )



# Group by attacker_id and gets number of rows as kills as each attacker event is a kill
kills = df.groupby('attacker_id_fixed').size().rename("kills")

# Group by player_id and gets number of rows as deaths as each player_id event in this context is a death
deaths = df.groupby('player_id_fixed').size().rename("deaths")

# Group by assister_id_fixed and gets number of rows as assists as each assister_id even is an assist.
assists = df.groupby('assister_id_fixed').size().rename("assists")


# Group by attacker_id and gets total number of headshot kills, Then used to calculate percentage of kills as headshots
headshots = df.groupby('attacker_id_fixed')['is_headshot'].sum().rename("headshots")
hs_kill_percent = round(
    100*(headshots/kills).rename("Headshot kill %")
    ,1)

# Group by round number to find if one player got an ace that round
kills_per_round = df.groupby(['round','attacker_id_fixed']).size()
aces = (kills_per_round == 5).groupby('attacker_id_fixed').sum().rename('aces')

# Group by attacker_id and find the most common weapon used for kills
best_weapon = df.groupby('attacker_id_fixed')['weapon_name'].agg(lambda x: x.mode().iloc[0]).rename('best weapon')

# Group by attacker_id and get their largest kill distance
longest_kill_dist = df.groupby('attacker_id_fixed')['kill_dist'].max().rename("longest kill distance")


# Locate in each round which player has opening frag then tally them all up
open_frag = (df.loc[
    df.groupby('round')['tick'].idxmin()
    , ['round', 'attacker_id_fixed']]
             .groupby('attacker_id_fixed').size().rename("opening frags"))


# Trade kills


# Concat kills and deaths together to get scoreboard. fillna(0) handles players with either no kill or deaths
scoreboard = pd.concat([kills, deaths, assists, longest_kill_dist, aces, open_frag], axis=1).fillna(0).astype(int)
scoreboard['best weapon'] = best_weapon
scoreboard['Headshot kill %'] = hs_kill_percent

# Gets index as its own column, name it player_id_fixed, and cast to int (source column is float due to NaN rows)
scoreboard = scoreboard.reset_index().rename(columns={'index':'player_id_fixed'})
scoreboard['player_id_fixed'] = scoreboard['player_id_fixed'].astype(int)

# Calculate the K/D ratio for each player.
scoreboard['K/D Ratio'] = round(scoreboard['kills'] / scoreboard['deaths'],2)

# Sort by kills, deaths, player_id desc
scoreboard = scoreboard.sort_values(by=['kills', 'deaths','player_id_fixed'], ascending=[False, False, False])

# Final organization of scoreboard
scoreboard = scoreboard[['player_id_fixed','kills','deaths','assists','K/D Ratio','Headshot kill %','aces','opening frags','best weapon', 'longest kill distance']]

with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(scoreboard)
# Export the scoreboard to csv
# scoreboard.to_csv("scoreboard.csv", index=False)