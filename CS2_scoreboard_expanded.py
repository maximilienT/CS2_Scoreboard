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



# Kills are only credited for killing an enemy. notna() drops the planted_c4
# deaths outright rather than relying on NaN comparing unequal to everything.
enemy_kills = df[(df['attacker_team_code'] != df['player_team_code'])
                & df['attacker_id_fixed'].notna()]

# If player_id and attacker_id match this is a suicide
suicides = df[df['player_id_fixed'] == df['attacker_id_fixed']]

# Matching team codes with differing ids is a team kill. The id check is what
# keeps suicides, which also share a team code, out of this group.
team_kills = df[(df['player_team_code'] == df['attacker_team_code'])
                & (df['player_id_fixed'] != df['attacker_id_fixed'])]

kills = enemy_kills.groupby('attacker_id_fixed').size().rename("kills")

penalty_rows = pd.concat([suicides, team_kills], ignore_index=True)

penalty_kills = penalty_rows.groupby('attacker_id_fixed').size()

# Deduct suicides and team kills, since in game both take away from your kill counter.
# Use .sub since the two series don't share an index (not every killer has a penalty).
final_kills = kills.sub(penalty_kills, fill_value=0).rename('kills')

# Group by player_id and gets number of rows as deaths as each player_id event in this context is a death
deaths = df.groupby('player_id_fixed').size().rename("deaths")

# Group by assister_id_fixed and gets number of rows as assists as each assister_id even is an assist.
assists = df.groupby('assister_id_fixed').size().rename("assists")


# Group by attacker_id and gets total number of headshot kills, Then used to calculate percentage of kills as headshots
headshots = df.groupby('attacker_id_fixed')['is_headshot'].sum().rename("headshots")
hs_kill_percent = round(
    100*(headshots/final_kills).rename("Headshot kill %")
    ,1)

# Group by round number to find if one player got an ace that round
#FIX THIS, KILLS NOT RIGHT
kills_per_round = df.groupby(['round','attacker_id_fixed']).size()
assists_per_round = df.groupby(['round','assister_id_fixed']).size()
death_round = df.groupby(['round','player_id_fixed']).size()


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



# Trade window of ~5 seconds is 320 ticks
trade_window = 320
# Full player and round lists, used to reindex so zero-trade players/rounds are not dropped
player_ids = sorted(df['player_id_fixed'].unique())
rounds = sorted(df['round'].unique())

# Deaths that are still recent enough to be traded, oldest first
trade_events, recent_deaths = [], []

# Trade kills For loop
# Each row is a kill that might avenge one of the recent deaths
# Drops deaths that have aged out of the window or belong to an earlier round
# Confirms the current victim is the killer from that earlier death
# Confirms the avenger and the earlier victim are on the same team to not count tk
for _, row in df.iterrows():
    recent_deaths = [d for d in recent_deaths
                     if d['round'] == row['round'] and row.tick - d.tick < trade_window]

    for i, dead in enumerate(recent_deaths):
        if row.player_id_fixed == dead.attacker_id_fixed:
            if row.attacker_team_code == dead.player_team_code:
                trade_events.append({
                    'round': row['round'],
                    'tick': row.tick,
                    'avenger': int(row.attacker_id_fixed),
                    'teammate_traded': int(dead.player_id_fixed),
                    'killer_traded': int(row.player_id_fixed),
                    'ticks_to_trade': int(row.tick - dead.tick),
                })
                del recent_deaths[i]
                break

    recent_deaths.append(row)

trades = pd.DataFrame(trade_events)

# Scoreboard column
trade_kills = trades.groupby('avenger').size().rename('trade kills')

# Per round
trades_per_round = trades.groupby('round').size().reindex(rounds, fill_value=0)

# Concat all tabulated fields together to get scoreboard. fillna(0) handles players with either no kill or deaths
scoreboard = pd.concat([final_kills, deaths, assists, aces, open_frag, trade_kills], axis=1).fillna(0).astype(int)
scoreboard['longest kill distance'] = longest_kill_dist
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
scoreboard = scoreboard[['player_id_fixed','kills','deaths', 'K/D Ratio','assists', 'trade kills','Headshot kill %','aces','opening frags','best weapon', 'longest kill distance']]

with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(scoreboard)
# Export the scoreboard to csv
# scoreboard.to_csv("scoreboard.csv", index=False)