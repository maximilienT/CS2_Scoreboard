"""Build a CS2 scoreboard from the player_death channel of a single match.

Reads example_player_death.csv (one row per death) and writes scoreboard.csv
with player_id_fixed, kills and deaths, sorted by kills, deaths, then
player_id_fixed, all descending and compared numerically.

Scoring rules applied, matching the in-game CS2 scoreboard:
  - A kill is credited only for killing a player on the opposing team.
  - Bomb deaths (planted_c4, no attacker) credit nobody.
  - Suicides and team kills credit nobody and deduct 1 from the killer.
  - Every row is a death for the victim, including bomb deaths and suicides.

The deduction is an interpretation; the task states only that team codes
matter. In this match it affects players 7, 8 and 10. Dropping the .sub()
call below reverts to plain enemy-kill counts.
"""

import pandas as pd

df = pd.read_csv("example_player_death.csv")

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

# Every row is one death, so a player's row count is their death count.
deaths = df.groupby('player_id_fixed').size().rename("deaths")

# fillna(0) covers anyone missing from either series (no kills, or no deaths).
scoreboard = pd.concat([final_kills, deaths], axis=1).fillna(0).astype(int)

# Move the id out of the index. concat drops the clashing index names, so the
# column arrives as 'index'. Cast to int as the source column is float via NaN.
scoreboard = scoreboard.reset_index().rename(columns={'index': 'player_id_fixed'})
scoreboard['player_id_fixed'] = scoreboard['player_id_fixed'].astype(int)

# Sort by kills, deaths, player_id desc. player_id_fixed is int by now, so this
# sorts numerically rather than as a string.
scoreboard = scoreboard.sort_values(by=['kills', 'deaths', 'player_id_fixed'],
                                    ascending=[False, False, False])

# Export the scoreboard to csv
scoreboard.to_csv("scoreboard.csv", index=False)