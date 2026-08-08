"""Build an expanded CS2 scoreboard from the player_death channel of one match.

Reads example_player_death.csv (one row per death) and prints a per-player
scoreboard with the core columns (kills, deaths, K/D) plus the extra stats a
CS2 end-of-match screen shows: assists, KAST%, trade kills, headshot %, aces,
opening frags, best weapon and longest kill.

Scoring rules applied, matching the in-game CS2 scoreboard:
  - A kill is credited only for killing a player on the opposing team.
  - Bomb deaths (planted_c4, no attacker) credit nobody.
  - Suicides and team kills credit nobody and deduct 1 from the killer.
  - Every row is a death for the victim, including bomb deaths and suicides.

Note on teams: sides swap at halftime, so a player's team_code changes mid
match. Every check below compares the codes recorded *on that row*, so a
side swap can never turn an enemy kill into a team kill.
"""

import pandas as pd
import math

# Load in CSV file as a dataframe
df = pd.read_csv("example_player_death.csv")

# Full player and round lists. Every per-round stat is reindexed onto the
# product of these two so players/rounds with a zero stay in the table as 0
# instead of silently vanishing from the groupby result.
player_ids = sorted(df['player_id_fixed'].unique())
rounds = sorted(df['round'].unique())

# ---------------------------------------------------------------------------
# Row classification: split every death into enemy kill / suicide / team kill.
# Everything downstream is built from these three views, so the CS2 scoring
# rules only have to be expressed once.
# ---------------------------------------------------------------------------

# Kills are only credited for killing an enemy. notna() drops the planted_c4
# deaths outright rather than relying on NaN comparing unequal to everything.
enemy_kills = df[(df['attacker_team_code'] != df['player_team_code'])
                 & df['attacker_id_fixed'].notna()]

# If player_id and attacker_id match this is a suicide (fall damage, own
# molotov, own grenade). CS2 takes a point off the victim's own kill count.
suicides = df[df['player_id_fixed'] == df['attacker_id_fixed']]

# Matching team codes with differing ids is a team kill. The id check is what
# keeps suicides, which also share a team code, out of this group.
team_kills = df[(df['player_team_code'] == df['attacker_team_code'])
                & (df['player_id_fixed'] != df['attacker_id_fixed'])]

# ---------------------------------------------------------------------------
# Core columns: kills, deaths, assists
# ---------------------------------------------------------------------------

# Raw enemy frags, before the suicide/team-kill deduction is applied. This is
# also the denominator used for headshot % below.
kills = enemy_kills.groupby('attacker_id_fixed').size().rename("kills")

# One combined penalty count per player, so a player who both team-killed and
# suicided loses a point for each.
penalty_rows = pd.concat([suicides, team_kills], ignore_index=True)
penalty_kills = penalty_rows.groupby('attacker_id_fixed').size()

# Deduct suicides and team kills, since in game both take away from your kill
# counter. Use .sub since the two series don't share an index (not every
# killer has a penalty), and fill_value keeps the penalty-free players intact.
final_kills = kills.sub(penalty_kills, fill_value=0).rename('kills')

# Every row is one death, so a player's row count is their death count. This
# deliberately includes bomb deaths and suicides, exactly as CS2 does.
deaths = df.groupby('player_id_fixed').size().rename("deaths")

# One assist per row that names an assister. groupby skips the NaN rows, so
# unassisted deaths drop out on their own. Assists are only ever recorded
# against an enemy in this feed, so no team filter is needed.
assists = df.groupby('assister_id_fixed').size().rename("assists")

# ---------------------------------------------------------------------------
# Accuracy and highlight stats
# ---------------------------------------------------------------------------

# Headshot kills over enemy frags. The denominator is `kills`, not
# `final_kills`: a team kill should not inflate the percentage by shrinking
# the bottom of the fraction.
headshots = enemy_kills.groupby('attacker_id_fixed')['is_headshot'].sum().rename("headshots")
hs_kill_percent = round(100 * (headshots / kills).rename("Headshot kill %"), 1)

# Aces are about clearing the enemy team, so count enemy kills only: a team
# kill must not pad the total to 5, and must not cancel a genuine 5-kill round
# either. >= 5 rather than == 5 so the count never silently misses a round.
kill_index = pd.MultiIndex.from_product([rounds, player_ids],
                                        names=['round', 'attacker_id_fixed'])
kills_per_round = (enemy_kills.groupby(['round', 'attacker_id_fixed']).size()
                   .reindex(kill_index, fill_value=0))
aces = (kills_per_round >= 5).groupby('attacker_id_fixed').sum().rename('aces')

# Best weapon and longest kill are built from enemy_kills, not df, so a
# team kill or a self-inflicted molotov can never become a player's
# signature weapon or their longest "kill".
best_weapon = (enemy_kills.groupby('attacker_id_fixed')['weapon_name']
               .agg(lambda x: x.mode().iloc[0]).rename('best weapon'))

# Straight-line 3D distance between attacker and victim at the moment of the
# kill. Bomb deaths have no attacker position, hence the notna() guard.
df['player_xyz'] = df[['player_x_pos', 'player_y_pos', 'player_z_pos']].apply(tuple, axis=1)
df['attacker_xyz'] = df[['attacker_x_pos', 'attacker_y_pos', 'attacker_z_pos']].apply(tuple, axis=1)
df['kill_dist'] = df.apply(
    lambda row: round(math.dist(row['player_xyz'], row['attacker_xyz']), 1)
    if pd.notna(row['attacker_x_pos']) else None,
    axis=1
)
longest_kill_dist = (df.loc[enemy_kills.index]
                     .groupby('attacker_id_fixed')['kill_dist']
                     .max().rename("longest kill distance"))

# Opening frag = the first *kill* of the round, not simply the first death.
# Taking idxmin over df would hand the frag to nobody when a round opens on a
# bomb death, and to the wrong player when it opens on a team kill or suicide,
# so the search is restricted to enemy kills.
open_frag = (enemy_kills.loc[enemy_kills.groupby('round')['tick'].idxmin(),
                             ['round', 'attacker_id_fixed']]
             .groupby('attacker_id_fixed').size().rename("opening frags"))

# ---------------------------------------------------------------------------
# Trade kills
# ---------------------------------------------------------------------------

# A trade is: a teammate dies, and within ~5 seconds someone on their team
# kills the player who killed them. Demos are 64 tick, so 5s = 320 ticks.
TRADE_WINDOW = 320

# One record per (death that got traded, avenging kill) pair. The two are not
# one-to-one: if a player fragged three of our team and then went down, that
# single revenge kill trades all three deaths. So the pairs are counted two
# different ways below - by avenging kill for the kill stat, by victim for
# KAST's T.
trade_events = []
recent_deaths = []  # deaths still inside the window, oldest first

# Rows arrive tick-ordered, but sort explicitly so the window logic does not
# depend on the export's ordering.
for _, row in df.sort_values(['round', 'tick']).iterrows():
    # Drop deaths that have aged out of the window or belong to an earlier round.
    recent_deaths = [d for d in recent_deaths
                     if d['round'] == row['round']
                     and row.tick - d.tick < TRADE_WINDOW]

    # Only a genuine enemy kill can trade. Skipping bomb deaths, suicides and
    # team kills here stops a team kill from being read as an avenging frag.
    is_enemy_kill = (pd.notna(row.attacker_id_fixed)
                     and row.attacker_team_code != row.player_team_code)

    if is_enemy_kill:
        for dead in recent_deaths:
            # The player who just died must be the one who made the earlier kill...
            if row.player_id_fixed != dead.attacker_id_fixed:
                continue
            # ...that earlier kill must itself have been an enemy kill (you do
            # not get traded for being team-killed)...
            if dead.attacker_team_code == dead.player_team_code:
                continue
            # ...and the avenger must be on the dead teammate's side.
            if row.attacker_team_code == dead.player_team_code:
                trade_events.append({
                    'round': row['round'],
                    'tick': row.tick,           # identifies the avenging kill
                    'avenger': int(row.attacker_id_fixed),
                    'teammate_traded': int(dead.player_id_fixed),
                    'killer_traded': int(row.player_id_fixed),
                    'ticks_to_trade': int(row.tick - dead.tick),
                })
                # No break: keep going so every teammate this player killed
                # inside the window is marked as traded.

    recent_deaths.append(row)

trades = pd.DataFrame(trade_events)

# Scoreboard column, credited to the player who got the revenge frag. Counted
# as distinct avenging kills - (round, tick) identifies the kill - so a single
# frag that avenges three teammates is still one trade kill, not three.
trade_kills = (trades.groupby('avenger')[['round', 'tick']]
               .apply(lambda g: len(g.drop_duplicates())).rename('trade kills'))

# ---------------------------------------------------------------------------
# KAST%: share of rounds in which a player got a Kill, an Assist, Survived,
# or was Traded.
# ---------------------------------------------------------------------------

assist_index = pd.MultiIndex.from_product([rounds, player_ids],
                                          names=['round', 'assister_id_fixed'])
assists_per_round = (df.groupby(['round', 'assister_id_fixed']).size()
                     .reindex(assist_index, fill_value=0))

# A player dies at most once per round, so 1 - (deaths that round) is 1 when
# they survived and 0 when they did not.
survive_index = pd.MultiIndex.from_product([rounds, player_ids],
                                           names=['round', 'player_id_fixed'])
survive_round = 1 - (df.groupby(['round', 'player_id_fixed']).size()
                     .reindex(survive_index, fill_value=0))

# The T in KAST belongs to the player who *was* traded, not to the avenger.
# Grouping by 'avenger' here would be a no-op: the avenger already scored a
# kill that round, so their K is set regardless. A player dies at most once
# per round and their killer dies at most once, so size() is 0 or 1 per cell.
traded_index = pd.MultiIndex.from_product([rounds, player_ids],
                                          names=['round', 'teammate_traded'])
traded_per_round = (trades.groupby(['round', 'teammate_traded']).size()
                    .reindex(traded_index, fill_value=0))

# All four components share the same (round, player) grid, so concat lines
# them up row for row even though each carries a different index level name.
kast_index = pd.concat([kills_per_round, assists_per_round,
                        survive_round, traded_per_round], axis=1)
kast_index['total'] = kast_index.any(axis=1).astype(int)

# concat drops the clashing level names, so the player level comes back as
# 'level_1' when the index is pushed into columns.
kast_index = kast_index.reset_index().rename(columns={'level_1': 'player_id_fixed'})

kast = round(100 * (kast_index.groupby('player_id_fixed')['total'].sum()
                    .div(len(rounds))), 1).rename('kast %')

# ---------------------------------------------------------------------------
# Assemble the scoreboard
# ---------------------------------------------------------------------------

# fillna(0) covers anyone missing from a series entirely (no kills, no aces,
# no opening frags, no trades); the counting columns are all ints.
scoreboard = pd.concat([final_kills, deaths, assists, aces, open_frag, trade_kills],
                       axis=1).fillna(0).astype(int)

# Non-integer columns are attached after the astype(int) so they keep their type.
scoreboard['longest kill distance'] = longest_kill_dist
scoreboard['kast %'] = kast
scoreboard['best weapon'] = best_weapon
scoreboard['Headshot kill %'] = hs_kill_percent

# Move the id out of the index. concat drops the clashing index names, so the
# column arrives as 'index'. Cast to int as the source column is float via NaN.
scoreboard = scoreboard.reset_index().rename(columns={'index': 'player_id_fixed'})
scoreboard['player_id_fixed'] = scoreboard['player_id_fixed'].astype(int)

# K/D ratio. Guarded against a player who never died, which would otherwise
# give inf rather than a usable number.
scoreboard['K/D Ratio'] = round(
    scoreboard['kills'] / scoreboard['deaths'].where(scoreboard['deaths'] > 0), 2)

# Sort by kills, deaths, player_id desc. player_id_fixed is int by now, so this
# sorts numerically rather than as a string.
scoreboard = scoreboard.sort_values(by=['kills', 'deaths', 'player_id_fixed'],
                                    ascending=[False, False, False])

# Final column order, roughly matching how CS2 lays out its scoreboard.
scoreboard = scoreboard[['player_id_fixed', 'kills', 'deaths', 'K/D Ratio', 'assists',
                         'kast %', 'trade kills', 'Headshot kill %', 'aces',
                         'opening frags', 'best weapon', 'longest kill distance']]

# with pd.option_context('display.max_rows', None, 'display.max_columns', None):
#     print(scoreboard)
# Export the scoreboard to csv
scoreboard.to_csv("scoreboard_extended.csv", index=False)