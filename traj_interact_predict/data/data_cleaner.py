"""
flight_data_cleanup.py

Removes invalid flights from drone trajectory datasets based on altitude criteria.

This module scans `drone_states.csv` and identifies flight IDs for which at least
one timestep contains a `pos_z` value below a specified threshold (pos_z < 1).
Once such a condition is detected for a flight, the entire flight is marked
for removal.

All marked flight IDs are then deleted from both:
- `drone_states.csv`
- `drone_relations.csv`

The cleaned datasets overwrite the original CSV files to maintain consistency
between state and relation data.

Intended for preprocessing multi-agent drone trajectory data prior to
model training or analysis.
"""

import pandas as pd


# File paths
STATES_PATH = "data/drone_relations_v8/drone_states.csv"
RELATIONS_PATH = "data/drone_relations_v8/drone_relations.csv"

# Load CSV files
df_states = pd.read_csv(STATES_PATH)
df_relations = pd.read_csv(RELATIONS_PATH)

# Find flight_ids where ANY pos_z < 1
flight_ids_to_delete = df_states.groupby("flight_id")["pos_z"].apply(
    lambda z: (z < 1).any()
)

flight_ids_to_delete = flight_ids_to_delete[flight_ids_to_delete].index

print(
    f"Deleting {len(flight_ids_to_delete)} flight_id(s): {list(flight_ids_to_delete)}"
)

# Remove those flight_ids from both dataframes
df_states_cleaned = df_states[~df_states["flight_id"].isin(flight_ids_to_delete)]
df_relations_cleaned = df_relations[
    ~df_relations["flight_id"].isin(flight_ids_to_delete)
]

# Overwrite original CSV files
df_states_cleaned.to_csv(STATES_PATH, index=False)
df_relations_cleaned.to_csv(RELATIONS_PATH, index=False)

print("Cleanup complete. Files overwritten successfully.")
