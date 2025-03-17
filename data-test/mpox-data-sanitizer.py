import pandas as pd
import numpy as np

# Load the CSV file
df = pd.read_csv("data/monkeypox-worldwide-raw.csv")

# Ensure data is sorted by date for each country
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(['location', 'date'])

# Compute the daily change in total cases
df['new_cases_change'] = df.groupby('location')['new_cases'].diff().fillna(0)

# Convert changes to -1, 0, or 1
df['trend'] = np.sign(df['new_cases_change'])

# Create the output format
output = df.groupby('location')['trend'].apply(lambda x: ",".join(map(str, x.astype(int)))).reset_index()

# Convert to dictionary format for writing
data_dict = {row['location']: row['trend'] for _, row in output.iterrows()}

# Write to .dat file
with open("data/mpox-data-sanitized.dat", "w") as f:
    for country, trend in data_dict.items():
        f.write(f"{country}:{trend}\n")

print("File saved as data/mpox-data-sanitized.dat")

