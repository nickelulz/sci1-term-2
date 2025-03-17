import pandas as pd

# Load the CSV file (replace 'your_file.csv' with the actual file path)
df = pd.read_csv('data/richmond-raw.csv')

# Extract the 'Total Precip (mm)' column, ensuring NaN values are handled
precip_values = df['Total Precip (mm)'].fillna(0)  # Replace NaN with 0

# reclassify each of the values by their respective weather case:
# 
# where P is the total precipitation:
#
# P = 0           -> sunny (0)
# 0 < P <= 2      -> partly cloudy (1)
# 2 < P <= 5      -> cloudy (2)
# 5 < P <= 10     -> light rain (3)
# 10 < P <= 20    -> rain (4)
# P > 20          -> heavy rain cloudy (4)

def classify_precipitation(P):
    if P == 0:
        return 0  # Sunny
    elif 0 < P <= 2:
        return 1  # Partly cloudy
    elif 2 < P <= 5:
        return 2  # Cloudy
    elif 5 < P <= 10:
        return 3  # Light rain
    elif 10 < P <= 20:
        return 4  # Rain
    else:  # P > 20
        return 5  # Heavy rain

# Apply classification to each precipitation value
df['Weather Category'] = df['Total Precip (mm)'].apply(classify_precipitation)

# Save the reclassified values as a .dat file (comma-separated)
df['Weather Category'].to_csv('weather_categories.dat', index=False, header=False, sep=',')

print("File 'weather_categories.dat' has been saved.")

