import nfl_data_py as nfl
import pandas as pd
from datetime import datetime

# Get the current year dynamically
current_year = datetime.now().year

# Import seasonal rosters for the current year (player-season level info, including team, position, etc.)
seasonal_rosters = nfl.import_seasonal_rosters([current_year])

# Import weekly rosters for the current year (player-game level, including status like active/inactive)
weekly_rosters = nfl.import_weekly_rosters([current_year])

# Filter weekly rosters to the most recent week available (assuming data is up-to-date)
# Get the maximum week number
max_week = weekly_rosters['week'].max()
current_week_rosters = weekly_rosters[weekly_rosters['week'] == max_week]

# Print or display the rosters
print("Seasonal Rosters for All Teams:")
print(seasonal_rosters.head())  # Preview first few rows

print("\nCurrent Week Rosters for All Teams:")
print(current_week_rosters.head())  # Preview first few rows

# Optionally, save to CSV files
seasonal_rosters.to_csv('seasonal_rosters.csv', index=False)
current_week_rosters.to_csv('current_week_rosters.csv', index=False)
print("\nRosters saved to CSV files.")