#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 11:30:04 2024

@author: hermanellingsrud
"""

import os
from metocean_api import ts

#%% Set the output directory
output_dir = '/Users/hermanellingsrud/Documents/Skole/Masteroppgave/Koder/data'
os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists
os.chdir(output_dir)  # Change the current working directory to the output directory

#%% Retrieve wind data
# Initialize the TimeSeries object with the appropriate parameters
df_ts = ts.TimeSeries(
    lon=5.0,
    lat=56.867,
    start_time='2014-01-01',  # Start date for 10 years of data
    end_time='2023-02-28',    # End date
    product='NORA3_wind_sub'  # Data source for vertical wind profiles
)

# Import or load data (use the appropriate method based on your setup)
df_ts.import_data()  # Use import_data or load_data as needed

print('Finished retrieving wind data')

#%% Retrieve atmospheric temperature data at height

temp_ts = ts.TimeSeries(
    lon=5.0,
    lat=56.867,
    start_time='2014-01-01',  # Start date for 10 years of data
    end_time='2023-12-31',    # End date
    product='NORA3_atm3hr_sub'  # Use 3-hourly atmospheric data from NORA3
)

# Import or load data for the temperature object
temp_ts.import_data()  # Use import_data or load_data as needed

print('Finished retrieving atmospheric temperature data')

#%% Retrieve surface temperature data for January 2023
# Define the date range from January 2014 to December 2023
start_datetime = '2014-01-01'
end_datetime = '2023-12-31'  # Until the end of December 31, 2023

# Initialize the TimeSeries object for surface temperature data for the entire period
temp_ts_surface = ts.TimeSeries(
    lon=5.000, 
    lat=56.867,
    start_time=start_datetime, 
    end_time=end_datetime,
    product='NORA3_atm_sub'  # Use the hourly surface data product
)

# Import or load data for the surface temperature object
temp_ts_surface.import_data()  # Use import_data or load_data as needed

# Save the surface temperature data to a file
output_filename = 'surface_temperature_SNII_10_years.csv'
temp_ts_surface.data.to_csv(output_filename)

print('Finished saving surface temperature data')

#%% Retrieve wave data

output_dir = '/Users/hermanellingsrud/Documents/Skole/Masteroppgave/Koder/data/wave'
os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists
os.chdir(output_dir)  # Change the current working directory to the output directory

from metocean_api import ts

# Define the TimeSeries object
df_ts = ts.TimeSeries(
    lon=4.508,
    lat=59.195,
    start_time='2023-01-01',
    end_time='2023-01-30',
    product='NORA3_wave'
)

# Import data from thredds.met.no and save it as a CSV file
df_ts.import_data(save_csv=True)
# Load data from the local CSV file
df_ts.load_data(local_file=df_ts.datafile)
