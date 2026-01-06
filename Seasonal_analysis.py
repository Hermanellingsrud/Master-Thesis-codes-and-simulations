#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  4 16:22:39 2025

@author: hermanellingsrud
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

#%%

SNII = '/Users/hermanellingsrud/Documents/Skole/Masteroppgave/Koder/data/Sørvest_F/NORA3_wind_sub_lon5.0_lat56.867_20140101_20231231.csv'
Utsira = "/Users/hermanellingsrud/Documents/Skole/Masteroppgave/Koder/data/Vestavind_F/NORA3_wind_sub_lon4.5_lat59.2_20140101_20231231.csv" #Utsira

# Read the merged wind data CSV
SNII_wind = pd.read_csv(SNII, skiprows=16)
SNII_wind['time'] = pd.to_datetime(SNII_wind['time'])  # Convert 'time' to datetime

Utsira_wind = pd.read_csv(Utsira, skiprows=16)
Utsira_wind['time'] = pd.to_datetime(Utsira_wind['time'])  # Convert 'time' to datetime

#%%

# Filtrer data for 2023
SNII_wind_2023 = SNII_wind[(SNII_wind['time'] >= '2023-01-01') & (SNII_wind['time'] <= '2023-12-31')]
Utsira_wind_2023 = Utsira_wind[(Utsira_wind['time'] >= '2023-01-01') & (Utsira_wind['time'] <= '2023-12-31')]

# Filtrer tidene for 03:00:00, 12:00:00 og 21:00:00
SNII_wind_2023_filtered = SNII_wind_2023[SNII_wind_2023['time'].dt.hour.isin([3, 12, 21])]
Utsira_wind_2023_filtered = Utsira_wind_2023[Utsira_wind_2023['time'].dt.hour.isin([3, 12, 21])]


#%% Filtrering av data for 2023
SNII_profiles_03 = SNII_wind_2023_filtered[SNII_wind_2023_filtered['time'].dt.strftime('%H:%M:%S') == '03:00:00']
SNII_profiles_12 = SNII_wind_2023_filtered[SNII_wind_2023_filtered['time'].dt.strftime('%H:%M:%S') == '12:00:00']
SNII_profiles_21 = SNII_wind_2023_filtered[SNII_wind_2023_filtered['time'].dt.strftime('%H:%M:%S') == '21:00:00']

Utsira_profiles_03 = Utsira_wind_2023_filtered[Utsira_wind_2023_filtered['time'].dt.strftime('%H:%M:%S') == '03:00:00']
Utsira_profiles_12 = Utsira_wind_2023_filtered[Utsira_wind_2023_filtered['time'].dt.strftime('%H:%M:%S') == '12:00:00']
Utsira_profiles_21 = Utsira_wind_2023_filtered[Utsira_wind_2023_filtered['time'].dt.strftime('%H:%M:%S') == '21:00:00']




#%%
# Funksjon for å hente sesongdata for et spesifisert tidspunkt
def get_seasonal_profiles(profiles, time_of_day):
    """Hent sesongbaserte vindprofiler for et spesifisert tidspunkt."""
    profiles_filtered = profiles[profiles['time'].dt.strftime('%H:%M:%S') == time_of_day]
    
    # Definer sesonger
    winter = profiles_filtered[profiles_filtered['time'].dt.month.isin([12, 1, 2])]  # Desember, Januar, Februar
    spring = profiles_filtered[profiles_filtered['time'].dt.month.isin([3, 4, 5])]  # Mars, April, Mai
    summer = profiles_filtered[profiles_filtered['time'].dt.month.isin([6, 7, 8])]  # Juni, Juli, August
    autumn = profiles_filtered[profiles_filtered['time'].dt.month.isin([9, 10, 11])]  # September, Oktober, November
    
    return winter, spring, summer, autumn

# Hent sesongdata for SNII og Utsira klokka 03:00
winter_SNII_03, spring_SNII_03, summer_SNII_03, autumn_SNII_03 = get_seasonal_profiles(SNII_profiles_03, '03:00:00')
winter_Utsira_03, spring_Utsira_03, summer_Utsira_03, autumn_Utsira_03 = get_seasonal_profiles(Utsira_profiles_03, '03:00:00')

# Hent sesongdata for SNII og Utsira klokka 12:00
winter_SNII_12, spring_SNII_12, summer_SNII_12, autumn_SNII_12 = get_seasonal_profiles(SNII_profiles_12, '12:00:00')
winter_Utsira_12, spring_Utsira_12, summer_Utsira_12, autumn_Utsira_12 = get_seasonal_profiles(Utsira_profiles_12, '12:00:00')

# Hent sesongdata for SNII og Utsira klokka 21:00
winter_SNII_21, spring_SNII_21, summer_SNII_21, autumn_SNII_21 = get_seasonal_profiles(SNII_profiles_21, '21:00:00')
winter_Utsira_21, spring_Utsira_21, summer_Utsira_21, autumn_Utsira_21 = get_seasonal_profiles(Utsira_profiles_21, '21:00:00')

#%%
# Funksjon for å hente vindhastigheter for en spesifisert sesong, tid og plassering
def get_wind_speeds_for_season_time_location(profiles, season, time_of_day):
    """Hent vindhastigheter for en spesifisert sesong, tid på dagen og plassering."""
    # Filtrer sesongdataene
    profiles_filtered = season[season['time'].dt.strftime('%H:%M:%S') == time_of_day]
    
    # Hent vindhastighetene for de relevante høydene
    V_10 = profiles_filtered['wind_speed_10m'].values
    V_20 = profiles_filtered['wind_speed_20m'].values
    V_50 = profiles_filtered['wind_speed_50m'].values
    V_100 = profiles_filtered['wind_speed_100m'].values
    V_250 = profiles_filtered['wind_speed_250m'].values
    V_500 = profiles_filtered['wind_speed_500m'].values
    V_750 = profiles_filtered['wind_speed_750m'].values
    
    return V_10, V_20, V_50, V_100, V_250, V_500, V_750


#%%



# Hent vindhastigheter for SNII kl 03:00 for alle sesonger
V_10_winter_SNII_03, V_20_winter_SNII_03, V_50_winter_SNII_03, V_100_winter_SNII_03, V_250_winter_SNII_03, V_500_winter_SNII_03, V_750_winter_SNII_03 = get_wind_speeds_for_season_time_location(SNII_profiles_03, winter_SNII_03, '03:00:00')
V_10_spring_SNII_03, V_20_spring_SNII_03, V_50_spring_SNII_03, V_100_spring_SNII_03, V_250_spring_SNII_03, V_500_spring_SNII_03, V_750_spring_SNII_03 = get_wind_speeds_for_season_time_location(SNII_profiles_03, spring_SNII_03, '03:00:00')
V_10_summer_SNII_03, V_20_summer_SNII_03, V_50_summer_SNII_03, V_100_summer_SNII_03, V_250_summer_SNII_03, V_500_summer_SNII_03, V_750_summer_SNII_03 = get_wind_speeds_for_season_time_location(SNII_profiles_03, summer_SNII_03, '03:00:00')
V_10_autumn_SNII_03, V_20_autumn_SNII_03, V_50_autumn_SNII_03, V_100_autumn_SNII_03, V_250_autumn_SNII_03, V_500_autumn_SNII_03, V_750_autumn_SNII_03 = get_wind_speeds_for_season_time_location(SNII_profiles_03, autumn_SNII_03, '03:00:00')

# Hent vindhastigheter for Utsira kl 03:00 for alle sesonger
V_10_winter_Utsira_03, V_20_winter_Utsira_03, V_50_winter_Utsira_03, V_100_winter_Utsira_03, V_250_winter_Utsira_03, V_500_winter_Utsira_03, V_750_winter_Utsira_03 = get_wind_speeds_for_season_time_location(Utsira_profiles_03, winter_Utsira_03, '03:00:00')
V_10_spring_Utsira_03, V_20_spring_Utsira_03, V_50_spring_Utsira_03, V_100_spring_Utsira_03, V_250_spring_Utsira_03, V_500_spring_Utsira_03, V_750_spring_Utsira_03 = get_wind_speeds_for_season_time_location(Utsira_profiles_03, spring_Utsira_03, '03:00:00')
V_10_summer_Utsira_03, V_20_summer_Utsira_03, V_50_summer_Utsira_03, V_100_summer_Utsira_03, V_250_summer_Utsira_03, V_500_summer_Utsira_03, V_750_summer_Utsira_03 = get_wind_speeds_for_season_time_location(Utsira_profiles_03, summer_Utsira_03, '03:00:00')
V_10_autumn_Utsira_03, V_20_autumn_Utsira_03, V_50_autumn_Utsira_03, V_100_autumn_Utsira_03, V_250_autumn_Utsira_03, V_500_autumn_Utsira_03, V_750_autumn_Utsira_03 = get_wind_speeds_for_season_time_location(Utsira_profiles_03, autumn_Utsira_03, '03:00:00')

# Hent vindhastigheter for SNII kl 12:00 for alle sesonger
V_10_winter_SNII_12, V_20_winter_SNII_12, V_50_winter_SNII_12, V_100_winter_SNII_12, V_250_winter_SNII_12, V_500_winter_SNII_12, V_750_winter_SNII_12 = get_wind_speeds_for_season_time_location(SNII_profiles_12, winter_SNII_12, '12:00:00')
V_10_spring_SNII_12, V_20_spring_SNII_12, V_50_spring_SNII_12, V_100_spring_SNII_12, V_250_spring_SNII_12, V_500_spring_SNII_12, V_750_spring_SNII_12 = get_wind_speeds_for_season_time_location(SNII_profiles_12, spring_SNII_12, '12:00:00')
V_10_summer_SNII_12, V_20_summer_SNII_12, V_50_summer_SNII_12, V_100_summer_SNII_12, V_250_summer_SNII_12, V_500_summer_SNII_12, V_750_summer_SNII_12 = get_wind_speeds_for_season_time_location(SNII_profiles_12, summer_SNII_12, '12:00:00')
V_10_autumn_SNII_12, V_20_autumn_SNII_12, V_50_autumn_SNII_12, V_100_autumn_SNII_12, V_250_autumn_SNII_12, V_500_autumn_SNII_12, V_750_autumn_SNII_12 = get_wind_speeds_for_season_time_location(SNII_profiles_12, autumn_SNII_12, '12:00:00')

# Hent vindhastigheter for Utsira kl 12:00 for alle sesonger
V_10_winter_Utsira_12, V_20_winter_Utsira_12, V_50_winter_Utsira_12, V_100_winter_Utsira_12, V_250_winter_Utsira_12, V_500_winter_Utsira_12, V_750_winter_Utsira_12 = get_wind_speeds_for_season_time_location(Utsira_profiles_12, winter_Utsira_12, '12:00:00')
V_10_spring_Utsira_12, V_20_spring_Utsira_12, V_50_spring_Utsira_12, V_100_spring_Utsira_12, V_250_spring_Utsira_12, V_500_spring_Utsira_12, V_750_spring_Utsira_12 = get_wind_speeds_for_season_time_location(Utsira_profiles_12, spring_Utsira_12, '12:00:00')
V_10_summer_Utsira_12, V_20_summer_Utsira_12, V_50_summer_Utsira_12, V_100_summer_Utsira_12, V_250_summer_Utsira_12, V_500_summer_Utsira_12, V_750_summer_Utsira_12 = get_wind_speeds_for_season_time_location(Utsira_profiles_12, summer_Utsira_12, '12:00:00')
V_10_autumn_Utsira_12, V_20_autumn_Utsira_12, V_50_autumn_Utsira_12, V_100_autumn_Utsira_12, V_250_autumn_Utsira_12, V_500_autumn_Utsira_12, V_750_autumn_Utsira_12 = get_wind_speeds_for_season_time_location(Utsira_profiles_12, autumn_Utsira_12, '12:00:00')

# Hent vindhastigheter for SNII kl 21:00 for alle sesonger
V_10_winter_SNII_21, V_20_winter_SNII_21, V_50_winter_SNII_21, V_100_winter_SNII_21, V_250_winter_SNII_21, V_500_winter_SNII_21, V_750_winter_SNII_21 = get_wind_speeds_for_season_time_location(SNII_profiles_21, winter_SNII_21, '21:00:00')
V_10_spring_SNII_21, V_20_spring_SNII_21, V_50_spring_SNII_21, V_100_spring_SNII_21, V_250_spring_SNII_21, V_500_spring_SNII_21, V_750_spring_SNII_21 = get_wind_speeds_for_season_time_location(SNII_profiles_21, spring_SNII_21, '21:00:00')
V_10_summer_SNII_21, V_20_summer_SNII_21, V_50_summer_SNII_21, V_100_summer_SNII_21, V_250_summer_SNII_21, V_500_summer_SNII_21, V_750_summer_SNII_21 = get_wind_speeds_for_season_time_location(SNII_profiles_21, summer_SNII_21, '21:00:00')
V_10_autumn_SNII_21, V_20_autumn_SNII_21, V_50_autumn_SNII_21, V_100_autumn_SNII_21, V_250_autumn_SNII_21, V_500_autumn_SNII_21, V_750_autumn_SNII_21 = get_wind_speeds_for_season_time_location(SNII_profiles_21, autumn_SNII_21, '21:00:00')

# Hent vindhastigheter for Utsira kl 21:00 for alle sesonger
V_10_winter_Utsira_21, V_20_winter_Utsira_21, V_50_winter_Utsira_21, V_100_winter_Utsira_21, V_250_winter_Utsira_21, V_500_winter_Utsira_21, V_750_winter_Utsira_21 = get_wind_speeds_for_season_time_location(Utsira_profiles_21, winter_Utsira_21, '21:00:00')
V_10_spring_Utsira_21, V_20_spring_Utsira_21, V_50_spring_Utsira_21, V_100_spring_Utsira_21, V_250_spring_Utsira_21, V_500_spring_Utsira_21, V_750_spring_Utsira_21 = get_wind_speeds_for_season_time_location(Utsira_profiles_21, spring_Utsira_21, '21:00:00')
V_10_summer_Utsira_21, V_20_summer_Utsira_21, V_50_summer_Utsira_21, V_100_summer_Utsira_21, V_250_summer_Utsira_21, V_500_summer_Utsira_21, V_750_summer_Utsira_21 = get_wind_speeds_for_season_time_location(Utsira_profiles_21, summer_Utsira_21, '21:00:00')
V_10_autumn_Utsira_21, V_20_autumn_Utsira_21, V_50_autumn_Utsira_21, V_100_autumn_Utsira_21, V_250_autumn_Utsira_21, V_500_autumn_Utsira_21, V_750_autumn_Utsira_21 = get_wind_speeds_for_season_time_location(Utsira_profiles_21, autumn_Utsira_21, '21:00:00')


def make_profile(v10, v20, v50, v100, v250, v500, v750):
    height = [10, 20, 50, 100, 250, 500, 750]  # Høydene i meter
    wind_speeds = [v10, v20, v50, v100, v250, v500, v750]  # Vindhastigheter for de respektive høydene
    
    # Lag en dictionary med høyde som nøkkel og vindhastighet som verdi
    profile = dict(zip(height, wind_speeds))
    
    return profile

# Vindprofiler for SNII (03:00) for vinter, vår, sommer og høst
winter_SNII_03_profile = make_profile(V_10_winter_SNII_03, V_20_winter_SNII_03, V_50_winter_SNII_03, V_100_winter_SNII_03, V_250_winter_SNII_03, V_500_winter_SNII_03, V_750_winter_SNII_03)
spring_SNII_03_profile = make_profile(V_10_spring_SNII_03, V_20_spring_SNII_03, V_50_spring_SNII_03, V_100_spring_SNII_03, V_250_spring_SNII_03, V_500_spring_SNII_03, V_750_spring_SNII_03)
summer_SNII_03_profile = make_profile(V_10_summer_SNII_03, V_20_summer_SNII_03, V_50_summer_SNII_03, V_100_summer_SNII_03, V_250_summer_SNII_03, V_500_summer_SNII_03, V_750_summer_SNII_03)
autumn_SNII_03_profile = make_profile(V_10_autumn_SNII_03, V_20_autumn_SNII_03, V_50_autumn_SNII_03, V_100_autumn_SNII_03, V_250_autumn_SNII_03, V_500_autumn_SNII_03, V_750_autumn_SNII_03)

# Vindprofiler for Utsira (03:00) for vinter, vår, sommer og høst
winter_Utsira_03_profile = make_profile(V_10_winter_Utsira_03, V_20_winter_Utsira_03, V_50_winter_Utsira_03, V_100_winter_Utsira_03, V_250_winter_Utsira_03, V_500_winter_Utsira_03, V_750_winter_Utsira_03)
spring_Utsira_03_profile = make_profile(V_10_spring_Utsira_03, V_20_spring_Utsira_03, V_50_spring_Utsira_03, V_100_spring_Utsira_03, V_250_spring_Utsira_03, V_500_spring_Utsira_03, V_750_spring_Utsira_03)
summer_Utsira_03_profile = make_profile(V_10_summer_Utsira_03, V_20_summer_Utsira_03, V_50_summer_Utsira_03, V_100_summer_Utsira_03, V_250_summer_Utsira_03, V_500_summer_Utsira_03, V_750_summer_Utsira_03)
autumn_Utsira_03_profile = make_profile(V_10_autumn_Utsira_03, V_20_autumn_Utsira_03, V_50_autumn_Utsira_03, V_100_autumn_Utsira_03, V_250_autumn_Utsira_03, V_500_autumn_Utsira_03, V_750_autumn_Utsira_03)

# Vindprofiler for SNII (12:00) for vinter, vår, sommer og høst
winter_SNII_12_profile = make_profile(V_10_winter_SNII_12, V_20_winter_SNII_12, V_50_winter_SNII_12, V_100_winter_SNII_12, V_250_winter_SNII_12, V_500_winter_SNII_12, V_750_winter_SNII_12)
spring_SNII_12_profile = make_profile(V_10_spring_SNII_12, V_20_spring_SNII_12, V_50_spring_SNII_12, V_100_spring_SNII_12, V_250_spring_SNII_12, V_500_spring_SNII_12, V_750_spring_SNII_12)
summer_SNII_12_profile = make_profile(V_10_summer_SNII_12, V_20_summer_SNII_12, V_50_summer_SNII_12, V_100_summer_SNII_12, V_250_summer_SNII_12, V_500_summer_SNII_12, V_750_summer_SNII_12)
autumn_SNII_12_profile = make_profile(V_10_autumn_SNII_12, V_20_autumn_SNII_12, V_50_autumn_SNII_12, V_100_autumn_SNII_12, V_250_autumn_SNII_12, V_500_autumn_SNII_12, V_750_autumn_SNII_12)

# Vindprofiler for Utsira (12:00) for vinter, vår, sommer og høst
winter_Utsira_12_profile = make_profile(V_10_winter_Utsira_12, V_20_winter_Utsira_12, V_50_winter_Utsira_12, V_100_winter_Utsira_12, V_250_winter_Utsira_12, V_500_winter_Utsira_12, V_750_winter_Utsira_12)
spring_Utsira_12_profile = make_profile(V_10_spring_Utsira_12, V_20_spring_Utsira_12, V_50_spring_Utsira_12, V_100_spring_Utsira_12, V_250_spring_Utsira_12, V_500_spring_Utsira_12, V_750_spring_Utsira_12)
summer_Utsira_12_profile = make_profile(V_10_summer_Utsira_12, V_20_summer_Utsira_12, V_50_summer_Utsira_12, V_100_summer_Utsira_12, V_250_summer_Utsira_12, V_500_summer_Utsira_12, V_750_summer_Utsira_12)
autumn_Utsira_12_profile = make_profile(V_10_autumn_Utsira_12, V_20_autumn_Utsira_12, V_50_autumn_Utsira_12, V_100_autumn_Utsira_12, V_250_autumn_Utsira_12, V_500_autumn_Utsira_12, V_750_autumn_Utsira_12)

# Vindprofiler for SNII (21:00) for vinter, vår, sommer og høst
winter_SNII_21_profile = make_profile(V_10_winter_SNII_21, V_20_winter_SNII_21, V_50_winter_SNII_21, V_100_winter_SNII_21, V_250_winter_SNII_21, V_500_winter_SNII_21, V_750_winter_SNII_21)
spring_SNII_21_profile = make_profile(V_10_spring_SNII_21, V_20_spring_SNII_21, V_50_spring_SNII_21, V_100_spring_SNII_21, V_250_spring_SNII_21, V_500_spring_SNII_21, V_750_spring_SNII_21)
summer_SNII_21_profile = make_profile(V_10_summer_SNII_21, V_20_summer_SNII_21, V_50_summer_SNII_21, V_100_summer_SNII_21, V_250_summer_SNII_21, V_500_summer_SNII_21, V_750_summer_SNII_21)
autumn_SNII_21_profile = make_profile(V_10_autumn_SNII_21, V_20_autumn_SNII_21, V_50_autumn_SNII_21, V_100_autumn_SNII_21, V_250_autumn_SNII_21, V_500_autumn_SNII_21, V_750_autumn_SNII_21)

# Vindprofiler for Utsira (21:00) for vinter, vår, sommer og høst
winter_Utsira_21_profile = make_profile(V_10_winter_Utsira_21, V_20_winter_Utsira_21, V_50_winter_Utsira_21, V_100_winter_Utsira_21, V_250_winter_Utsira_21, V_500_winter_Utsira_21, V_750_winter_Utsira_21)
spring_Utsira_21_profile = make_profile(V_10_spring_Utsira_21, V_20_spring_Utsira_21, V_50_spring_Utsira_21, V_100_spring_Utsira_21, V_250_spring_Utsira_21, V_500_spring_Utsira_21, V_750_spring_Utsira_21)
summer_Utsira_21_profile = make_profile(V_10_summer_Utsira_21, V_20_summer_Utsira_21, V_50_summer_Utsira_21, V_100_summer_Utsira_21, V_250_summer_Utsira_21, V_500_summer_Utsira_21, V_750_summer_Utsira_21)
autumn_Utsira_21_profile = make_profile(V_10_autumn_Utsira_21, V_20_autumn_Utsira_21, V_50_autumn_Utsira_21, V_100_autumn_Utsira_21, V_250_autumn_Utsira_21, V_500_autumn_Utsira_21, V_750_autumn_Utsira_21)


def profil(values):
    profile_list = list(values.values())
    all_profiles = []
    for i in range(len(profile_list[0])):
        profile = [profile_list[0][i],profile_list[1][i],profile_list[2][i],profile_list[3][i],profile_list[4][i],profile_list[5][i],profile_list[6][i]]
        all_profiles.append(profile)
    return all_profiles
        
winter_SNII_03_profile = profil(winter_SNII_03_profile)
spring_SNII_03_profile = profil(spring_SNII_03_profile)
summer_SNII_03_profile = profil(summer_SNII_03_profile)
autumn_SNII_03_profile = profil(autumn_SNII_03_profile)
winter_Utsira_03_profile = profil(winter_Utsira_03_profile)
spring_Utsira_03_profile = profil(spring_Utsira_03_profile)
summer_Utsira_03_profile = profil(summer_Utsira_03_profile)
autumn_Utsira_03_profile = profil(autumn_Utsira_03_profile)
winter_SNII_12_profile = profil(winter_SNII_12_profile)
spring_SNII_12_profile = profil(spring_SNII_12_profile)
summer_SNII_12_profile = profil(summer_SNII_12_profile)
autumn_SNII_12_profile = profil(autumn_SNII_12_profile)
winter_Utsira_12_profile = profil(winter_Utsira_12_profile)
spring_Utsira_12_profile = profil(spring_Utsira_12_profile)
summer_Utsira_12_profile = profil(summer_Utsira_12_profile)
autumn_Utsira_12_profile = profil(autumn_Utsira_12_profile)
winter_SNII_21_profile = profil(winter_SNII_21_profile)
spring_SNII_21_profile = profil(spring_SNII_21_profile)
summer_SNII_21_profile = profil(summer_SNII_21_profile)
autumn_SNII_21_profile = profil(autumn_SNII_21_profile)
winter_Utsira_21_profile = profil(winter_Utsira_21_profile)
spring_Utsira_21_profile = profil(spring_Utsira_21_profile)
summer_Utsira_21_profile = profil(summer_Utsira_21_profile)
autumn_Utsira_21_profile = profil(autumn_Utsira_21_profile)
    


#%%
seasons = ['winter', 'spring', 'summer', 'autumn']
times = ['03', '12', '21']
locations = ['SNII', 'Utsira']

# Høydeakse
heights = [10, 20, 50, 100, 250, 500, 750]

# Lag subplot
fig, axs = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
axs = axs.flatten()

for i, season in enumerate(seasons):
    ax = axs[i]
    for location in locations:
        for time in times:
            var_name = f"{season}_{location}_{time}_profile"
            profiles = globals()[var_name]  # liste av profiler, hver = liste med hastigheter

            # Gjør om til numpy array
            arr = np.array(profiles)  # shape (n_profiler, n_heights)

            # Beregn gjennomsnitt per høyde
            mean_profile = arr.mean(axis=0)

            # Velg linjetype og farge
            linestyle = {'03': '-', '12': '--', '21': ':'}[time]
            color = {'SNII': 'tab:blue', 'Utsira': 'tab:orange'}[location]

            # Plott profil
            ax.plot(mean_profile,heights,
                    label=f"{location} {time}:00:00",
                    linestyle=linestyle,
                    color=color)

    ax.set_title(season.capitalize(), fontsize=18)
    ax.set_ylabel('Height (m)', fontsize=14)
    ax.set_xlabel('Mean Wind Speed (m/s)', fontsize=14)
    ax.legend(fontsize=13)
    ax.tick_params(axis='both', labelsize=14)
    ax.grid(True)


#plt.suptitle('Mean Wind Profiles by Season', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('/Users/hermanellingsrud/Documents/Skole/Masteroppgave/Koder/figures_master/Sesonger/sesong_profiler.png', dpi = 300)
plt.show()
#%%
def hub_speed(v100,v250):
    v150 = v100 + (v250 - v100) * ((150 - 100) / (250 - 100))
    return v150

winter_SNII_03_hub = hub_speed(V_100_winter_SNII_03, V_250_winter_SNII_03)

# For Winter SNII 12:00
winter_SNII_12_hub = hub_speed(V_100_winter_SNII_12, V_250_winter_SNII_12)

# For Winter SNII 21:00
winter_SNII_21_hub = hub_speed(V_100_winter_SNII_21, V_250_winter_SNII_21)

# For Winter Utsira 03:00
winter_Utsira_03_hub = hub_speed(V_100_winter_Utsira_03, V_250_winter_Utsira_03)

# For Winter Utsira 12:00
winter_Utsira_12_hub = hub_speed(V_100_winter_Utsira_12, V_250_winter_Utsira_12)

# For Winter Utsira 21:00
winter_Utsira_21_hub = hub_speed(V_100_winter_Utsira_21, V_250_winter_Utsira_21)

# For Spring SNII 03:00
spring_SNII_03_hub = hub_speed(V_100_spring_SNII_03, V_250_spring_SNII_03)

# For Spring SNII 12:00
spring_SNII_12_hub = hub_speed(V_100_spring_SNII_12, V_250_spring_SNII_12)

# For Spring SNII 21:00
spring_SNII_21_hub = hub_speed(V_100_spring_SNII_21, V_250_spring_SNII_21)

# For Spring Utsira 03:00
spring_Utsira_03_hub = hub_speed(V_100_spring_Utsira_03, V_250_spring_Utsira_03)

# For Spring Utsira 12:00
spring_Utsira_12_hub = hub_speed(V_100_spring_Utsira_12, V_250_spring_Utsira_12)

# For Spring Utsira 21:00
spring_Utsira_21_hub = hub_speed(V_100_spring_Utsira_21, V_250_spring_Utsira_21)

# For Summer SNII 03:00
summer_SNII_03_hub = hub_speed(V_100_summer_SNII_03, V_250_summer_SNII_03)

# For Summer SNII 12:00
summer_SNII_12_hub = hub_speed(V_100_summer_SNII_12, V_250_summer_SNII_12)

# For Summer SNII 21:00
summer_SNII_21_hub = hub_speed(V_100_summer_SNII_21, V_250_summer_SNII_21)

# For Summer Utsira 03:00
summer_Utsira_03_hub = hub_speed(V_100_summer_Utsira_03, V_250_summer_Utsira_03)

# For Summer Utsira 12:00
summer_Utsira_12_hub = hub_speed(V_100_summer_Utsira_12, V_250_summer_Utsira_12)

# For Summer Utsira 21:00
summer_Utsira_21_hub = hub_speed(V_100_summer_Utsira_21, V_250_summer_Utsira_21)

# For Autumn SNII 03:00
autumn_SNII_03_hub = hub_speed(V_100_autumn_SNII_03, V_250_autumn_SNII_03)

# For Autumn SNII 12:00
autumn_SNII_12_hub = hub_speed(V_100_autumn_SNII_12, V_250_autumn_SNII_12)

# For Autumn SNII 21:00
autumn_SNII_21_hub = hub_speed(V_100_autumn_SNII_21, V_250_autumn_SNII_21)

# For Autumn Utsira 03:00
autumn_Utsira_03_hub = hub_speed(V_100_autumn_Utsira_03, V_250_autumn_Utsira_03)

# For Autumn Utsira 12:00
autumn_Utsira_12_hub = hub_speed(V_100_autumn_Utsira_12, V_250_autumn_Utsira_12)

# For Autumn Utsira 21:00
autumn_Utsira_21_hub = hub_speed(V_100_autumn_Utsira_21, V_250_autumn_Utsira_21)




#%%


def calculate_alpha_least_squares(wind_speeds, z_r=10):
    """
    Beregn vindskjæringseksponenten alpha ved hjelp av lineær regresjon over flere høyder.
    
    Parameters:
    - heights: liste eller array med høydenivåer (f.eks. [10, 20, 50, 100, 250])
    - wind_speeds: tilsvarende vindhastigheter i samme rekkefølge
    - z_r: referansehøyde (default: 10 m)
    
    Returns:
    - alpha: estimert vindskjæringseksponent
    """
    # Sørg for at input er numpy arrays
    heights = [10, 20, 50, 100, 250]
    heights = np.array(heights)
    wind_speeds = np.array(wind_speeds[0:5])
    
    # Fjern eventuelle NaN-verdier
    valid = ~np.isnan(wind_speeds)
    heights = heights[valid]
    wind_speeds = wind_speeds[valid]
    
    # Transformasjon til logaritmisk form
    x = np.log(heights / z_r)
    y = np.log(wind_speeds)

    # Lineær regresjon: y = alpha * x + intercept
    alpha, intercept = np.polyfit(x, y, 1)
    return alpha

def get_alpha_for_profiles_least_squares_from_list(profiles_list):
    alpha_values = []
    for row in profiles_list:
        wind_speeds = row[:5]  # antar at de fem første verdiene er vindhastigheter
        alpha = calculate_alpha_least_squares(wind_speeds)
        alpha_values.append(alpha)
    return alpha_values



alpha_winter_SNII_03 = get_alpha_for_profiles_least_squares_from_list(winter_SNII_03_profile)
alpha_spring_SNII_03 = get_alpha_for_profiles_least_squares_from_list(spring_SNII_03_profile)
alpha_summer_SNII_03 = get_alpha_for_profiles_least_squares_from_list(summer_SNII_03_profile)
alpha_autumn_SNII_03 = get_alpha_for_profiles_least_squares_from_list(autumn_SNII_03_profile)

alpha_winter_Utsira_03 = get_alpha_for_profiles_least_squares_from_list(winter_Utsira_03_profile)
alpha_spring_Utsira_03 = get_alpha_for_profiles_least_squares_from_list(spring_Utsira_03_profile)
alpha_summer_Utsira_03 = get_alpha_for_profiles_least_squares_from_list(summer_Utsira_03_profile)
alpha_autumn_Utsira_03 = get_alpha_for_profiles_least_squares_from_list(autumn_Utsira_03_profile)

alpha_winter_SNII_12 = get_alpha_for_profiles_least_squares_from_list(winter_SNII_12_profile)
alpha_spring_SNII_12 = get_alpha_for_profiles_least_squares_from_list(spring_SNII_12_profile)
alpha_summer_SNII_12 = get_alpha_for_profiles_least_squares_from_list(summer_SNII_12_profile)
alpha_autumn_SNII_12 = get_alpha_for_profiles_least_squares_from_list(autumn_SNII_12_profile)

alpha_winter_Utsira_12 = get_alpha_for_profiles_least_squares_from_list(winter_Utsira_12_profile)
alpha_spring_Utsira_12 = get_alpha_for_profiles_least_squares_from_list(spring_Utsira_12_profile)
alpha_summer_Utsira_12 = get_alpha_for_profiles_least_squares_from_list(summer_Utsira_12_profile)
alpha_autumn_Utsira_12 = get_alpha_for_profiles_least_squares_from_list(autumn_Utsira_12_profile)

alpha_winter_SNII_21 = get_alpha_for_profiles_least_squares_from_list(winter_SNII_21_profile)
alpha_spring_SNII_21 = get_alpha_for_profiles_least_squares_from_list(spring_SNII_21_profile)
alpha_summer_SNII_21 = get_alpha_for_profiles_least_squares_from_list(summer_SNII_21_profile)
alpha_autumn_SNII_21 = get_alpha_for_profiles_least_squares_from_list(autumn_SNII_21_profile)

alpha_winter_Utsira_21 = get_alpha_for_profiles_least_squares_from_list(winter_Utsira_21_profile)
alpha_spring_Utsira_21 = get_alpha_for_profiles_least_squares_from_list(spring_Utsira_21_profile)
alpha_summer_Utsira_21 = get_alpha_for_profiles_least_squares_from_list(summer_Utsira_21_profile)
alpha_autumn_Utsira_21 = get_alpha_for_profiles_least_squares_from_list(autumn_Utsira_21_profile)


#%%

# Beregn gjennomsnittet av alfa for hver sesong og tidspunkt
average_alpha_winter_SNII_03 = np.mean(alpha_winter_SNII_03)
average_alpha_spring_SNII_03 = np.mean(alpha_spring_SNII_03)
average_alpha_summer_SNII_03 = np.mean(alpha_summer_SNII_03)
average_alpha_autumn_SNII_03 = np.mean(alpha_autumn_SNII_03)

average_alpha_winter_Utsira_03 = np.mean(alpha_winter_Utsira_03)
average_alpha_spring_Utsira_03 = np.mean(alpha_spring_Utsira_03)
average_alpha_summer_Utsira_03 = np.mean(alpha_summer_Utsira_03)
average_alpha_autumn_Utsira_03 = np.mean(alpha_autumn_Utsira_03)

average_alpha_winter_SNII_12 = np.mean(alpha_winter_SNII_12)
average_alpha_spring_SNII_12 = np.mean(alpha_spring_SNII_12)
average_alpha_summer_SNII_12 = np.mean(alpha_summer_SNII_12)
average_alpha_autumn_SNII_12 = np.mean(alpha_autumn_SNII_12)

average_alpha_winter_Utsira_12 = np.mean(alpha_winter_Utsira_12)
average_alpha_spring_Utsira_12 = np.mean(alpha_spring_Utsira_12)
average_alpha_summer_Utsira_12 = np.mean(alpha_summer_Utsira_12)
average_alpha_autumn_Utsira_12 = np.mean(alpha_autumn_Utsira_12)

average_alpha_winter_SNII_21 = np.mean(alpha_winter_SNII_21)
average_alpha_spring_SNII_21 = np.mean(alpha_spring_SNII_21)
average_alpha_summer_SNII_21 = np.mean(alpha_summer_SNII_21)
average_alpha_autumn_SNII_21 = np.mean(alpha_autumn_SNII_21)

average_alpha_winter_Utsira_21 = np.mean(alpha_winter_Utsira_21)
average_alpha_spring_Utsira_21 = np.mean(alpha_spring_Utsira_21)
average_alpha_summer_Utsira_21 = np.mean(alpha_summer_Utsira_21)
average_alpha_autumn_Utsira_21 = np.mean(alpha_autumn_Utsira_21)

# Lag en liste med gjennomsnittene og deres tilhørende etiketter
alpha_averages = [
    ('Winter SNII 03:00', average_alpha_winter_SNII_03),
    ('Winter SNII 12:00', average_alpha_winter_SNII_12),
    ('Winter SNII 21:00', average_alpha_winter_SNII_21),
    ('Winter Utsira 03:00', average_alpha_winter_Utsira_03),
    ('Winter Utsira 12:00', average_alpha_winter_Utsira_12),
    ('Winter Utsira 21:00', average_alpha_winter_Utsira_21),

    ('Spring SNII 03:00', average_alpha_spring_SNII_03),
    ('Spring SNII 12:00', average_alpha_spring_SNII_12),
    ('Spring SNII 21:00', average_alpha_spring_SNII_21),
    ('Spring Utsira 03:00', average_alpha_spring_Utsira_03),
    ('Spring Utsira 12:00', average_alpha_spring_Utsira_12),
    ('Spring Utsira 21:00', average_alpha_spring_Utsira_21),

    ('Summer SNII 03:00', average_alpha_summer_SNII_03),
    ('Summer SNII 12:00', average_alpha_summer_SNII_12),
    ('Summer SNII 21:00', average_alpha_summer_SNII_21),
    ('Summer Utsira 03:00', average_alpha_summer_Utsira_03),
    ('Summer Utsira 12:00', average_alpha_summer_Utsira_12),
    ('Summer Utsira 21:00', average_alpha_summer_Utsira_21),

    ('Autumn SNII 03:00', average_alpha_autumn_SNII_03),
    ('Autumn SNII 12:00', average_alpha_autumn_SNII_12),
    ('Autumn SNII 21:00', average_alpha_autumn_SNII_21),
    ('Autumn Utsira 03:00', average_alpha_autumn_Utsira_03),
    ('Autumn Utsira 12:00', average_alpha_autumn_Utsira_12),
    ('Autumn Utsira 21:00', average_alpha_autumn_Utsira_21),
]


# Sorter listen basert på gjennomsnittet (fra høyeste til laveste)
sorted_alpha_averages = sorted(alpha_averages, key=lambda x: x[1], reverse=True)

# Print ut de sorterte gjennomsnittene
print("Sorted alpha averages:")
for label, avg in sorted_alpha_averages:
    print(f"{label}: {avg:.4f}")

# Beregn gjennomsnittet av hver hub-speed
hub_averages = [
    ("winter_SNII_03_hub", np.mean(winter_SNII_03_hub)),
    ("winter_SNII_12_hub", np.mean(winter_SNII_12_hub)),
    ("winter_SNII_21_hub", np.mean(winter_SNII_21_hub)),
    ("winter_Utsira_03_hub", np.mean(winter_Utsira_03_hub)),
    ("winter_Utsira_12_hub", np.mean(winter_Utsira_12_hub)),
    ("winter_Utsira_21_hub", np.mean(winter_Utsira_21_hub)),
    ("spring_SNII_03_hub", np.mean(spring_SNII_03_hub)),
    ("spring_SNII_12_hub", np.mean(spring_SNII_12_hub)),
    ("spring_SNII_21_hub", np.mean(spring_SNII_21_hub)),
    ("spring_Utsira_03_hub", np.mean(spring_Utsira_03_hub)),
    ("spring_Utsira_12_hub", np.mean(spring_Utsira_12_hub)),
    ("spring_Utsira_21_hub", np.mean(spring_Utsira_21_hub)),
    ("summer_SNII_03_hub", np.mean(summer_SNII_03_hub)),
    ("summer_SNII_12_hub", np.mean(summer_SNII_12_hub)),
    ("summer_SNII_21_hub", np.mean(summer_SNII_21_hub)),
    ("summer_Utsira_03_hub", np.mean(summer_Utsira_03_hub)),
    ("summer_Utsira_12_hub", np.mean(summer_Utsira_12_hub)),
    ("summer_Utsira_21_hub", np.mean(summer_Utsira_21_hub)),
    ("autumn_SNII_03_hub", np.mean(autumn_SNII_03_hub)),
    ("autumn_SNII_12_hub", np.mean(autumn_SNII_12_hub)),
    ("autumn_SNII_21_hub", np.mean(autumn_SNII_21_hub)),
    ("autumn_Utsira_03_hub", np.mean(autumn_Utsira_03_hub)),
    ("autumn_Utsira_12_hub", np.mean(autumn_Utsira_12_hub)),
    ("autumn_Utsira_21_hub", np.mean(autumn_Utsira_21_hub)),
]

# Sorter listen fra høyest til lavest gjennomsnitt
sorted_hub_averages = sorted(hub_averages, key=lambda x: x[1], reverse=True)

# Print ut de sorterte gjennomsnittene
print("Sorted hub speed averages:")
for label, avg in sorted_hub_averages:
    print(f"{label}: {avg:.4f}")



#%%


# Første plot (Hub Speed)  
labels, values = zip(*hub_averages)
snii_values = [val for label, val in hub_averages if 'SNII' in label]
utsira_values = [val for label, val in hub_averages if 'Utsira' in label]
x_snii = np.arange(0, len(snii_values))
x_utsira = np.arange(0, len(utsira_values))

# Første plot (Hub Speed)
plt.figure(figsize=(14, 6))
plt.plot(x_snii, snii_values, marker='o', label='SNII', color='tab:blue')
plt.plot(x_utsira, utsira_values, marker='s', label='Utsira', color='tab:orange')
ticks = [
    "Winter 03:00:00", "Winter 12:00:00", "Winter 21:00:00",
    "Spring 03:00:00", "Spring 12:00:00", "Spring 21:00:00",
    "Summer 03:00:00", "Summer 12:00:00", "Summer 21:00:00",
    "Autumn 03:00:00", "Autumn 12:00:00", "Autumn 21:00:00"
]
plt.xticks(x_snii, ticks, rotation=45, fontsize=18)
plt.yticks(fontsize=18)
plt.ylabel('Hub Speed (m/s)', fontsize=18)
plt.title('Hub Speed — SNII vs Utsira', fontsize=18)
plt.legend(fontsize=18)
plt.grid()
plt.tight_layout()
plt.savefig('/Users/hermanellingsrud/Documents/Skole/Masteroppgave/Koder/figures_master/Sesonger/hub_wind_speed.png', dpi = 300)
plt.show()

# Andre plot (Alpha)
labels, values = zip(*alpha_averages)
snii_values = [val for label, val in alpha_averages if 'SNII' in label]
utsira_values = [val for label, val in alpha_averages if 'Utsira' in label]
x_snii = np.arange(0, len(snii_values))
x_utsira = np.arange(0, len(utsira_values))

# Lag x-ticks for SNII (Winter → Spring → Summer → Autumn)
ticks_snii = [
    "Winter 03:00:00", "Winter 12:00:00", "Winter 21:00:00",
    "Spring 03:00:00", "Spring 12:00:00", "Spring 21:00:00",
    "Summer 03:00:00", "Summer 12:00:00", "Summer 21:00:00",
    "Autumn 03:00:00", "Autumn 12:00:00", "Autumn 21:00:00"
]

plt.figure(figsize=(14, 6))
plt.plot(x_snii, snii_values, marker='o', label='SNII', color='tab:blue')
plt.plot(x_utsira, utsira_values, marker='s', label='Utsira', color='tab:orange')

plt.xticks(x_snii, ticks_snii, rotation=45, fontsize=18)
plt.yticks(fontsize=18)
plt.ylabel('Alpha', fontsize=18)
plt.title('Alpha — SNII vs Utsira', fontsize=18)
plt.legend(fontsize=18)
plt.grid()
plt.tight_layout()
plt.savefig('/Users/hermanellingsrud/Documents/Skole/Masteroppgave/Koder/figures_master/Sesonger/wind_shear_exponent.png', dpi = 300)
plt.show()



#%%
# Base path
base_path = "/Users/hermanellingsrud/Documents/Skole/Masteroppgave/Koder/data/Baseline"

# Mapper og sesonger
folders = ["SNII_3", "SNII_12", "SNII_21", "Utsira_3", "Utsira_12", "Utsira_21"]
seasons = ["autumn", "spring", "summer", "winter"]

# Tom dictionary for dataframes
data_dict = {}

# Loop gjennom mapper og sesonger
for folder in folders:
    folder_path = os.path.join(base_path, folder)
    site, time = folder.split('_')  # f.eks. SNII, 3

    for season in seasons:
        # Filnavn som passer alle
        file_name = f"profile_comparison_{season}_{site}_{time}.csv"
        file_path = os.path.join(folder_path, file_name)

        # Sjekk og les inn
        if os.path.isfile(file_path):
            df = pd.read_csv(file_path)
            key = f"{folder}_{season}"
            data_dict[key] = df



# SNII 3
df_snII_autumn_3 = data_dict["SNII_3_autumn"]
df_snII_spring_3 = data_dict["SNII_3_spring"]
df_snII_summer_3 = data_dict["SNII_3_summer"]
df_snII_winter_3 = data_dict["SNII_3_winter"]

# SNII 12
df_snII_autumn_12 = data_dict["SNII_12_autumn"]
df_snII_spring_12 = data_dict["SNII_12_spring"]
df_snII_summer_12 = data_dict["SNII_12_summer"]
df_snII_winter_12 = data_dict["SNII_12_winter"]

# SNII 21
df_snII_autumn_21 = data_dict["SNII_21_autumn"]
df_snII_spring_21 = data_dict["SNII_21_spring"]
df_snII_summer_21 = data_dict["SNII_21_summer"]
df_snII_winter_21 = data_dict["SNII_21_winter"]

# Utsira 3
df_utsira_autumn_3 = data_dict["Utsira_3_autumn"]
df_utsira_spring_3 = data_dict["Utsira_3_spring"]
df_utsira_summer_3 = data_dict["Utsira_3_summer"]
df_utsira_winter_3 = data_dict["Utsira_3_winter"]

# Utsira 12
df_utsira_autumn_12 = data_dict["Utsira_12_autumn"]
df_utsira_spring_12 = data_dict["Utsira_12_spring"]
df_utsira_summer_12 = data_dict["Utsira_12_summer"]
df_utsira_winter_12 = data_dict["Utsira_12_winter"]

# Utsira 21
df_utsira_autumn_21 = data_dict["Utsira_21_autumn"]
df_utsira_spring_21 = data_dict["Utsira_21_spring"]
df_utsira_summer_21 = data_dict["Utsira_21_summer"]
df_utsira_winter_21 = data_dict["Utsira_21_winter"]

#%%
# Sesong og tid i ønsket rekkefølge
seasons = ['winter', 'spring', 'summer', 'autumn']
times = ['3', '12', '21']

# Lag kronologisk rekkefølge for SNII og Utsira
snii_order = [f"SNII_{time}_{season}" for season in seasons for time in times]
utsira_order = [f"Utsira_{time}_{season}" for season in seasons for time in times]

# Lag tomme lister
snii_means = []
utsira_means = []

# Hent ut gjennomsnittene
for key in snii_order:
    mean_value = data_dict[key]['Diff Power (MW)'].mean()
    snii_means.append(mean_value)

for key in utsira_order:
    mean_value = data_dict[key]['Diff Power (MW)'].mean()
    utsira_means.append(mean_value)

# Lag x-aksen
x_ticks = [
    "Winter 03:00:00", "Winter 12:00:00", "Winter 21:00:00",
    "Spring 03:00:00", "Spring 12:00:00", "Spring 21:00:00",
    "Summer 03:00:00", "Summer 12:00:00", "Summer 21:00:00",
    "Autumn 03:00:00", "Autumn 12:00:00", "Autumn 21:00:00"
]
x = np.arange(len(x_ticks))

# Lag plot
plt.figure(figsize=(14, 6))
plt.plot(x, snii_means, marker='o', label='SNII', color='tab:blue')
plt.plot(x, utsira_means, marker='s', label='Utsira', color='tab:orange')

plt.xticks(x, x_ticks, rotation=45, fontsize=12)
plt.yticks(fontsize=12)
plt.ylabel('Mean Diff Power (MW)', fontsize=14)
plt.title('Chronological Mean Diff Power — SNII vs Utsira', fontsize=16)
plt.legend(fontsize=12)
plt.grid()
plt.tight_layout()
plt.savefig('/Users/hermanellingsrud/Documents/Skole/Masteroppgave/Koder/figures_master/Sesonger/power_diff.png', dpi = 300)
plt.show()

#%%


# Tomme lister
snii_means_percent = []
utsira_means_percent = []

# Filtrer og regn gjennomsnitt for SNII
for key in snii_order:
    df = data_dict[key]
    filtered = df[(df['Hub Speed (m/s)'] >= 3) & (df['Hub Speed (m/s)'] <= 25)]['% Diff Power']
    mean_value = filtered.mean()
    snii_means_percent.append(mean_value)

# Filtrer og regn gjennomsnitt for Utsira
for key in utsira_order:
    df = data_dict[key]
    filtered = df[(df['Hub Speed (m/s)'] >= 3) & (df['Hub Speed (m/s)'] <= 25)]['% Diff Power']
    mean_value = filtered.mean()
    utsira_means_percent.append(mean_value)



# Lag plot
plt.figure(figsize=(14, 6))
plt.plot(x, snii_means_percent, marker='o', label='SNII', color='tab:blue')
plt.plot(x, utsira_means_percent, marker='s', label='Utsira', color='tab:orange')

plt.xticks(x, x_ticks, rotation=45, fontsize=18)
plt.yticks(fontsize=18)
plt.ylabel('Mean % Diff Power', fontsize=18)
plt.title('Mean % Diff Power — SNII vs Utsira', fontsize=20)
plt.legend(fontsize=18)
plt.tight_layout()
plt.grid()
plt.savefig('/Users/hermanellingsrud/Documents/Skole/Masteroppgave/Koder/figures_master/Sesonger/precent_power_diff.png', dpi = 300)
plt.show()

#%%
# Tomme lister
snii_means_flapwise = []
utsira_means_flapwise = []

flapwise_limit = 15  # Sett cutoff for |% Diff Flapwise|

# SNII
for key in snii_order:
    df = data_dict[key]
    filtered = df[
        (df['Hub Speed (m/s)'] >= 3) &
        (df['Hub Speed (m/s)'] <= 25) &
        (df['% Diff Flapwise'].abs() <= flapwise_limit)
    ]['% Diff Flapwise']
    mean_value = filtered.mean()
    snii_means_flapwise.append(mean_value)

# Utsira
for key in utsira_order:
    df = data_dict[key]
    filtered = df[
        (df['Hub Speed (m/s)'] >= 3) &
        (df['Hub Speed (m/s)'] <= 25) &
        (df['% Diff Flapwise'].abs() <= flapwise_limit)
    ]['% Diff Flapwise']
    mean_value = filtered.mean()
    utsira_means_flapwise.append(mean_value)

# Lag plot
plt.figure(figsize=(14, 6))
plt.plot(x, snii_means_flapwise, marker='o', label='SNII', color='tab:blue')
plt.plot(x, utsira_means_flapwise, marker='s', label='Utsira', color='tab:orange')

plt.xticks(x, x_ticks, rotation=45, fontsize=18)
plt.yticks(fontsize=18)
plt.ylabel('Mean % Diff Flapwise', fontsize=18)
plt.title('Mean % Diff Flapwise — SNII vs Utsira', fontsize=20)
plt.legend(fontsize=18)
plt.tight_layout()
plt.grid()
plt.savefig('/Users/hermanellingsrud/Documents/Skole/Masteroppgave/Koder/figures_master/Sesonger/precent_flapwise_diff.png', dpi=300)
plt.show()

# Tomme lister
snii_means_edgewise = []
utsira_means_edgewise = []

edgewise_limit = 25  # Dette setter cutoff for |% Diff Edgewise|

# SNII
for key in snii_order:
    df = data_dict[key]
    filtered = df[
        (df['Hub Speed (m/s)'] >= 3) &
        (df['Hub Speed (m/s)'] <= 25 ) & 
        (df['% Diff Edgewise'].abs() <= edgewise_limit)
    ]['% Diff Edgewise']
    mean_value = filtered.mean()
    snii_means_edgewise.append(mean_value)

# Utsira
for key in utsira_order:
    df = data_dict[key]
    filtered = df[
        (df['Hub Speed (m/s)'] >= 3) &
        (df['Hub Speed (m/s)'] <= 25 ) & 
        (df['% Diff Edgewise'].abs() <= edgewise_limit)
    ]['% Diff Edgewise']
    mean_value = filtered.mean()
    utsira_means_edgewise.append(mean_value)

# Lag plot
plt.figure(figsize=(14, 6))
plt.plot(x, snii_means_edgewise, marker='o', label='SNII', color='tab:blue')
plt.plot(x, utsira_means_edgewise, marker='s', label='Utsira', color='tab:orange')

plt.xticks(x, x_ticks, rotation=45, fontsize=18)
plt.yticks(fontsize=18)
plt.ylabel('Mean % Diff Edgewise', fontsize=18)
plt.title('Mean % Diff Edgewise — SNII vs Utsira', fontsize=20)
plt.legend(fontsize=18)
plt.tight_layout()
plt.grid()
plt.savefig('/Users/hermanellingsrud/Documents/Skole/Masteroppgave/Koder/figures_master/Sesonger/precent_edgewise_diff.png', dpi=300)
plt.show()

#%%
import seaborn as sns


# Først splitte opp order og alpha-lister
snii_orders = snii_order
utsira_orders = utsira_order

snii_alpha_lists = [
    alpha_winter_SNII_03, alpha_winter_SNII_12, alpha_winter_SNII_21,
    alpha_spring_SNII_03, alpha_spring_SNII_12, alpha_spring_SNII_21,
    alpha_summer_SNII_03, alpha_summer_SNII_12, alpha_summer_SNII_21,
    alpha_autumn_SNII_03, alpha_autumn_SNII_12, alpha_autumn_SNII_21
]

utsira_alpha_lists = [
    alpha_winter_Utsira_03, alpha_winter_Utsira_12, alpha_winter_Utsira_21,
    alpha_spring_Utsira_03, alpha_spring_Utsira_12, alpha_spring_Utsira_21,
    alpha_summer_Utsira_03, alpha_summer_Utsira_12, alpha_summer_Utsira_21,
    alpha_autumn_Utsira_03, alpha_autumn_Utsira_12, alpha_autumn_Utsira_21
]

# Funksjon som bygger DataFrame
def build_df(orders, alpha_lists, data_dict):
    alpha_values = []
    power_diff_values = []
    hub_speed_values = []
    
    for key, alpha_list in zip(orders, alpha_lists):
        df = data_dict[key]
        power_diff_series = df['Diff Power (MW)']
        hub_speed_series = df['Hub Speed (m/s)']
        
        if len(alpha_list) == len(power_diff_series) == len(hub_speed_series):
            alpha_values.extend(alpha_list)
            power_diff_values.extend(power_diff_series)
            hub_speed_values.extend(hub_speed_series)
        else:
            print(f"Advarsel: Mismatch i lengde for {key} → alpha: {len(alpha_list)}, power_diff: {len(power_diff_series)}, hub_speed: {len(hub_speed_series)}")
    
    df = pd.DataFrame({
        'Alpha': alpha_values,
        'Diff Power (MW)': power_diff_values,
        'Hub Speed (m/s)': hub_speed_values
    })
    
    # Filter
    df = df[
        (df['Alpha'] >= 0) & (df['Alpha'] <= 0.3) &
        (df['Hub Speed (m/s)'] >= 3) & (df['Hub Speed (m/s)'] <= 25)
    ]
    
    return df


#%%


# Sesonger
seasons = ['winter', 'spring', 'summer', 'autumn']
times = ['3', '12', '21']

# Bestem felles y-grenser (juster etter dataene dine)
y_min, y_max = -0.01, 0.31  # For Alpha
x_min, x_max = 3, 25       # For Hub Speed

# Loop per plassering
for site, orders, alpha_lists in [('SNII', snii_order, snii_alpha_lists), ('Utsira', utsira_order, utsira_alpha_lists)]:
    color = 'blue' if site == 'SNII' else 'orange'
    plt.figure(figsize=(16, 8))
    for i, season in enumerate(seasons):
        season_orders = [o for o in orders if season in o]
        season_alphas = [a for o, a in zip(orders, alpha_lists) if season in o]
        df_season = build_df(season_orders, season_alphas, data_dict)
        corr = df_season['Alpha'].corr(df_season['Hub Speed (m/s)'])
        
        plt.subplot(2, 2, i + 1)
        sns.regplot(x='Hub Speed (m/s)', y='Alpha', data=df_season, color=color, line_kws={'color': 'gray'})
        plt.title(f'{season.capitalize()} (corr={corr:.2f})',fontsize=20)
        plt.ylim(y_min, y_max)  # Husk å definere y_min og y_max i koden før!
        plt.xlim(2.7, 25)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.xlabel('Hub Speed (m/s)',fontsize= 18)
        plt.ylabel('Alpha',fontsize = 18)
        plt.grid()
        

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f'/Users/hermanellingsrud/Documents/Skole/Masteroppgave/Koder/figures_master/Sesonger/{site}_alfa_hub_speed.png', dpi = 300)
    plt.show()

    
#%%

# Funksjon som bygger DataFrame
def build_df(orders, alpha_lists, data_dict):
    alpha_values, power_diff_values, hub_speed_values = [], [], []
    for key, alpha_list in zip(orders, alpha_lists):
        df = data_dict[key]
        power_diff_series = df['Diff Power (MW)']
        hub_speed_series = df['Hub Speed (m/s)']
        if len(alpha_list) == len(power_diff_series) == len(hub_speed_series):
            alpha_values.extend(alpha_list)
            power_diff_values.extend(power_diff_series)
            hub_speed_values.extend(hub_speed_series)
        else:
            print(f"Advarsel: Mismatch i {key} → alpha: {len(alpha_list)}, power_diff: {len(power_diff_series)}, hub_speed: {len(hub_speed_series)}")
    df = pd.DataFrame({
        'Alpha': alpha_values,
        'Diff Power (MW)': power_diff_values,
        'Hub Speed (m/s)': hub_speed_values
    })
    df = df[(df['Alpha'] >= 0) & (df['Alpha'] <= 0.3) & (df['Hub Speed (m/s)'] >= 3) & (df['Hub Speed (m/s)'] <= 25)]
    return df

# Loop per plassering
for site, orders, alpha_lists in [('SNII', snii_order, snii_alpha_lists), ('Utsira', utsira_order, utsira_alpha_lists)]:
    color = 'blue' if site == 'SNII' else 'orange'
    plt.figure(figsize=(16, 8))
    for i, season in enumerate(seasons):
        season_orders = [o for o in orders if season in o]
        season_alphas = [a for o, a in zip(orders, alpha_lists) if season in o]
        df_season = build_df(season_orders, season_alphas, data_dict)
        corr = df_season['Alpha'].corr(df_season['Diff Power (MW)'])
        
        plt.subplot(2, 2, i + 1)
        sns.regplot(x='Alpha', y='Diff Power (MW)',data=df_season, color=color, line_kws={'color': 'gray'})
        plt.title(f'{season.capitalize()} (corr={corr:.2f})', fontsize = 18)
        plt.xlim(-0.01, 0.3)
        plt.ylim(-0.89, 0.61)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.ylabel('Diff Power (MW)',fontsize= 16)
        plt.xlabel('Alpha',fontsize= 16)
        plt.grid()
    plt.suptitle(f'{site} — Alpha vs Diff Power (MW)', fontsize=20)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f'/Users/hermanellingsrud/Documents/Skole/Masteroppgave/Koder/figures_master/Sesonger/{site}_alfa_diff_power.png', dpi = 300)
    plt.show()




#%%
# Funksjon som bygger DataFrame for % Diff Power
def build_df_percent_power(orders, alpha_lists, data_dict):
    alpha_values, percent_diff_values, hub_speed_values = [], [], []
    for key, alpha_list in zip(orders, alpha_lists):
        df = data_dict[key]
        percent_diff_series = df['% Diff Power']
        hub_speed_series = df['Hub Speed (m/s)']
        if len(alpha_list) == len(percent_diff_series) == len(hub_speed_series):
            alpha_values.extend(alpha_list)
            percent_diff_values.extend(percent_diff_series)
            hub_speed_values.extend(hub_speed_series)
        else:
            print(f"Advarsel: Mismatch i {key} → alpha: {len(alpha_list)}, %diff_power: {len(percent_diff_series)}, hub_speed: {len(hub_speed_series)}")
    df = pd.DataFrame({
        'Alpha': alpha_values,
        '% Diff Power': percent_diff_values,
        'Hub Speed (m/s)': hub_speed_values
    })
    df = df[(df['Alpha'] >= 0) & (df['Alpha'] <= 0.3) &
            (df['Hub Speed (m/s)'] >= 3) & (df['Hub Speed (m/s)'] <= 25) &
            (df['% Diff Power'] >= -10) & (df['% Diff Power'] <= 10)]
    return df

for site, orders, alpha_lists in [('SNII', snii_order, snii_alpha_lists), ('Utsira', utsira_order, utsira_alpha_lists)]:
    color = 'blue' if site == 'SNII' else 'orange'
    plt.figure(figsize=(16, 8))
    for i, season in enumerate(seasons):
        season_orders = [o for o in orders if season in o]
        season_alphas = [a for o, a in zip(orders, alpha_lists) if season in o]
        df_season = build_df_percent_power(season_orders, season_alphas, data_dict)
        corr = df_season['Alpha'].corr(df_season['% Diff Power'])
        
        plt.subplot(2, 2, i + 1)
        sns.regplot(x='Alpha', y='% Diff Power', data=df_season, color=color, line_kws={'color': 'gray'})
        plt.title(f'{season.capitalize()} (corr={corr:.2f})', fontsize=20)
        plt.xlim(-0.01, 0.3)
        plt.ylim(-10, 10)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.ylabel('% Diff Power', fontsize=18)
        plt.xlabel('Alpha', fontsize=18)
        plt.grid()
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f'/Users/hermanellingsrud/Documents/Skole/Masteroppgave/Koder/figures_master/Sesonger/{site}_alfa_diff_power_percent.png', dpi = 300)
    plt.show()


for site, orders, alpha_lists in [('SNII', snii_order, snii_alpha_lists), ('Utsira', utsira_order, utsira_alpha_lists)]:
    color = 'blue' if site == 'SNII' else 'orange'
    plt.figure(figsize=(16, 8))
    for i, season in enumerate(seasons):
        season_orders = [o for o in orders if season in o]
        season_alphas = [a for o, a in zip(orders, alpha_lists) if season in o]
        df_season = build_df_percent_power(season_orders, season_alphas, data_dict)
        corr = df_season['Hub Speed (m/s)'].corr(df_season['% Diff Power'])
        
        plt.subplot(2, 2, i + 1)
        sns.regplot(x='Hub Speed (m/s)', y='% Diff Power', data=df_season, color=color, line_kws={'color': 'gray'})
        plt.title(f'{season.capitalize()} (corr={corr:.2f})', fontsize=20)
        plt.xlim(2.9, 25)
        plt.ylim(-10, 10)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.ylabel('% Diff Power', fontsize=18)
        plt.xlabel('Hub Speed (m/s)', fontsize=18)
        plt.grid()
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f'/Users/hermanellingsrud/Documents/Skole/Masteroppgave/Koder/figures_master/Sesonger/{site}_hub_speed_diff_power_percent.png', dpi = 300)
    plt.show()

#%%
# Funksjon som bygger DataFrame for % Diff Flapwise mot Alpha
def build_df_percent_flapwise(orders, alpha_lists, data_dict):
    alpha_values, percent_flapwise_values, hub_speed_values = [], [], []
    for key, alpha_list in zip(orders, alpha_lists):
        df = data_dict[key]
        percent_flapwise_series = df['% Diff Flapwise']
        hub_speed_series = df['Hub Speed (m/s)']
        if len(alpha_list) == len(percent_flapwise_series) == len(hub_speed_series):
            alpha_values.extend(alpha_list)
            percent_flapwise_values.extend(percent_flapwise_series)
            hub_speed_values.extend(hub_speed_series)
        else:
            print(f"Advarsel: Mismatch i {key} → alpha: {len(alpha_list)}, %diff_flapwise: {len(percent_flapwise_series)}, hub_speed: {len(hub_speed_series)}")
    df = pd.DataFrame({
        'Alpha': alpha_values,
        '% Diff Flapwise': percent_flapwise_values,
        'Hub Speed (m/s)': hub_speed_values
    })
    # Filter: Alpha, Hub Speed og % Diff Flapwise
    df = df[(df['Alpha'] >= 0) & (df['Alpha'] <= 0.3) &
            (df['Hub Speed (m/s)'] >= 3) & (df['Hub Speed (m/s)'] <= 25) &
            (df['% Diff Flapwise'] >= -10) & (df['% Diff Flapwise'] <= 10)]
    return df

# Loop per plassering
for site, orders, alpha_lists in [('SNII', snii_order, snii_alpha_lists), ('Utsira', utsira_order, utsira_alpha_lists)]:
    color = 'blue' if site == 'SNII' else 'orange'
    plt.figure(figsize=(16, 8))
    for i, season in enumerate(seasons):
        season_orders = [o for o in orders if season in o]
        season_alphas = [a for o, a in zip(orders, alpha_lists) if season in o]
        df_season = build_df_percent_flapwise(season_orders, season_alphas, data_dict)
        corr = df_season['Alpha'].corr(df_season['% Diff Flapwise'])
        
        plt.subplot(2, 2, i + 1)
        sns.regplot(x='Alpha', y='% Diff Flapwise', data=df_season, color=color, line_kws={'color': 'gray'})
        plt.title(f'{season.capitalize()} (corr={corr:.2f})', fontsize=20)
        plt.xlim(-0.01, 0.3)
        plt.ylim(-10, 10)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.ylabel('% Diff Flapwise', fontsize=18)
        plt.xlabel('Alpha', fontsize=18)
        plt.grid()
   # plt.suptitle(f'{site} — Alpha vs % Diff Flapwise', fontsize=20)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f'/Users/hermanellingsrud/Documents/Skole/Masteroppgave/Koder/figures_master/Sesonger/{site}_alfa_diff_flap_percent.png', dpi = 300)
    plt.show()




#%%
for site, orders, alpha_lists in [('SNII', snii_order, snii_alpha_lists), ('Utsira', utsira_order, utsira_alpha_lists)]:
    color = 'blue' if site == 'SNII' else 'orange'
    plt.figure(figsize=(16, 8))
    for i, season in enumerate(seasons):
        season_orders = [o for o in orders if season in o]
        season_alphas = [a for o, a in zip(orders, alpha_lists) if season in o]
        df_season = build_df_percent_flapwise(season_orders, season_alphas, data_dict)
        corr = df_season['Hub Speed (m/s)'].corr(df_season['% Diff Flapwise'])
        
        plt.subplot(2, 2, i + 1)
        sns.regplot(x='Hub Speed (m/s)', y='% Diff Flapwise', data=df_season, color=color, line_kws={'color': 'gray'})
        plt.title(f'{season.capitalize()} (corr={corr:.2f})', fontsize=20)
        plt.xlim(2.9, 25)
        plt.ylim(-10, 10)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.ylabel('% Diff Flapwise', fontsize=18)
        plt.xlabel('Hub Speed (m/s)', fontsize=18)
        plt.grid()
    #plt.suptitle(f'{site} — Hub Speed vs % Diff Flapwise', fontsize=20)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f'/Users/hermanellingsrud/Documents/Skole/Masteroppgave/Koder/figures_master/Sesonger/{site}_hub_speed_diff_flap_percent.png', dpi = 300)
    plt.show()

#%%
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# Eksempel for én site
for site, orders, alpha_lists in [('SNII', snii_order, snii_alpha_lists), ('Utsira', utsira_order, utsira_alpha_lists)]:
    df_all = build_df_percent_flapwise(orders, alpha_lists, data_dict)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    x = df_all['Alpha']
    y = df_all['Hub Speed (m/s)']
    z = df_all['% Diff Flapwise']

    # Scatter-plot i 3D
    sc = ax.scatter(x, y, z, c=z, cmap=cm.seismic, marker='o')
    
    ax.set_xlabel('Alpha', fontsize=14)
    ax.set_ylabel('Hub Speed (m/s)', fontsize=14)
    ax.set_zlabel('% Diff Flapwise', fontsize=14)
    ax.set_title(f'{site} — % Diff Flapwise vs Alpha and Hub Speed', fontsize=16)

    fig.colorbar(sc, ax=ax, label='% Diff Flapwise')
    plt.tight_layout()
    plt.show()



#%%
# Funksjon som bygger DataFrame for % Diff Edgewise mot Alpha
def build_df_percent_edgewise_alpha(orders, alpha_lists, data_dict):
    alpha_values, percent_edgewise_values, hub_speed_values = [], [], []
    for key, alpha_list in zip(orders, alpha_lists):
        df = data_dict[key]
        percent_edgewise_series = df['% Diff Edgewise']
        hub_speed_series = df['Hub Speed (m/s)']
        if len(alpha_list) == len(percent_edgewise_series) == len(hub_speed_series):
            alpha_values.extend(alpha_list)
            percent_edgewise_values.extend(percent_edgewise_series)
            hub_speed_values.extend(hub_speed_series)
        else:
            print(f"Advarsel: Mismatch i {key} → alpha: {len(alpha_list)}, %diff_edgewise: {len(percent_edgewise_series)}, hub_speed: {len(hub_speed_series)}")
    df = pd.DataFrame({
        'Alpha': alpha_values,
        '% Diff Edgewise': percent_edgewise_values,
        'Hub Speed (m/s)': hub_speed_values
    })
    # Filter: Alpha, Hub Speed og % Diff Edgewise
    df = df[(df['Alpha'] >= 0) & (df['Alpha'] <= 0.3) &
            (df['Hub Speed (m/s)'] >= 3) & (df['Hub Speed (m/s)'] <= 25) &
            (df['% Diff Edgewise'] >= -20) & (df['% Diff Edgewise'] <= 20)]
    return df

for site, orders, alpha_lists in [('SNII', snii_order, snii_alpha_lists), ('Utsira', utsira_order, utsira_alpha_lists)]:
    color = 'blue' if site == 'SNII' else 'orange'
    plt.figure(figsize=(16, 8))
    for i, season in enumerate(seasons):
        season_orders = [o for o in orders if season in o]
        season_alphas = [a for o, a in zip(orders, alpha_lists) if season in o]
        df_season = build_df_percent_edgewise_alpha(season_orders, season_alphas, data_dict)
        corr = df_season['Alpha'].corr(df_season['% Diff Edgewise'])
        
        plt.subplot(2, 2, i + 1)
        sns.regplot(x='Alpha', y='% Diff Edgewise', data=df_season, color=color, line_kws={'color': 'gray'})
        plt.title(f'{season.capitalize()} (corr={corr:.2f})', fontsize=20)
        plt.xlim(-0.01, 0.3)
        plt.ylim(-20, 20)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.ylabel('% Diff Edgewise', fontsize=18)
        plt.xlabel('Alpha', fontsize=18)
        plt.grid()
   # plt.suptitle(f'{site} — Alpha vs % Diff Edgewise', fontsize=20)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f'/Users/hermanellingsrud/Documents/Skole/Masteroppgave/Koder/figures_master/Sesonger/{site}_alfa_diff_edge_percent.png', dpi = 300)
    plt.show()



#%%
def build_df_percent_edgewise(orders, alpha_lists, data_dict):
    hub_speed_values = []
    percent_diff_edgewise_values = []
    alpha_values = []  # Vi tar fortsatt med alpha for filtrering
    
    for key, alpha_list in zip(orders, alpha_lists):
        df = data_dict[key]
        percent_diff_series = df['% Diff Edgewise']
        hub_speed_series = df['Hub Speed (m/s)']
        
        if len(alpha_list) == len(percent_diff_series) == len(hub_speed_series):
            alpha_values.extend(alpha_list)
            percent_diff_edgewise_values.extend(percent_diff_series)
            hub_speed_values.extend(hub_speed_series)

        
    df = pd.DataFrame({
        'Alpha': alpha_values,
        '% Diff Edgewise': percent_diff_edgewise_values,
        'Hub Speed (m/s)': hub_speed_values
    })
    
    # Filter: Outliers
    df = df[
        (df['Alpha'] >= 0) & (df['Alpha'] <= 0.3) &
        (df['Hub Speed (m/s)'] >= 3) & (df['Hub Speed (m/s)'] <= 25) &
        (df['% Diff Edgewise'] >= -20) & (df['% Diff Edgewise'] <= 20)
    ]
    
    return df



for site, orders, alpha_lists in [('SNII', snii_order, snii_alpha_lists), ('Utsira', utsira_order, utsira_alpha_lists)]:
    color = 'blue' if site == 'SNII' else 'orange'
    plt.figure(figsize=(16, 8))
    for i, season in enumerate(seasons):
        season_orders = [o for o in orders if season in o]
        season_alphas = [a for o, a in zip(orders, alpha_lists) if season in o]
        df_season = build_df_percent_edgewise(season_orders, season_alphas, data_dict)
        corr = df_season['Hub Speed (m/s)'].corr(df_season['% Diff Edgewise'])
        
        plt.subplot(2, 2, i + 1)
        sns.regplot(x='Hub Speed (m/s)', y='% Diff Edgewise', data=df_season, color=color, line_kws={'color': 'gray'})
        plt.title(f'{season.capitalize()} (corr={corr:.2f})', fontsize=20)
        plt.xlim(2.9, 25)
        plt.ylim(-20, 20)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.ylabel('% Diff Edgewise', fontsize=18)
        plt.xlabel('Hub Speed (m/s)', fontsize=18)
        plt.grid()
    #plt.suptitle(f'{site} — Hub Speed vs % Diff Edgewise', fontsize=20)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f'/Users/hermanellingsrud/Documents/Skole/Masteroppgave/Koder/figures_master/Sesonger/{site}_hub_speed_diff_edge_percent.png', dpi = 300)
    plt.show()



