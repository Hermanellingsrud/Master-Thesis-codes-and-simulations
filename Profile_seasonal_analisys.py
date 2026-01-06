#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  1 12:54:59 2025

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

# For å se de første radene fra sesongdataene, for eksempel vinterdata fra SNII kl 03:00:
print("Winter SNII 03:00 profiles:")
print(winter_SNII_03.head())

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

def calculate_alpha(h_l, h_h, V_l, V_h):
    """Beregn vindskjæringseksponent (alpha) mellom to høyder."""
    alpha = np.log(V_h / V_l) / np.log(h_h / h_l)
    return alpha

def get_alpha_for_profiles(profiles):
    """Beregn alfa for alle profiler ved hjelp av to høyder (10m og 250m)."""
    h_l = 10  # Lavere høyde (10m)
    h_h = 250  # Høyere høyde (250m)

    # Hent vindhastigheter for de valgte høydene
    V_l = profiles['wind_speed_10m'].values  # Bruk .values for å få numpy array
    V_h = profiles['wind_speed_250m'].values  # Bruk .values for å få numpy array
    
    # Beregn alfa for hver profil
    alpha_values = [calculate_alpha(h_l, h_h, V_l[i], V_h[i]) for i in range(len(profiles))]
    
    return alpha_values



# Beregn alfa for alle sesonger og tidspunkter
alpha_winter_SNII_03 = get_alpha_for_profiles(winter_SNII_03)
alpha_spring_SNII_03 = get_alpha_for_profiles(spring_SNII_03)
alpha_summer_SNII_03 = get_alpha_for_profiles(summer_SNII_03)
alpha_autumn_SNII_03 = get_alpha_for_profiles(autumn_SNII_03)

alpha_winter_Utsira_03 = get_alpha_for_profiles(winter_Utsira_03)
alpha_spring_Utsira_03 = get_alpha_for_profiles(spring_Utsira_03)
alpha_summer_Utsira_03 = get_alpha_for_profiles(summer_Utsira_03)
alpha_autumn_Utsira_03 = get_alpha_for_profiles(autumn_Utsira_03)

# Beregn for kl 12:00
alpha_winter_SNII_12 = get_alpha_for_profiles(winter_SNII_12)
alpha_spring_SNII_12 = get_alpha_for_profiles(spring_SNII_12)
alpha_summer_SNII_12 = get_alpha_for_profiles(summer_SNII_12)
alpha_autumn_SNII_12 = get_alpha_for_profiles(autumn_SNII_12)

alpha_winter_Utsira_12 = get_alpha_for_profiles(winter_Utsira_12)
alpha_spring_Utsira_12 = get_alpha_for_profiles(spring_Utsira_12)
alpha_summer_Utsira_12 = get_alpha_for_profiles(summer_Utsira_12)
alpha_autumn_Utsira_12 = get_alpha_for_profiles(autumn_Utsira_12)

# Beregn for kl 21:00
alpha_winter_SNII_21 = get_alpha_for_profiles(winter_SNII_21)
alpha_spring_SNII_21 = get_alpha_for_profiles(spring_SNII_21)
alpha_summer_SNII_21 = get_alpha_for_profiles(summer_SNII_21)
alpha_autumn_SNII_21 = get_alpha_for_profiles(autumn_SNII_21)

alpha_winter_Utsira_21 = get_alpha_for_profiles(winter_Utsira_21)
alpha_spring_Utsira_21 = get_alpha_for_profiles(spring_Utsira_21)
alpha_summer_Utsira_21 = get_alpha_for_profiles(summer_Utsira_21)
alpha_autumn_Utsira_21 = get_alpha_for_profiles(autumn_Utsira_21)

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
    ('Spring SNII 03:00', average_alpha_spring_SNII_03),
    ('Summer SNII 03:00', average_alpha_summer_SNII_03),
    ('Autumn SNII 03:00', average_alpha_autumn_SNII_03),

    ('Winter Utsira 03:00', average_alpha_winter_Utsira_03),
    ('Spring Utsira 03:00', average_alpha_spring_Utsira_03),
    ('Summer Utsira 03:00', average_alpha_summer_Utsira_03),
    ('Autumn Utsira 03:00', average_alpha_autumn_Utsira_03),

    ('Winter SNII 12:00', average_alpha_winter_SNII_12),
    ('Spring SNII 12:00', average_alpha_spring_SNII_12),
    ('Summer SNII 12:00', average_alpha_summer_SNII_12),
    ('Autumn SNII 12:00', average_alpha_autumn_SNII_12),

    ('Winter Utsira 12:00', average_alpha_winter_Utsira_12),
    ('Spring Utsira 12:00', average_alpha_spring_Utsira_12),
    ('Summer Utsira 12:00', average_alpha_summer_Utsira_12),
    ('Autumn Utsira 12:00', average_alpha_autumn_Utsira_12),

    ('Winter SNII 21:00', average_alpha_winter_SNII_21),
    ('Spring SNII 21:00', average_alpha_spring_SNII_21),
    ('Summer SNII 21:00', average_alpha_summer_SNII_21),
    ('Autumn SNII 21:00', average_alpha_autumn_SNII_21),

    ('Winter Utsira 21:00', average_alpha_winter_Utsira_21),
    ('Spring Utsira 21:00', average_alpha_spring_Utsira_21),
    ('Summer Utsira 21:00', average_alpha_summer_Utsira_21),
    ('Autumn Utsira 21:00', average_alpha_autumn_Utsira_21),
]

# Sorter listen basert på gjennomsnittet (fra høyeste til laveste)
sorted_alpha_averages = sorted(alpha_averages, key=lambda x: x[1], reverse=True)

# Print ut de sorterte gjennomsnittene
print("Sorted alpha averages:")
for label, avg in sorted_alpha_averages:
    print(f"{label}: {avg:.4f}")



#%% airfoil properties
# Define the file path
file_path_airfoil_properties = '/Users/hermanellingsrud/Documents/Skole/Masteroppgave/Koder/data/IEA_15MW_AIRFOILS/IEA-15-240-RWT_AeroDyn15_blade.dat'

# Initialize lists to store extracted data
blade_span = []
blade_curve_ac = []
blade_sweep_ac = []
blade_curve_angle = []
blade_twist = []
blade_chord = []

# Open the file and read the data
with open(file_path_airfoil_properties, 'r') as file:
    # Skip the header lines
    for _ in range(6):  # Adjust the number based on how many header lines to skip
        file.readline()
    
    # Read and extract data
    for line_number, line in enumerate(file, start=7):  # Start at line 7 (after skipping headers)
        # Debug: print the current line being processed
        #print(f"Processing line {line_number}: {line.strip()}")
        
        # Split the line into components
        values = line.split()
        if len(values) >= 6:  # Ensure there are enough values in the line
            # Append the values to the corresponding lists
            try:
                blade_span.append(float(values[0]))
                blade_curve_ac.append(float(values[1]))
                blade_sweep_ac.append(float(values[2]))
                blade_curve_angle.append(float(values[3]))
                blade_twist.append(float(values[4]))
                blade_chord.append(float(values[5]))
            except ValueError as e:
                print(f"Error converting values on line {line_number}: {e}")
        else:
            print(f"Insufficient values on line {line_number}: {values}")


#%% rotor preformance

preformance_path = '/Users/hermanellingsrud/Documents/Skole/Masteroppgave/Koder/data/IEA_15MW_AIRFOILS/Alle_verdier/IEA-15-240-RWT_tabular/Rotor Performance-Table 1.csv'

preformance_data = pd.read_csv(preformance_path, sep=';', decimal=',')

wind_speeds_preformance = preformance_data['Wind [m/s]']
pitch_angles = preformance_data['Pitch [deg]']
rotor_speeds = preformance_data['Rotor Speed [rpm]']
ct_values = preformance_data['Thrust Coefficient [-]']
cp_values = preformance_data['Aero Power Coefficient [-]']
power = preformance_data['Power [MW]']
thrust = preformance_data['Thrust [MN]']
torque = preformance_data['Torque [MNm]']

# Function to calculate axial induction factor from Ct
def calculate_axial_induction_factor(Ct):
    return (1 - np.sqrt(1 - Ct)) / 2

# Calculate axial induction factors for each Ct value
a_values = calculate_axial_induction_factor(ct_values)


# Function to find or interpolate `a` based on wind speed
def get_axial_induction_factor(wind_speed, wind_speeds_performance, ct_values):
    """
    Interpolates Ct based on wind speed and calculates the axial induction factor `a`.
    
    Args:
        wind_speed (float): The current wind speed at the section.
        wind_speeds_performance (array): Array of wind speeds for which Ct values are provided.
        ct_values (array): Array of Ct values corresponding to the wind speeds.

    Returns:
        float: Calculated axial induction factor `a` for the given wind speed.
    """
    # Interpolate `Ct` based on the actual wind speed at the section
    ct_interpolated = np.interp(wind_speed, wind_speeds_performance, ct_values)
    # Calculate axial induction factor `a` from the interpolated `Ct`
    return (1 - np.sqrt(1 - ct_interpolated)) / 2


pitch_angles[0] = pitch_angles[0] +0.7
pitch_angles[1] = pitch_angles[1] +0.9
pitch_angles[2] = pitch_angles[2] +0.9
pitch_angles[3] = pitch_angles[3] +0.9
pitch_angles[4] = pitch_angles[4] +1
pitch_angles[5] = pitch_angles[5] +1
pitch_angles[6] = pitch_angles[6] +1.4
pitch_angles[7] = pitch_angles[7] +1.6
pitch_angles[8] = pitch_angles[8] +1.6
pitch_angles[9] = pitch_angles[9] +1.6
pitch_angles[10] = pitch_angles[10] +1.9

pitch_angles[28] = pitch_angles[28] +1.69
pitch_angles[29] = pitch_angles[29] +1.63
pitch_angles[30] = pitch_angles[30] +0.84
pitch_angles[31] = pitch_angles[31] +0.7
pitch_angles[32] = pitch_angles[32] +0.68
pitch_angles[33] = pitch_angles[33] +0.63
pitch_angles[34] = pitch_angles[34] +0.63
pitch_angles[35] = pitch_angles[35] +0.67
pitch_angles[36] = pitch_angles[36] +0.66
pitch_angles[37] = pitch_angles[37] +0.7
pitch_angles[38] = pitch_angles[38] +0.75
pitch_angles[39] = pitch_angles[39] +0.8
pitch_angles[40] = pitch_angles[40] +0.87
pitch_angles[41] = pitch_angles[41] +0.88
pitch_angles[42] = pitch_angles[42] +1
pitch_angles[43] = pitch_angles[43] +1# 20m/s
pitch_angles[44] = pitch_angles[44] +1.1
pitch_angles[45] = pitch_angles[45] +1.15# 21
pitch_angles[46] = pitch_angles[46] +1.22
pitch_angles[47] = pitch_angles[47] +1.28
pitch_angles[48] = pitch_angles[48] +1.38
pitch_angles[49] = pitch_angles[49] +1.43

#%%
# Filsti til ElastoDyn-bladdata
file_path_blade_data = "/Users/hermanellingsrud/Documents/Skole/Masteroppgave/Koder/data/IEA_15MW_AIRFOILS/IEA-15-240-RWT_ElastoDyn_blade.dat"

# Total bladmasse (kg)
total_blade_mass = 65250  
blade_length = 120  # meter

# Les inn filen og finn tabellen
with open(file_path_blade_data, "r") as file:
    lines = file.readlines()

# Finn start- og slutten av tabellen
start_index = None
for i, line in enumerate(lines):
    if "BlFract" in line and "BMassDen" in line:
        start_index = i + 2  # Tabellen starter to linjer etter dette
        break

end_index = None
for i, line in enumerate(lines[start_index:], start=start_index):
    if "BLADE MODE SHAPES" in line:  # Slutten av tabellen
        end_index = i
        break

# Les tabellen inn som en DataFrame
blade_data = pd.read_csv(
    file_path_blade_data,
    skiprows=start_index,
    nrows=end_index - start_index,
    sep=r"\s+",
    header=None,
    names=["BlFract", "PitchAxis", "StrcTwst", "BMassDen", "FlpStff", "EdgStff"],
    engine="python",
)

# Beregn lengden på hver seksjon
blade_data["Section Length"] = np.diff(blade_data["BlFract"], prepend=0) * blade_length

# Hent tetthet og lengde for hver seksjon
section_densities = blade_data["BMassDen"].values  # kg/m
section_lengths = blade_data["Section Length"].values  # meter

# Beregn massen for hver seksjon
mass_per_section = (section_lengths * section_densities) / np.sum(section_lengths * section_densities) * total_blade_mass

# Legg til massen i DataFrame
blade_data["Section Mass"] = mass_per_section

#%% Air foils

# Define the folder path where your files are stored
folder_path = '/Users/hermanellingsrud/Documents/Skole/Masteroppgave/Koder/data/IEA_15MW_AIRFOILS/'

# Initialize a dictionary to hold all the aerodynamic coefficients for all airfoils
aero_data_all = {}

# Function to extract aerodynamic coefficients from a file
def extract_aero_data(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Initialize the dictionary to store aoa, cl, cd, and cm
    aero_data = {'aoa': [], 'cl': [], 'cd': [], 'cm': []}

    # Find the line with the number of angles
    for line in lines:
        if 'NumAlf' in line:
            num_aoa = int(line.split()[0])  # Get number of angles
            break

    # Read the aerodynamic coefficients
    for line in lines[lines.index(line) + 1:lines.index(line) + 1 + num_aoa]:
        parts = line.split()
        if len(parts) >= 4:
            try:
                # Attempt to convert each part to a float
                aoa_value = float(parts[0])
                cl_value = float(parts[1])
                cd_value = float(parts[2])
                cm_value = float(parts[3])
                
                # Append values to their respective lists
                aero_data['aoa'].append(aoa_value)
                aero_data['cl'].append(cl_value)
                aero_data['cd'].append(cd_value)
                aero_data['cm'].append(cm_value)
            except ValueError:
                # Skip lines that contain invalid float values
                continue
    
    # Convert lists to numpy arrays for easier manipulation
    aero_data['aoa'] = np.array(aero_data['aoa'])
    aero_data['cl'] = np.array(aero_data['cl'])
    aero_data['cd'] = np.array(aero_data['cd'])
    aero_data['cm'] = np.array(aero_data['cm'])

    return aero_data

# Extract data for all 50 airfoils
for i in range(50):
    file_name = f'IEA-15-240-RWT_AeroDyn15_Polar_{i:02d}.dat'
    file_path = os.path.join(folder_path, file_name)
    
    # Extract data and store it in a dictionary
    aero_data_all[f'airfoil_{i:02d}'] = extract_aero_data(file_path)


# Initialize lists to store data for all airfoils
airfoil_key = []
polar_data = []
aoa = []
cl = []
cd = []
cm = []


# Loop through all airfoils and process the data
for i in range(50):
    # Generate airfoil key and store it in the list
    airfoil_key.append(f'airfoil_{i:02d}')
    
    # Retrieve polar data and store it in the list
    polar_data.append(aero_data_all[airfoil_key[i]])

    # Access aoa, cl, cd, cm for each airfoil and store them in corresponding lists
    aoa.append(polar_data[i]['aoa'])
    cl.append(polar_data[i]['cl'])
    cd.append(polar_data[i]['cd'])
    cm.append(polar_data[i]['cm'])


#%% General Constants and Physical Parameters

rho = 1.225  # Lufttetthet i kg/m^3

rotor_diameter = 240  # Rotor diameter (i meter)
rotor_radius = rotor_diameter / 2  # Rotor radius (halvparten av diameteren, i meter)
hub_height = 150

# Høyder for bladets topp- og bunnposisjon basert på rotorens diameter og navhøyde
H_top = hub_height + rotor_radius  # Høyde på bladspissen øverst (270 m)
H_bottom = hub_height - rotor_radius  # Høyde på bladspissen nederst (30 m)

# Vindhastighet parametre
cut_in_speed = 3.0  # Cut-in vindhastighet (i m/s)
rated_wind = 10.59  # Nominell vindhastighet (i m/s)
cut_out_speed = 25.0  # Cut-out vindhastighet (i m/s)

# Rotorhastighetsparametre
min_rot = 5.0  # Minimum rotasjonshastighet (i RPM)
rated_speed_rot = 7.56  # Nominell rotasjonshastighet (i RPM)


a_prime = 0

# Bladgeometri og posisjon
blade_positions = np.linspace(0, 360, num=360, endpoint=False)

# Høyder til bladposisjoner under rotasjon
blade_heights = hub_height + rotor_radius * np.cos(np.radians(blade_positions))

# Lister for lagring av verdier for hver vindhastighet
wind_speed_hub_list = []
rot_speed_hub_list = []
omega_hub_list = []

# Loop over sesonger og tilhørende vindhastigheter ved hub height
for wind_speed_hub in autumn_SNII_21_hub:
    # Lagre navhastighet
    wind_speed_hub_list.append(wind_speed_hub)


    # Beregn rotorhastighet (RPM), satt til 0 utenfor operasjonsområde
    if wind_speed_hub < cut_in_speed or wind_speed_hub > cut_out_speed:
        rot_speed_hub = 0
    else:
        rot_speed_hub = np.interp(wind_speed_hub, wind_speeds_preformance, rotor_speeds)

    rot_speed_hub_list.append(rot_speed_hub)

    # Konverter RPM til rad/s
    omega_hub = (2 * np.pi * rot_speed_hub) / 60
    omega_hub_list.append(omega_hub)
    
#%%
    
blade_span_total = np.append(blade_span, 120)  # Total bladlengde
num_blade_sections = len(blade_span_total)
blade_heights_sections = np.array(blade_span_total)
heights = [10,20,50,100,250,500,750]

# Dictionary som lagrer vindfelt for hver sesong (SNII)
wind_speed_rotation_all = []

# Loop over hver sesong
for profile in range(len(autumn_SNII_21_profile)):
    # Hent vindprofil fra SNII
    wind_profile = autumn_SNII_21_profile[profile]

    # Interpoler vind langs høyde
    wind_speed_interp_func = lambda z: np.interp(z, heights, wind_profile)

    # Init arrays
    blade_section_rotation = np.zeros((num_blade_sections, len(blade_positions)))  # høyder
    wind_speed_rotation = np.zeros((num_blade_sections, len(blade_positions)))     # vindhastighet

    # Loop over bladseksjoner og rotasjonsposisjoner
    for section in range(num_blade_sections):
        blade_section_rotation[section, :] = hub_height + blade_heights_sections[section] * np.cos(np.radians(blade_positions))
        wind_speed_rotation[section, :] = wind_speed_interp_func(blade_section_rotation[section, :])

    # Legg til i listen
    wind_speed_rotation_all.append(wind_speed_rotation)

# Konverter til np.array hvis du vil
wind_speed_rotation_all = np.array(wind_speed_rotation_all)


#%%
blade_span_total_50 = blade_span_total[:50]
num_blade_sections_50 = len(blade_span_total_50)

# Create dictionaries to store results for each profile
phi_values_all = {}
w_sections_all = {}
rot_speed_sections_all = {}
omega_sections_all = {}
u_sections_all = {}
v_app_sections_all = {}
a_sections_all = {}

# Loop over each profile and its corresponding hub wind speed
for profile_idx, (profile, wind_speed_hub) in enumerate(zip(wind_speed_rotation_all, autumn_SNII_21_hub)):
    # Interpolate rotor speed for this hub wind speed
    rot_speed_hub = np.where(
        (wind_speed_hub < cut_in_speed) | (wind_speed_hub > cut_out_speed),
        0,
        np.interp(wind_speed_hub, wind_speeds_preformance, rotor_speeds)
    )
    omega_hub = (2 * np.pi * rot_speed_hub) / 60  # rad/s

    # Initialize arrays
    phi_values_sections = np.zeros((num_blade_sections_50, len(blade_positions)))
    w_sections = np.zeros((num_blade_sections_50, len(blade_positions)))
    rot_speed_sections = np.zeros((num_blade_sections_50, len(blade_positions)))
    omega_sections = np.zeros((num_blade_sections_50, len(blade_positions)))
    u_sections = np.zeros((num_blade_sections_50, len(blade_positions)))
    v_app_sections = np.zeros((num_blade_sections_50, len(blade_positions)))
    a_sections = np.zeros((num_blade_sections_50, len(blade_positions)))

    # Loop over blade sections and rotation positions
    for section in range(num_blade_sections_50):
        if section < num_blade_sections_50 - 1:
            r_central = (blade_span_total_50[section] + blade_span_total_50[section + 1]) / 2
        else:
            r_central = (rotor_radius + blade_span_total_50[-1]) / 2

        for i in range(len(blade_positions)):
            wind_speed_current = profile[section, i]

            # Calculate axial induction factor
            a_section = get_axial_induction_factor(wind_speed_hub, wind_speeds_preformance, ct_values)
            a_sections[section, i] = a_section

            # Effective wind speed
            u_sections[section, i] = wind_speed_current * (1 - a_section)

            # Rotor speed and omega
            rot_speed_sections[section, i] = rot_speed_hub
            omega_sections[section, i] = omega_hub

            # Tangential velocity
            w_sections[section, i] = omega_sections[section, i] * r_central * (1 + a_prime)

            # Inflow angle
            phi_values_sections[section, i] = np.arctan(u_sections[section, i] / w_sections[section, i])

            # Apparent wind speed
            v_app_sections[section, i] = np.sqrt(u_sections[section, i]**2 + w_sections[section, i]**2)

    # Store results
    phi_values_all[profile_idx] = phi_values_sections
    w_sections_all[profile_idx] = w_sections
    rot_speed_sections_all[profile_idx] = rot_speed_sections
    omega_sections_all[profile_idx] = omega_sections
    u_sections_all[profile_idx] = u_sections
    v_app_sections_all[profile_idx] = v_app_sections
    a_sections_all[profile_idx] = a_sections

#%%
# Pitchverdier for hver sesong (Utsira)
blade_pitch_values_all = {}

for profile_idx, (profile, hub_speed) in enumerate(zip(wind_speed_rotation_all, autumn_SNII_21_hub)):
    # Hent navhastighet fra tidligere beregning
    wind_speed_hub = hub_speed

    # Interpoler pitch-vinkel fra performance-data
    blade_pitch_value = np.interp(wind_speed_hub, wind_speeds_preformance, pitch_angles)

    # Initialiser 2D-array for hele bladet gjennom rotasjon
    blade_pitch_values_sections = np.full((num_blade_sections_50, len(blade_positions)), blade_pitch_value)

    # Lagre i dict
    blade_pitch_values_all[profile_idx] = blade_pitch_values_sections
    
#%%

alpha_values_all = {}

# Loop over each rotation profile
for profile_idx, (profile, hub_speed) in enumerate(zip(wind_speed_rotation_all, autumn_SNII_21_hub)):
    # Retrieve or calculate values for the current rotation profile using profile_idx
    phi_values_sections = phi_values_all[profile_idx]
    blade_pitch_values_sections = blade_pitch_values_all[profile_idx]
    
    # Initialize a 2D array for angle of attack (alpha) for each section during the rotation
    alpha_sections = np.zeros_like(phi_values_sections)

    # Loop through each section to calculate the angle of attack (alpha)
    for section in range(num_blade_sections_50):
        # Convert inflow angle (phi) from radians to degrees for the current section
        phi_deg_sections = np.degrees(phi_values_sections[section, :])

        # Calculate the angle of attack: alpha = phi (deg) - blade pitch (deg) - blade twist (deg)
        alpha_sections[section, :] = phi_deg_sections - blade_pitch_values_sections[section, :] - blade_twist[section]

    # Store the alpha values for this rotation profile in the dictionary
    alpha_values_all[profile_idx] = alpha_sections

#%%
# Dictionary to store Cl and Cd values for each profile
Cl_values_all = {}
Cd_values_all = {}

# Loop over each rotation profile
for profile_idx, (profile, hub_speed) in enumerate(zip(wind_speed_rotation_all, autumn_SNII_21_hub)):
    # Initialize arrays for Cl and Cd for each section for this profile
    Cl_sections = np.zeros((num_blade_sections_50, len(blade_positions)))
    Cd_sections = np.zeros((num_blade_sections_50, len(blade_positions)))
    
    # Get the alpha values for each section and position for this profile
    alpha_sections = alpha_values_all[profile_idx]
    
    # Loop over each blade section
    for section in range(num_blade_sections_50):
        # Retrieve the airfoil data for Cl, Cd, and aoa for this section
        Cl_current = cl[section]  # Lift coefficient values for this section's airfoil
        Cd_current = cd[section]  # Drag coefficient values for this section's airfoil
        aoa_current = aoa[section]  # Angle of attack values for this section's airfoil

        # Loop through each position for the current section
        for i in range(len(blade_positions)):
            # Clip the angle of attack to fit within the range of aoa_current
            alpha_clipped = np.clip(alpha_sections[section, i], aoa_current.min(), aoa_current.max())
            
            # Interpolate Cl and Cd based on the clipped alpha for this section and position
            Cl_sections[section, i] = np.interp(alpha_clipped, aoa_current, Cl_current)
            Cd_sections[section, i] = np.interp(alpha_clipped, aoa_current, Cd_current)

    # Store the detailed Cl and Cd arrays for each section and position for this profile
    Cl_values_all[profile_idx] = Cl_sections
    Cd_values_all[profile_idx] = Cd_sections

#%%

# Dictionaries to store lift and drag forces for each profile
lift_force_all = {}
drag_force_all = {}

# Loop over each rotation profile
for profile_idx, (profile, hub_speed) in enumerate(zip(wind_speed_rotation_all, autumn_SNII_21_hub)):
    # Get the Cl, Cd, and v_app values for the current profile
    Cl_sections = Cl_values_all[profile_idx]
    Cd_sections = Cd_values_all[profile_idx]
    v_app_sections = v_app_sections_all[profile_idx]
    
    # Initialize arrays for lift and drag forces for each section and position
    lift_force_sections = np.zeros_like(Cl_sections)
    drag_force_sections = np.zeros_like(Cd_sections)

    # Loop over each blade section
    for section in range(num_blade_sections_50):
        # Calculate the blade segment length for each section
        if section < num_blade_sections_50 - 1:
            blade_segment = blade_span_total_50[section + 1] - blade_span_total_50[section]
        else:
            blade_segment = rotor_radius - blade_span_total_50[-1]

        # Get the chord length for the current section
        chord_length = blade_chord[section]

        # Reference area A = chord length * blade segment
        area = chord_length * blade_segment

        # Loop through each blade position (0° to 360°)
        for i in range(len(blade_positions)):
            # Calculate lift force: F_lift = 0.5 * rho * v_app² * Cl * A
            lift_force_sections[section, i] = 0.5 * rho * v_app_sections[section, i]**2 * Cl_sections[section, i] * area

            # Calculate drag force: F_drag = 0.5 * rho * v_app² * Cd * A
            drag_force_sections[section, i] = 0.5 * rho * v_app_sections[section, i]**2 * Cd_sections[section, i] * area

    # Store the lift and drag forces for this profile in the dictionaries
    lift_force_all[profile_idx] = lift_force_sections
    drag_force_all[profile_idx] = drag_force_sections

#%%
# Dictionaries to store normal and tangential forces for each profile
P_n_all = {}
P_t_all = {}

# Loop over each rotation profile
for profile_idx, (profile, hub_speed) in enumerate(zip(wind_speed_rotation_all, autumn_SNII_21_hub)):
    # Get lift and drag forces and phi values for the current profile
    lift_force_sections = lift_force_all[profile_idx]
    drag_force_sections = drag_force_all[profile_idx]
    phi_values_sections = phi_values_all[profile_idx]
    
    # Initialize arrays for normal and tangential forces for each section and position
    P_n_sections = np.zeros_like(lift_force_sections)
    P_t_sections = np.zeros_like(lift_force_sections)

    # Loop through each blade section to calculate the normal and tangential forces
    for section in range(num_blade_sections_50):
        # Calculate normal and tangential forces for each blade position
        for i in range(len(blade_positions)):
            # Precompute sin and cos of the inflow angle (phi)
            cos_phi = np.cos(phi_values_sections[section, i])
            sin_phi = np.sin(phi_values_sections[section, i])

            # Normal force: Lift * cos(phi) + Drag * sin(phi)
            P_n_sections[section, i] = (
                lift_force_sections[section, i] * cos_phi + drag_force_sections[section, i] * sin_phi
            )

            # Tangential force: Lift * sin(phi) - Drag * cos(phi)
            P_t_sections[section, i] = (
                lift_force_sections[section, i] * sin_phi - drag_force_sections[section, i] * cos_phi
            )

    # Store the normal and tangential forces for this profile in the dictionaries
    P_n_all[profile_idx] = P_n_sections
    P_t_all[profile_idx] = P_t_sections

#%%
# Dictionary to store tip loss factor F for each profile
tip_loss_factor_all = {}

# Set number of blades
B = 3  # Adjust if the turbine model requires more blades

# Loop over each rotation profile
for profile_idx, (profile, hub_speed) in enumerate(zip(wind_speed_rotation_all, autumn_SNII_21_hub)):
    # Retrieve the inflow angle (phi) values for the current profile
    phi_values_sections = phi_values_all[profile_idx]
    
    # Initialize arrays for f and F for each section and position
    f = np.zeros((num_blade_sections_50, len(blade_positions)))
    F = np.zeros((num_blade_sections_50, len(blade_positions)))

    # Loop through each section
    for section in range(num_blade_sections_50):
        # Calculate r_central for each section (average radius of the section)
        if section < num_blade_sections_50 - 1:
            r_central = (blade_span_total_50[section] + blade_span_total_50[section + 1]) / 2
        else:
            r_central = (rotor_radius + blade_span_total_50[-1]) / 2  # Last section
        
        # Loop through blade positions for each section
        for i in range(len(blade_positions)):
            # Calculate f_tip based on the inflow angle phi and central radius
            f[section, i] = (B / 2) * (((rotor_radius + blade_span_total_50[-1]) / 2) - r_central) / (r_central * np.sin(phi_values_sections[section, i]))

            # Compute F_tip using f_tip
            F[section, i] = (2 / np.pi) * np.arccos(np.exp(-f[section, i]))

    # Store the tip loss factor F for this profile in the dictionary
    tip_loss_factor_all[profile_idx] = F
#%%
# Dictionaries to store thrust and torque for each profile
thrust_all = {}
torque_all = {}

# Number of blades
B = 1  # Adjust if needed

# Loop over each rotation profile
for profile_idx, (profile, hub_speed) in enumerate(zip(wind_speed_rotation_all, autumn_SNII_21_hub)):
    # Retrieve normal and tangential forces, and tip loss factor for the current profile
    P_n_sections = P_n_all[profile_idx]
    P_t_sections = P_t_all[profile_idx]
    F_sections = tip_loss_factor_all[profile_idx]
    
    # Initialize arrays for thrust and torque for each section and position
    thrust_sections = np.zeros_like(P_n_sections)
    torque_sections = np.zeros_like(P_t_sections)

    # Calculate thrust and torque for each section
    for section in range(num_blade_sections_50):
        # Calculate thrust: Thrust = P_n * B * F
        thrust_sections[section, :] = P_n_sections[section, :] * B * F_sections[section, :]

        # Calculate r_central for each section (average radius of the section)
        if section < num_blade_sections_50 - 1:
            r_central = (blade_span_total_50[section] + blade_span_total_50[section + 1]) / 2
        else:
            r_central = (rotor_radius + blade_span_total_50[-1]) / 2  # Last section

        # Calculate torque: Torque = P_t * B * r_central * F
        torque_sections[section, :] = P_t_sections[section, :] * B * r_central * F_sections[section, :]

    # Store the thrust and torque for this profile in the dictionaries
    thrust_all[profile_idx] = thrust_sections
    torque_all[profile_idx] = torque_sections
    
#%%

# Initialize dictionaries to store total thrust and total torque for each profile
total_thrust_all = {}
total_torque_all = {}

# Number of blades
B = 3  # Set to 3 blades for the total calculations

# Loop over each rotation profile
for profile_idx, (profile, hub_speed) in enumerate(zip(wind_speed_rotation_all, autumn_SNII_21_hub)):
    # Retrieve normal and tangential forces, and tip loss factor for the current profile
    P_n_sections = P_n_all[profile_idx]
    P_t_sections = P_t_all[profile_idx]
    F_sections = tip_loss_factor_all[profile_idx]
    
    # Initialize variables for total thrust and total torque for the current profile
    total_thrust = 0
    total_torque = 0

    # Loop through each blade section to calculate thrust and torque
    for section in range(num_blade_sections_50):
        # Sum thrust and torque contributions over all rotational positions for this section
        section_thrust_sum = 0
        section_torque_sum = 0
        
        # Loop over each rotational position (100 points per rotation)
        for i in range(len(blade_positions)):
            # Thrust at this rotational position
            thrust = P_n_sections[section, i] * B * F_sections[section, i]
            section_thrust_sum += thrust
            
            # Calculate r_central for each section (average radius of the section)
            if section < num_blade_sections_50 - 1:
                r_central = (blade_span_total_50[section] + blade_span_total_50[section + 1]) / 2
            else:
                r_central = (rotor_radius + blade_span_total_50[-1]) / 2  # Last section
            
            # Torque at this rotational position
            torque = P_t_sections[section, i] * B * r_central * F_sections[section, i]
            section_torque_sum += torque
        
        # Calculate the average thrust and torque for this section
        section_thrust_avg = section_thrust_sum / len(blade_positions)
        section_torque_avg = section_torque_sum / len(blade_positions)
        
        # Add the section's average thrust and torque to the total
        total_thrust += section_thrust_avg
        total_torque += section_torque_avg

    # Convert total thrust and torque to MN and MN·m
    total_thrust_MN = total_thrust / 1e6
    total_torque_MNm = total_torque / 1e6
    
    # Store the total thrust and torque for this profile in the dictionaries
    total_thrust_all[profile_idx] = total_thrust_MN
    total_torque_all[profile_idx] = total_torque_MNm

#%%
# Dictionary to store total power output (in MW) for each profile
power_output_all = {}

# Set the number of blades
B = 3  # Number of blades

# Loop over each rotation profile
for profile_idx, (profile, hub_speed) in enumerate(zip(wind_speed_rotation_all, autumn_SNII_21_hub)):
    if hub_speed == 0.0:
        # Set power output to 0 for 0 m/s wind speed directly
        power_output_all[profile_idx] = 0
        continue
    elif hub_speed <= 2.5 or hub_speed > 25:
        # Set rotor speed and omega to 0 for low or over cut-out wind speeds
        rot_speed_hub = 0
        omega_hub = 0
    else:
        # Convert rotor speed from RPM to rad/s for this hub wind speed
        rot_speed_hub = np.interp(hub_speed, wind_speeds_preformance, rotor_speeds)
        omega_hub = (2 * np.pi * rot_speed_hub) / 60  # Angular velocity in rad/s

    # Initialize total power variable for this profile
    total_power = 0

    # Loop over each blade
    for blade_offset in [0, 120, 240]:  # Blade positions (0°, 120°, 240°)
        # Loop over each blade section
        for section in range(num_blade_sections_50):
            # Sum torque contributions for all rotational positions of this section
            section_torque_sum = 0

            # Loop over each rotational position
            for i, blade_position in enumerate(blade_positions):
                # Adjust blade position for the current blade
                adjusted_position = (blade_position + blade_offset) % 360

                # Find the nearest rotor index for this adjusted position
                rotor_index = (np.abs(blade_positions - adjusted_position)).argmin()

                # Torque at this rotational position
                r_central = (blade_span_total_50[section] + blade_span_total_50[section + 1]) / 2 if section < num_blade_sections_50 - 1 else (rotor_radius + blade_span_total_50[-1]) / 2
                torque = P_t_all[profile_idx][section, rotor_index] * r_central * tip_loss_factor_all[profile_idx][section, rotor_index]

                # Add to the section's total torque
                section_torque_sum += torque

            # Average torque contribution for this section over the full rotation
            section_torque_avg = section_torque_sum / len(blade_positions)

            # Power contribution from this section using P = Q * omega
            power_section = section_torque_avg * omega_hub

            # Add to the total power for this blade
            total_power += power_section

    # Convert to MW for all blades
    total_power_mw_total = total_power / 1e6

    # Store the total power output for this profile
    power_output_all[profile_idx] = total_power_mw_total
    
#%% Bøymoment
bending_moments_all_profiles = {}

for profile_idx, (profile, hub_speed) in enumerate(zip(wind_speed_rotation_all, autumn_SNII_21_hub)):
    # Hent nødvendige aerodynamiske data
    P_n_sections = P_n_all[profile_idx]
    P_t_sections = P_t_all[profile_idx]  # Tangential forces
    phi_sections = phi_values_all[profile_idx]  # Inflow angle (radians)
    F_sections = tip_loss_factor_all[profile_idx]
    pitch_sections = blade_pitch_values_all[profile_idx]  # Pitch for hver seksjon og posisjon

    # Initialiser array for bøyemomenter
    bending_moments_sections = np.zeros_like(P_n_sections)

    for section in range(num_blade_sections_50):
        # Mid-radius of the section
        if section < num_blade_sections_50 - 1:
            r_central = (blade_span_total_50[section] + blade_span_total_50[section + 1]) / 2
        else:
            r_central = (rotor_radius + blade_span_total_50[-1]) / 2

        # Få twist for denne seksjonen (i radianer)
        beta_twist = np.radians(blade_twist[section])

        for i in range(len(blade_positions)):
            phi = phi_sections[section, i]  # Inflow angle in radians
            theta_pitch = np.radians(pitch_sections[section, i])  # Pitch per section and position

            F_n = P_n_sections[section, i]  # Normal force
            F_t = P_t_sections[section, i]  # Tangential force
            F = F_sections[section, i]  # Tip loss factor

            # Beregn lokal flapwise kraft (justert for twist og pitch)
            phi_eff = phi - (theta_pitch + beta_twist)  # Effektiv innstrømningsvinkel for flapwise-beregning
            flapwise_local = F_n * np.cos(phi_eff) - F_t * np.sin(phi_eff)

            # Beregn bøyemoment for denne seksjonen ved denne posisjonen
            bending_moments_sections[section, i] = flapwise_local * r_central * F  # Moment = Kraft * Radius * Tip loss factor

    # Lagre bøyemomentene for denne profilen
    bending_moments_all_profiles[profile_idx] = bending_moments_sections

# Konstant for gravitasjonsakselerasjon
g = 9.81  # m/s²

# Tiltvinkel
tilt_angle = 6  # grader
cone_angle = 4  # Konevinkel for rotasjon

# Initialisere dictionary for gravitasjonsmoment
gravitational_moments_all = {}

# Beregning av gravitasjonsmoment for alle seksjoner og posisjoner
# Antas uavhengig av vindhastighet, men vi bruker profile_idx for konsistens
for profile_idx, (profile, hub_speed) in enumerate(zip(wind_speed_rotation_all, autumn_SNII_21_hub)):
    # Initialiser en 2D-array for seksjoner og rotorposisjoner
    gravitational_moments_sections = np.zeros((num_blade_sections_50, len(blade_positions)))
    
    for section in range(len(blade_data)):
        # Midtradius for seksjonen (gjennomsnittlig radius)
        if section < len(blade_data) - 1:
            r_central = (blade_data["BlFract"].iloc[section] + blade_data["BlFract"].iloc[section + 1]) / 2 * blade_length
        else:
            r_central = blade_length  # Siste seksjon
        
        # Massen for seksjonen
        m_i = blade_data["Section Mass"].iloc[section]
        
        # Gravitasjonsmomentet for alle posisjoner (avhengig av rotorposisjon)
        for i in range(len(blade_positions)):
            gravitational_moment = m_i * g * r_central * np.sin(np.radians(tilt_angle - cone_angle * np.cos(np.radians(i))))
            
            gravitational_moments_sections[section, i] = gravitational_moment

    # Lagre gravitasjonsmomentene for denne profilen
    gravitational_moments_all[profile_idx] = gravitational_moments_sections

# Initialisere dictionary for totalt flapwise bøyemoment
total_flapwise_moments_all_profiles = {}

for profile_idx in bending_moments_all_profiles:
    if profile_idx in gravitational_moments_all:
        # Hent aerodynamisk og gravitasjonsmoment for profilen
        aero_moments = bending_moments_all_profiles[profile_idx]
        grav_moments = gravitational_moments_all[profile_idx]
        
        # Totalt moment som en sum av aerodynamisk og gravitasjonsmoment
        total_moments = aero_moments + grav_moments
        
        # Lagre resultatene for denne profilen
        total_flapwise_moments_all_profiles[profile_idx] = total_moments


# Beregning av kumulative aerodynamiske momenter for profiler
cumulative_bending_moments_all_profiles = {}
for profile_idx in bending_moments_all_profiles:
    aero_moments = bending_moments_all_profiles[profile_idx]
    cumulative_moments_sections = np.zeros_like(aero_moments)

    # Iterer baklengs for å beregne kumulative momenter
    for section in range(num_blade_sections_50 - 1, -1, -1):  # Fra siste til første seksjon
        if section == num_blade_sections_50 - 1:  # Siste seksjon
            cumulative_moments_sections[section] = aero_moments[section]
        else:
            cumulative_moments_sections[section] = (
                aero_moments[section] + cumulative_moments_sections[section + 1]
            )
    
    cumulative_bending_moments_all_profiles[profile_idx] = cumulative_moments_sections

# Beregning av kumulative gravitasjonsmomenter for profiler
cumulative_gravitational_moments_all_profiles = {}
for profile_idx in gravitational_moments_all:
    gravitational_moments = gravitational_moments_all[profile_idx]
    cumulative_grav_moments = np.zeros_like(gravitational_moments)

    for section in range(len(blade_data) - 1, -1, -1):  # Fra siste til første seksjon
        if section == len(blade_data) - 1:  # Siste seksjon
            cumulative_grav_moments[section, :] = gravitational_moments[section, :]
        else:
            cumulative_grav_moments[section, :] = (
                gravitational_moments[section, :] + cumulative_grav_moments[section + 1, :]
            )
    
    cumulative_gravitational_moments_all_profiles[profile_idx] = cumulative_grav_moments

# Kombinere kumulative momenter for total flapwise moment per profil
cumulative_total_moments_all_profiles = {}
for profile_idx in cumulative_bending_moments_all_profiles:
    if profile_idx in cumulative_gravitational_moments_all_profiles:
        aero_cumulative = cumulative_bending_moments_all_profiles[profile_idx]
        grav_cumulative = cumulative_gravitational_moments_all_profiles[profile_idx]
        
        cumulative_total_moments = aero_cumulative + grav_cumulative
        
        cumulative_total_moments_all_profiles[profile_idx] = cumulative_total_moments

#%% edge
# Forskyt rotorposisjonene slik at 0° er toppen av rotasjonen
adjusted_blade_positions = (blade_positions + 270) % 360

# Dictionary for å lagre edgewise-momenter per profil
edgewise_moments_all_profiles = {}

# Loop over profilene
for profile_idx, (profile, hub_speed) in enumerate(zip(wind_speed_rotation_all, autumn_SNII_21_hub)):
    # Hent nødvendige aerodynamiske data
    P_n_sections = P_n_all[profile_idx]
    P_t_sections = P_t_all[profile_idx]
    phi_sections = phi_values_all[profile_idx]
    F_sections = tip_loss_factor_all[profile_idx]
    pitch_sections = blade_pitch_values_all[profile_idx]

    # Initialiser array for edgewise-momenter
    edgewise_moments_sections = np.zeros_like(P_n_sections)

    for section in range(num_blade_sections_50):
        # Beregn midradius for seksjonen
        if section < num_blade_sections_50 - 1:
            r_central = (blade_span_total_50[section] + blade_span_total_50[section + 1]) / 2
        else:
            r_central = (rotor_radius + blade_span_total_50[-1]) / 2

        # Masse for seksjonen
        m_i = blade_data["Section Mass"].iloc[section]
        beta_twist = np.radians(blade_twist[section])

        for i, blade_position in enumerate(adjusted_blade_positions):
            # Hent phi og pitch for denne seksjonen og posisjonen
            phi = phi_sections[section, i]
            theta_pitch = np.radians(pitch_sections[section, i])

            # Aerodynamiske krefter
            F_n = P_n_sections[section, i]
            F_t = P_t_sections[section, i]
            F = F_sections[section, i]

            # Beregn lokal edgewise kraft
            phi_eff = phi - (theta_pitch + beta_twist)
            edgewise_local = F_n * np.sin(phi_eff) + F_t * np.cos(phi_eff)

            # Aerodynamisk edgewise moment
            aerodynamic_moment = edgewise_local * r_central * F

            # Gravitasjonsmoment
            gravitational_moment = m_i * g * r_central * np.cos(np.radians(blade_position))

            # Totalt edgewise moment
            edgewise_moments_sections[section, i] = aerodynamic_moment + gravitational_moment

    # Lagre edgewise-momenter for denne profilen
    edgewise_moments_all_profiles[profile_idx] = edgewise_moments_sections

# Initialiser dictionary for kumulative torque-momenter per profil
cumulative_torque_moments_all_profiles = {}

for profile_idx in P_n_all:
    # Hent nødvendige data
    P_n_sections = P_n_all[profile_idx]
    P_t_sections = P_t_all[profile_idx]
    phi_sections = phi_values_all[profile_idx]
    F_sections = tip_loss_factor_all[profile_idx]
    pitch_sections = blade_pitch_values_all[profile_idx]

    torque_sections = np.zeros_like(P_n_sections)

    for section in range(num_blade_sections_50):
        if section < num_blade_sections_50 - 1:
            r_central = (blade_span_total_50[section] + blade_span_total_50[section + 1]) / 2
        else:
            r_central = (rotor_radius + blade_span_total_50[-1]) / 2

        beta_twist = np.radians(blade_twist[section])

        for i in range(len(blade_positions)):
            phi = phi_sections[section, i]
            theta_pitch = np.radians(pitch_sections[section, i])

            F_n = P_n_sections[section, i]
            F_t = P_t_sections[section, i]
            F = F_sections[section, i]

            phi_eff = phi - (theta_pitch + beta_twist)

            tangential_force = F_t * np.cos(phi_eff) + F_n * np.sin(phi_eff)
            torque_sections[section, i] = tangential_force * r_central * F

    cumulative_torque_moments = np.zeros_like(torque_sections)
    for section in range(num_blade_sections_50 - 1, -1, -1):
        if section == num_blade_sections_50 - 1:
            cumulative_torque_moments[section, :] = torque_sections[section, :]
        else:
            cumulative_torque_moments[section, :] = (
                torque_sections[section, :] + cumulative_torque_moments[section + 1, :]
            )

    cumulative_torque_moments_all_profiles[profile_idx] = cumulative_torque_moments

# Initialiser dictionaries for gravitasjonsmomenter per profil
gravitational_moments_all_profiles = {}
cumulative_gravitational_moments_all_profiles = {}

for profile_idx in P_n_all:
    gravitational_moments_sections = np.zeros((num_blade_sections_50, len(adjusted_blade_positions)))

    for section in range(num_blade_sections_50):
        if section < num_blade_sections_50 - 1:
            r_central = (blade_span_total_50[section] + blade_span_total_50[section + 1]) / 2
        else:
            r_central = (rotor_radius + blade_span_total_50[-1]) / 2

        m_i = blade_data["Section Mass"].iloc[section]

        for i, blade_position in enumerate(adjusted_blade_positions):
            gravitational_moment = m_i * g * r_central * np.cos(np.radians(blade_position))
            gravitational_moments_sections[section, i] = gravitational_moment

    gravitational_moments_all_profiles[profile_idx] = gravitational_moments_sections

    cumulative_grav_moments = np.zeros_like(gravitational_moments_sections)
    for section in range(num_blade_sections_50 - 1, -1, -1):
        if section == num_blade_sections_50 - 1:
            cumulative_grav_moments[section, :] = gravitational_moments_sections[section, :]
        else:
            cumulative_grav_moments[section, :] = (
                gravitational_moments_sections[section, :] + cumulative_grav_moments[section + 1, :]
            )

    cumulative_gravitational_moments_all_profiles[profile_idx] = cumulative_grav_moments

# Initialiser dictionary for totalt kumulativt edgewise-moment per profil
cumulative_total_edgewise_moments_all_profiles = {}

for profile_idx in cumulative_torque_moments_all_profiles:
    if profile_idx in cumulative_gravitational_moments_all_profiles:
        cumulative_torque = cumulative_torque_moments_all_profiles[profile_idx]
        cumulative_gravitational = cumulative_gravitational_moments_all_profiles[profile_idx]
        cumulative_total_edgewise_moments = cumulative_torque + cumulative_gravitational
        cumulative_total_edgewise_moments_all_profiles[profile_idx] = cumulative_total_edgewise_moments



#%%

import pandas as pd
from scipy.interpolate import interp1d

# Filbane til baseline kraftdata
kraft_path = '/Users/hermanellingsrud/Documents/Skole/Masteroppgave/Koder/data/Kraft/power_output.csv'

# Les inn CSV
baseline_power = pd.read_csv(kraft_path)

# Lag interpolasjonsfunksjon for baseline
baseline_interp = interp1d(
    baseline_power['Wind Speed (m/s)'], 
    baseline_power['Power Output (MW)'], 
    kind='linear', 
    fill_value="extrapolate"
)

# Filbaner til baseline-data
flapwise_path = '/Users/hermanellingsrud/Documents/Skole/Masteroppgave/Koder/data/Baseline/baseline_flapwise.csv'
edgewise_path = '/Users/hermanellingsrud/Documents/Skole/Masteroppgave/Koder/data/Baseline/baseline_edgewise.csv'

# Les inn CSV
baseline_flapwise = pd.read_csv(flapwise_path)
baseline_edgewise = pd.read_csv(edgewise_path)

# Lag interpolasjonsfunksjoner
baseline_flapwise_interp = interp1d(
    baseline_flapwise['Wind Speed (m/s)'],
    baseline_flapwise['Average Flapwise Bending Moment (MNm)'],
    kind='linear',
    fill_value="extrapolate"
)

baseline_edgewise_interp = interp1d(
    baseline_edgewise['Wind Speed (m/s)'],
    baseline_edgewise['Average Edgewise Bending Moment (MNm)'],
    kind='linear',
    fill_value="extrapolate"
)
#%%

print("Sammenligning av hver profil mot baseline (kraft og bøymoment):\n")
dates = autumn_SNII_21['time'].values

for profile_idx, (hub_speed, alpha) in enumerate(zip(autumn_SNII_21_hub, alpha_autumn_SNII_21)):
    date = pd.to_datetime(dates[profile_idx]).strftime('%Y-%m-%d %H:%M')
    model_val = power_output_all[profile_idx]
    
    # Kraft baseline
    if hub_speed > 25:
        baseline_power_val = 0.0
    else:
        baseline_power_val = float(baseline_interp(hub_speed))
    diff_power = model_val - baseline_power_val
    pct_diff_power = 100 * diff_power / baseline_power_val if baseline_power_val != 0 else 0

    # Flapwise moment
    avg_flapwise_moment = np.mean(cumulative_total_moments_all_profiles[profile_idx][0, :])
    baseline_flapwise_val = float(baseline_flapwise_interp(hub_speed))
    diff_flapwise = avg_flapwise_moment - baseline_flapwise_val
    pct_diff_flapwise = 100 * diff_flapwise / baseline_flapwise_val if baseline_flapwise_val != 0 else 0

    # Edgewise moment
    avg_edgewise_moment = np.mean(cumulative_total_edgewise_moments_all_profiles[profile_idx][0, :])
    baseline_edgewise_val = float(baseline_edgewise_interp(hub_speed))
    diff_edgewise = avg_edgewise_moment - baseline_edgewise_val
    pct_diff_edgewise = 100 * diff_edgewise / baseline_edgewise_val if baseline_edgewise_val != 0 else 0

    print(
        f"Profil {profile_idx} ({date}):\n"
        f"  Hub-hastighet:                  {hub_speed:.2f} m/s\n"
        f"  Vindskjær (α):                   {alpha:.3f}\n"
        f"  Modell kraft:                    {model_val:.3f} MW\n"
        f"  Baseline kraft:                  {baseline_power_val:.3f} MW\n"
        f"  Differanse kraft:                {diff_power:.3f} MW ({pct_diff_power:+.1f} %)\n"
        f"  Flapwise moment (model):         {avg_flapwise_moment:.2f} Nm\n"
        f"  Flapwise moment (baseline):      {baseline_flapwise_val:.2f} Nm\n"
        f"  Differanse flapwise:             {diff_flapwise:.2f} Nm ({pct_diff_flapwise:+.1f} %)\n"
        f"  Edgewise moment (model):         {avg_edgewise_moment:.2f} Nm\n"
        f"  Edgewise moment (baseline):      {baseline_edgewise_val:.2f} Nm\n"
        f"  Differanse edgewise:             {diff_edgewise:.2f} Nm ({pct_diff_edgewise:+.1f} %)\n"
    )

#%%



# Samle data
data = []
dates = autumn_SNII_21['time'].values

for profile_idx, (hub_speed, alpha) in enumerate(zip(autumn_SNII_21_hub, alpha_autumn_SNII_21)):
    date = pd.to_datetime(dates[profile_idx]).strftime('%Y-%m-%d %H:%M')
    model_val = power_output_all[profile_idx]
    
    if hub_speed > 25:
        baseline_power_val = 0.0
    else:
        baseline_power_val = float(baseline_interp(hub_speed))
    diff_power = model_val - baseline_power_val
    pct_diff_power = 100 * diff_power / baseline_power_val if baseline_power_val != 0 else 0

    avg_flapwise_moment = np.mean(cumulative_total_moments_all_profiles[profile_idx][0, :])
    baseline_flapwise_val = float(baseline_flapwise_interp(hub_speed))
    diff_flapwise = avg_flapwise_moment - baseline_flapwise_val
    pct_diff_flapwise = 100 * diff_flapwise / baseline_flapwise_val if baseline_flapwise_val != 0 else 0

    avg_edgewise_moment = np.mean(cumulative_total_edgewise_moments_all_profiles[profile_idx][0, :])
    baseline_edgewise_val = float(baseline_edgewise_interp(hub_speed))
    diff_edgewise = avg_edgewise_moment - baseline_edgewise_val
    pct_diff_edgewise = 100 * diff_edgewise / baseline_edgewise_val if baseline_edgewise_val != 0 else 0

    data.append({
        'Profile': profile_idx,
        'Date': date,
        'Hub Speed (m/s)': hub_speed,
        'Alpha': alpha,
        'Model Power (MW)': model_val,
        'Baseline Power (MW)': baseline_power_val,
        'Diff Power (MW)': diff_power,
        '% Diff Power': pct_diff_power,
        'Model Flapwise (MNm)': avg_flapwise_moment,
        'Baseline Flapwise (MNm)': baseline_flapwise_val,
        'Diff Flapwise (MNm)': diff_flapwise,
        '% Diff Flapwise': pct_diff_flapwise,
        'Model Edgewise (MNm)': avg_edgewise_moment,
        'Baseline Edgewise (MNm)': baseline_edgewise_val,
        'Diff Edgewise (MNm)': diff_edgewise,
        '% Diff Edgewise': pct_diff_edgewise
    })

# Lag DataFrame
df = pd.DataFrame(data)

# Lagre til CSV med spesifikt navn
output_path = '/Users/hermanellingsrud/Documents/Skole/Masteroppgave/Koder/data/Baseline/profile_comparison_autumn_SNII_21.csv'
df.to_csv(output_path, index=False)
print(f"CSV lagret til: {output_path}")


# Beregn gjennomsnitt
avg_values = df[['Hub Speed (m/s)', 'Alpha', 'Model Power (MW)', 'Model Flapwise (MNm)', 'Model Edgewise (MNm)', 'Diff Power (MW)']].mean()
avg_values.rename({'Diff Power (MW)': 'Average Diff Power (MW)'}, inplace=True)

print("\nGjennomsnitt over alle profiler:")
print(avg_values)

# Filtrer bort rader utenfor driftshastigheter
filtered_df = df[(df['Hub Speed (m/s)'] >= 3) & (df['Hub Speed (m/s)'] <= 25)]

# Finn største avvik innenfor dette intervallet
max_power_idx = filtered_df['Diff Power (MW)'].abs().idxmax()
max_flapwise_idx = filtered_df['Diff Flapwise (MNm)'].abs().idxmax()
max_edgewise_idx = filtered_df['Diff Edgewise (MNm)'].abs().idxmax()

print("\nStørste kraftavvik:")
print(filtered_df.loc[max_power_idx])

print("\nStørste flapwise-avvik:")
print(filtered_df.loc[max_flapwise_idx])

print("\nStørste edgewise-avvik:")
print(filtered_df.loc[max_edgewise_idx])

# Sett scenario-navn én gang
scenario_name = 'autumn_SNII_21_hub'

# Definer mappe
folder = '/Users/hermanellingsrud/Documents/Skole/Masteroppgave/Koder/data/Baseline/Eksstremaler_gjennomsnitt/'

# Lagre gjennomsnitt
avg_output_path = f'{folder}{scenario_name}_avg.csv'
avg_values.to_frame(name='Value').to_csv(avg_output_path)
print(f"Gjennomsnitt lagret til: {avg_output_path}")


# Lagre største avvik
extremes = pd.DataFrame({
    'Max Power Deviation': filtered_df.loc[max_power_idx],
    'Max Flapwise Deviation': filtered_df.loc[max_flapwise_idx],
    'Max Edgewise Deviation': filtered_df.loc[max_edgewise_idx]
})

extremes_output_path = f'{folder}{scenario_name}_extremes.csv'
extremes.to_csv(extremes_output_path)
print(f"Ekstremaler lagret til: {extremes_output_path}")






