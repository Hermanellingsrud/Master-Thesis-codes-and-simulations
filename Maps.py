#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 13:28:59 2024

@author: hermanellingsrud
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import netCDF4 as nc

# Specify the path to your NetCDF file
file_path = '/Users/hermanellingsrud/Documents/Skole/Masteroppgave/03/fc20200302.nc'

# Open the NetCDF file
dataset_n3 = nc.Dataset(file_path)

variable_names = dataset_n3.variables.keys()
print(variable_names)


# Retrieve wind data
u_x = dataset_n3.variables['x_wind_10m'][:]  # Wind speed in x-direction
u_y = dataset_n3.variables['y_wind_10m'][:]  # Wind speed in y-direction

# Calculate total wind speed
u = np.sqrt(u_x**2 + u_y**2)

# Retrieve coordinates
lat = dataset_n3.variables['latitude'][:]
lon = dataset_n3.variables['longitude'][:]

# Calculate total wind speed for a specific time step (e.g., first time step)
tot_wind = np.sqrt(u[0, 0, :, :]**2 + u[1, 0, :, :]**2)

# Set up the map
center = (6.58766667, 62.0)  # Adjusted central latitude to better fit the extent
map_extent = [-20, 40, 40, 90]  # Adjusted extent to include up to 74°N

fig, ax = plt.subplots(1, 1, figsize=(16, 10), subplot_kw={'projection': ccrs.NorthPolarStereo(central_longitude=center[0])})
ax.coastlines('10m')
ax.set_title('Wind Speed 02.03.2020')

# Gridlines with labels (updated for deprecation warnings)
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=1, color='gray', alpha=0.5)
gl.top_labels = False  # Hide top labels
gl.right_labels = False  # Hide right labels
gl.xlabel_style = {'size': 12, 'color': 'black'}
gl.ylabel_style = {'size': 12, 'color': 'black'}


# Plot land masses with shading
land = cfeature.NaturalEarthFeature(
    category='physical',
    name='land',
    scale='10m',
    facecolor=cfeature.COLORS['land']
)
ax.add_feature(land)

# Plot wind data with higher zorder to overlay land features
cs = ax.pcolormesh(lon, lat, tot_wind, transform=ccrs.PlateCarree(), cmap=plt.cm.coolwarm, zorder=2)
cs.set_clim(0, 25)  # Set color limits directly on the pcolormesh object




# Add vertical colorbar
cbar = plt.colorbar(cs, ax=ax, orientation='vertical', pad=0.05, fraction=0.02, label='Wind Speed (m/s)')

# Set extent
ax.set_extent(map_extent, crs=ccrs.PlateCarree())
ax.add_feature(cfeature.BORDERS, linestyle='--')


# Save the plot to your folder
save_path = '/Users/hermanellingsrud/Documents/Skole/Masteroppgave/Koder/figures_master/domain_outline.png'
plt.savefig(save_path, dpi=300, bbox_inches='tight')

plt.show()

#%%

# Specify the path to your NetCDF file
file_path = '/Users/hermanellingsrud/Documents/Skole/Masteroppgave/03/fc20200302.nc'

# Åpne NetCDF-filen
dataset_n3 = nc.Dataset(file_path)

# Hent ut koordinater
lat = dataset_n3.variables['latitude'][:]
lon = dataset_n3.variables['longitude'][:]

# Spesifiser koordinatene som skal markeres
mark_lon, mark_lat = 5.000, 56.867

# Sett opp kartet
center = (6.58766667, 62.0)
map_extent = [0, 10, 52, 63]

fig, ax = plt.subplots(1, 1, figsize=(16, 10), subplot_kw={'projection': ccrs.NorthPolarStereo(central_longitude=center[0])})
ax.coastlines('10m')


# Gridlines med etiketter
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='gray', alpha=0.5)
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': 12, 'color': 'black'}
gl.ylabel_style = {'size': 12, 'color': 'black'}

# Plot land med skygge
land = cfeature.NaturalEarthFeature(
    category='physical',
    name='land',
    scale='10m',
    facecolor=cfeature.COLORS['land']
)
ax.add_feature(land)

# Sett extent
ax.set_extent(map_extent, crs=ccrs.PlateCarree())
ax.add_feature(cfeature.BORDERS, linestyle='--')

# Marker koordinatet ditt
ax.plot(mark_lon, mark_lat, marker='X', color='red', markersize=10, transform=ccrs.PlateCarree())
#ax.text(mark_lon + 0.5, mark_lat, 'Koordinat (5.000, 56.867)', fontsize=12, color='red', transform=ccrs.PlateCarree())

# Vis plot
plt.show()

#%%

# Filsti til NetCDF-fil
file_path = '/Users/hermanellingsrud/Documents/Skole/Masteroppgave/03/fc20200302.nc'

# Åpne NetCDF-filen
dataset_n3 = nc.Dataset(file_path)

# Hent ut koordinater fra NetCDF-fil
lat = dataset_n3.variables['latitude'][:]
lon = dataset_n3.variables['longitude'][:]

# Spesifiser koordinatet som skal markeres
mark_lon, mark_lat = 5.000, 56.867

# Koordinatene til Sørlige Nordsjø II
coords = [
    (5.1681, 57.0933),  # Nordvest hjørne
    (5.4975, 56.7381),  # Nordøst hjørne
    (5.0336, 56.5917),  # Sørøst hjørne
    (4.6414, 56.4839),  # Sørvest hjørne
    (4.3467, 56.8233),  # Ekstra sørvest hjørne
    (5.1681, 57.0933),  # Tilbake til start for å lukke polygonet
]

# Sett opp kartet
center = (6.58766667, 62.0)
map_extent = [2, 10, 54, 60]

fig, ax = plt.subplots(1, 1, figsize=(16, 10), subplot_kw={'projection': ccrs.NorthPolarStereo(central_longitude=center[0])})
ax.coastlines('10m')

# Gridlines med etiketter
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='gray', alpha=0.5)
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': 12, 'color': 'black'}
gl.ylabel_style = {'size': 12, 'color': 'black'}

# Plot land med skygge
land = cfeature.NaturalEarthFeature(
    category='physical',
    name='land',
    scale='10m',
    facecolor=cfeature.COLORS['land']
)
ax.add_feature(land)

# Sett extent
ax.set_extent(map_extent, crs=ccrs.PlateCarree())
ax.add_feature(cfeature.BORDERS, linestyle='--')

# Marker området "Sørlige Nordsjø II" med polygon
lon_coords, lat_coords = zip(*coords)
ax.plot(lon_coords, lat_coords, marker='o', color='blue', markersize=5, linewidth=2, label='Sørlige Nordsjø II', transform=ccrs.PlateCarree())

# Marker koordinatet ditt
ax.plot(mark_lon, mark_lat, marker='X', color='red', markersize=10, label='Koordinat (5.000, 56.867)', transform=ccrs.PlateCarree())

# Legg til tittel og legende
ax.set_title('Område: Sørlige Nordsjø II og markert koordinat', fontsize=16)
ax.legend(loc='upper left')

# Vis plot
plt.show()

#%%
# Filsti til NetCDF-fil
file_path = '/Users/hermanellingsrud/Documents/Skole/Masteroppgave/03/fc20200302.nc'

# Åpne NetCDF-filen
dataset_n3 = nc.Dataset(file_path)

# Hent ut koordinater fra NetCDF-fil
lat = dataset_n3.variables['latitude'][:]
lon = dataset_n3.variables['longitude'][:]

# Spesifiser koordinatet som skal markeres
mark_lon, mark_lat = 5.000, 56.867

# Koordinatene til Sørvest F med 7 punkter
coords_sv_f = [
    (4.9025, 57.0003),  # Nordøst hjørne
    (4.5197, 57.2219),  # Nordvest hjørne
    (4.4483, 57.2247),  # Nordvest ekstra punkt
    (4.5928, 56.4703),  # Sørvest ekstra punkt
    (5.0319, 56.5911),  # Sørvest hjørne
    (5.4975, 56.7381),  # Sørøst hjørne
    (5.3458, 56.9033),  # Sørøst ekstra punkt
    (4.9025, 57.0003),  # Tilbake til start for å lukke polygonet
]

# Sett opp kartet
center = (6.58766667, 62.0)
map_extent = [3, 10, 56, 60]

fig, ax = plt.subplots(1, 1, figsize=(16, 10), subplot_kw={'projection': ccrs.NorthPolarStereo(central_longitude=center[0])})
ax.coastlines('10m')

# Gridlines med etiketter
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='gray', alpha=0.5)
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': 16, 'color': 'black'}
gl.ylabel_style = {'size': 16, 'color': 'black'}

# Plot land med skygge
land = cfeature.NaturalEarthFeature(
    category='physical',
    name='land',
    scale='10m',
    facecolor=cfeature.COLORS['land']
)
ax.add_feature(land)

# Sett extent
ax.set_extent(map_extent, crs=ccrs.PlateCarree())
ax.add_feature(cfeature.BORDERS, linestyle='--')

# Marker området "Sørvest F" med polygon
lon_coords, lat_coords = zip(*coords_sv_f)
ax.plot(lon_coords, lat_coords, marker='o', color='blue', markersize=5, linewidth=2, label='Sørvest F', transform=ccrs.PlateCarree())

# Marker koordinatet ditt
ax.plot(mark_lon, mark_lat, marker='X', color='red', markersize=10, label='NORA3 Data (5.000, 56.867)', transform=ccrs.PlateCarree())

# Legg til tittel og juster plassering av legenden
#ax.set_title('Område: Sørvest F og markert koordinat', fontsize=16)
ax.legend(loc='upper left', fontsize = 16)

# Vis plot
plt.show()
#%%
# Koordinatene til Sørlige Nordsjø II
coords_ns_ii = [
    (5.1681, 57.0933),  # Nordvest hjørne
    (5.4975, 56.7381),  # Nordøst hjørne
    (5.0336, 56.5917),  # Sørøst hjørne
    (4.6414, 56.4839),  # Sørvest hjørne
    (4.3467, 56.8233),  # Ekstra sørvest hjørne
    (5.1681, 57.0933),  # Tilbake til start for å lukke polygonet
]

# Koordinatene til Sørvest F
coords_sv_f = [
    (4.9025, 57.0003),  # Nordøst hjørne
    (4.5197, 57.2219),  # Nordvest hjørne
    (4.4483, 57.2247),  # Nordvest ekstra punkt
    (4.5928, 56.4703),  # Sørvest ekstra punkt
    (5.0319, 56.5911),  # Sørvest hjørne
    (5.4975, 56.7381),  # Sørøst hjørne
    (5.3458, 56.9033),  # Sørøst ekstra punkt
    (4.9025, 57.0003),  # Tilbake til start for å lukke polygonet
]

# Sett opp kartet
center = (6.58766667, 62.0)
map_extent = [3, 9, 56, 59]

fig, ax = plt.subplots(1, 1, figsize=(16, 10), subplot_kw={'projection': ccrs.NorthPolarStereo(central_longitude=center[0])})
ax.coastlines('10m')

# Gridlines med etiketter
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='gray', alpha=0.5)
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': 14, 'color': 'black'}
gl.ylabel_style = {'size': 14, 'color': 'black'}

# Plot land med skygge
land = cfeature.NaturalEarthFeature(
    category='physical',
    name='land',
    scale='10m',
    facecolor=cfeature.COLORS['land']
)
ax.add_feature(land)

# Sett extent
ax.set_extent(map_extent, crs=ccrs.PlateCarree())
ax.add_feature(cfeature.BORDERS, linestyle='--')

# Plot "Sørlige Nordsjø II" med polygon
lon_coords_ns, lat_coords_ns = zip(*coords_ns_ii)
ax.plot(lon_coords_ns, lat_coords_ns, marker='o', color='green', markersize=5, linewidth=2, label='Sørlige Nordsjø II', transform=ccrs.PlateCarree())

# Plot "Sørvest F" med polygon
lon_coords_sv, lat_coords_sv = zip(*coords_sv_f)
ax.plot(lon_coords_sv, lat_coords_sv, marker='o', color='blue', markersize=5, linewidth=2, label='Sørvest F', transform=ccrs.PlateCarree())

# Marker koordinatet ditt
ax.plot(mark_lon, mark_lat, marker='X', color='red', markersize=10, label='NORA3 Data (5.000, 56.867)', transform=ccrs.PlateCarree())

# Legg til byer ved Sørlige Nordsjø II
cities_ns_ii = {
    "Kristiansand": (8.000, 58.146),
    "Egersund": (6.004, 58.451),
    "Farsund": (6.804, 58.091),
}

for city, (lon, lat) in cities_ns_ii.items():
    ax.text(lon, lat, city, fontsize=14, color='black', transform=ccrs.PlateCarree(), weight='bold')

cities_denmark = {
    "Thyborøn": (8.217, 56.701),
}

for city, (lon, lat) in cities_denmark.items():
    ax.text(lon, lat, city, fontsize=14, color='black', transform=ccrs.PlateCarree(), weight='bold')

# Legg til tittel og legende
#ax.set_title('Områder: Sørlige Nordsjø II og Sørvest F med markert koordinat', fontsize=16)
ax.legend(loc='upper left', fontsize=16)

output_path = '/Users/hermanellingsrud/Documents/Skole/Masteroppgave/Koder/data/Kart/SNII_kart.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Figuren er lagret som: {output_path}")

# Vis plot
plt.show()


#%%

# Koordinatene til Vestavind F
coords_vestavind_f = [
    (4.6181, 59.6286),  # Nordøst hjørne
    (4.2364, 59.6286),  # Nordvest hjørne
    (4.2339, 59.0294),  # Midtre vest hjørne
    (4.2897, 58.9992),  # Sørvest hjørne
    (4.7983, 59.0003),  # Sørøst hjørne
    (4.8122, 59.1050),  # Øvre sørøst hjørne
    (4.6181, 59.6286),  # Tilbake til start for å lukke polygonet
]

center_point = (4.5, 59.2)
# Sett opp kartet
center = (4.5, 59.2)
map_extent = [3, 7, 58, 61]

fig, ax = plt.subplots(1, 1, figsize=(11, 13), subplot_kw={'projection': ccrs.NorthPolarStereo(central_longitude=center[0])})
ax.coastlines('10m')

# Gridlines med etiketter
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='gray', alpha=0.5)
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': 16, 'color': 'black'}
gl.ylabel_style = {'size': 16, 'color': 'black'}

# Plot land med skygge
land = cfeature.NaturalEarthFeature(
    category='physical',
    name='land',
    scale='10m',
    facecolor=cfeature.COLORS['land']
)
ax.add_feature(land)

# Sett extent
ax.set_extent(map_extent, crs=ccrs.PlateCarree())
ax.add_feature(cfeature.BORDERS, linestyle='--')

# Marker området "Vestavind F" med polygon
lon_coords, lat_coords = zip(*coords_vestavind_f)
ax.plot(lon_coords, lat_coords, marker='x', color='blue', markersize=5, linewidth=2, label='Vestavind F', transform=ccrs.PlateCarree())

# Marker midtpunktet
ax.plot(center_point[0], center_point[1], marker='X', color='red', markersize=10, transform=ccrs.PlateCarree(), label='NORA3 Data (4.5, 59.2)')

# Legg til tittel og juster plassering av legenden
#ax.set_title('Område: Vestavind F', fontsize=16)
ax.legend(loc='upper left', fontsize = 16)

# Vis plot
plt.show()
#%%

# Koordinatene til Vestavind F
coords_vestavind_f = [
    (4.6181, 59.6286),  # Nordøst hjørne
    (4.2364, 59.6286),  # Nordvest hjørne
    (4.2339, 59.0294),  # Midtre vest hjørne
    (4.2897, 58.9992),  # Sørvest hjørne
    (4.7983, 59.0003),  # Sørøst hjørne
    (4.8122, 59.1050),  # Øvre sørøst hjørne
    (4.6181, 59.6286),  # Tilbake til start for å lukke polygonet
]

# Koordinatene til Utsira Nord (korrigert)
coords_utsira_nord = [
    (4.4075, 59.0694),  # Sørvest hjørne
    (4.2692, 59.4481),  # Nordvest hjørne
    (4.6736, 59.4822),  # Nordøst hjørne
    (4.8122, 59.1050),  # Sørøst hjørne
    (4.4075, 59.0694),  # Tilbake til start for å lukke polygonet
]

# Midtpunktet til Vestavind F
center_point_vestavind_f = (4.508, 59.195)



# Sett opp kartet
center = (4.5, 59.2)
map_extent = [3, 7, 58, 60.4]

fig, ax = plt.subplots(1, 1, figsize=(11, 13), subplot_kw={'projection': ccrs.NorthPolarStereo(central_longitude=center[0])})
ax.coastlines('10m')

# Gridlines med etiketter
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='gray', alpha=0.5)
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': 16, 'color': 'black'}
gl.ylabel_style = {'size': 16, 'color': 'black'}

# Plot land med skygge
land = cfeature.NaturalEarthFeature(
    category='physical',
    name='land',
    scale='10m',
    facecolor=cfeature.COLORS['land']
)
ax.add_feature(land)

# Sett extent
ax.set_extent(map_extent, crs=ccrs.PlateCarree())
ax.add_feature(cfeature.BORDERS, linestyle='--')

# Marker området "Vestavind F" med polygon
lon_coords_vf, lat_coords_vf = zip(*coords_vestavind_f)
ax.plot(lon_coords_vf, lat_coords_vf, marker='x', color='blue', markersize=5, linewidth=2, label='Vestavind F', transform=ccrs.PlateCarree())

# Marker området "Utsira Nord" med polygon
lon_coords_un, lat_coords_un = zip(*coords_utsira_nord)
ax.plot(lon_coords_un, lat_coords_un, marker='o', color='green', markersize=5, linewidth=2, label='Utsira Nord', transform=ccrs.PlateCarree())

# Marker midtpunktet for begge områder
ax.plot(center_point_vestavind_f[0], center_point_vestavind_f[1], marker='X', color='red', markersize=10, transform=ccrs.PlateCarree(), label='NORA3 Data (4.508, 59.195)')


# Legg til byer
cities = {
    "Haugesund": (5.267, 59.414),
    "Stavanger": (5.733, 58.970),
    "Bryne": (5.646, 58.735),
    "Leirvik": (5.299, 59.779),
    "Sauda": (6.353, 59.650),
}

for city, (lon, lat) in cities.items():
    ax.text(lon, lat, city, fontsize=14, color='black', transform=ccrs.PlateCarree(), weight='bold')
# Legg til tittel og legende
#ax.set_title('Områder: Vestavind F og Utsira Nord', fontsize=16)
ax.legend(loc='upper left', fontsize=16)

output_path = '/Users/hermanellingsrud/Documents/Skole/Masteroppgave/Koder/data/Kart/Utsira_kart.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Figuren er lagret som: {output_path}")
# Vis plot
plt.show()

#%%


# Sett opp kartet
center = (6.5, 62)
map_extent = [3, 21, 55, 72]

fig, ax = plt.subplots(1, 1, figsize=(10, 14), subplot_kw={'projection': ccrs.NorthPolarStereo(central_longitude=center[0])})
ax.coastlines('10m')

# Gridlines with labels, adjusted for better visibility
gl = ax.gridlines(
    crs=ccrs.PlateCarree(),
    draw_labels=True,
    linewidth=1,
    color='gray',
    alpha=0.5,
    x_inline=False,
    y_inline=False
)
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': 18, 'color': 'black', 'weight': 'bold'}
gl.ylabel_style = {'size': 18, 'color': 'black', 'weight': 'bold'}

# Plot land med skygge
land = cfeature.NaturalEarthFeature(
    category='physical',
    name='land',
    scale='10m',
    facecolor=cfeature.COLORS['land']
)
ax.add_feature(land)

# Sett extent
ax.set_extent(map_extent, crs=ccrs.PlateCarree())
ax.add_feature(cfeature.BORDERS, linestyle='--')

# Fill the "Vestavind F" area
lon_vest_f, lat_vest_f = zip(*coords_vestavind_f)
ax.fill(lon_vest_f, lat_vest_f, color='blue', alpha=0.7, transform=ccrs.PlateCarree(), label='Vestavind F')

# Fill the "Sørvest F" area
lon_sv_f, lat_sv_f = zip(*coords_sv_f)
ax.fill(lon_sv_f, lat_sv_f, color='red', alpha=0.7, transform=ccrs.PlateCarree(), label='Sørvest F')

# Legg til tittel og juster plassering av legenden
#ax.set_title('Study Locations', fontsize=16)
ax.legend(loc='upper left', fontsize = 20)

# Lagre figuren
output_path = '/Users/hermanellingsrud/Documents/Skole/Masteroppgave/Koder/data/Kart/Norges_kart.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Figuren er lagret som: {output_path}")
# Vis plot
plt.show()

#%%

# Sett opp kartet
center = (6.5, 62)
map_extent = [3, 10, 56, 60.5]

fig, ax = plt.subplots(1, 1, figsize=(12, 12), subplot_kw={'projection': ccrs.NorthPolarStereo(central_longitude=center[0])})
ax.coastlines('10m')

# Gridlines with labels, adjusted for better visibility
gl = ax.gridlines(
    crs=ccrs.PlateCarree(),
    draw_labels=True,
    linewidth=1,
    color='gray',
    alpha=0.5,
    x_inline=False,
    y_inline=False
)
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': 18, 'color': 'black', 'weight': 'bold'}
gl.ylabel_style = {'size': 18, 'color': 'black', 'weight': 'bold'}

# Plot land med skygge
land = cfeature.NaturalEarthFeature(
    category='physical',
    name='land',
    scale='10m',
    facecolor=cfeature.COLORS['land']
)
ax.add_feature(land)

# Sett extent
ax.set_extent(map_extent, crs=ccrs.PlateCarree())
ax.add_feature(cfeature.BORDERS, linestyle='--')

# Fill the "Vestavind F" area
lon_vest_f, lat_vest_f = zip(*coords_vestavind_f)
ax.fill(lon_vest_f, lat_vest_f, color='blue', alpha=0.7, transform=ccrs.PlateCarree(), label='Vestavind F')

# Fill the "Sørvest F" area
lon_sv_f, lat_sv_f = zip(*coords_sv_f)
ax.fill(lon_sv_f, lat_sv_f, color='red', alpha=0.7, transform=ccrs.PlateCarree(), label='Sørvest F')

# Legg til tittel og juster plassering av legenden
#ax.set_title('Study Locations', fontsize=16)
ax.legend(loc='upper left', fontsize = 16)

# Legg til byer
cities = {
    "Haugesund": (5.267, 59.414),
    "Stavanger": (5.733, 58.970),
}

for city, (lon, lat) in cities.items():
    ax.text(lon, lat, city, fontsize=16, color='black', transform=ccrs.PlateCarree(), weight='bold')
    
# Legg til byer ved Sørlige Nordsjø II
cities_ns_ii = {
    "Kristiansand": (8.000, 58.146),
}

for city, (lon, lat) in cities_ns_ii.items():
    ax.text(lon, lat, city, fontsize=16, color='black', transform=ccrs.PlateCarree(), weight='bold')

cities_denmark = {
    "Thyborøn": (8.217, 56.701),
}

for city, (lon, lat) in cities_denmark.items():
    ax.text(lon, lat, city, fontsize=20, color='black', transform=ccrs.PlateCarree(), weight='bold')

output_path = '/Users/hermanellingsrud/Documents/Skole/Masteroppgave/Koder/data/Kart/Nærmere_kart.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Figuren er lagret som: {output_path}")
# Vis plot
plt.show()

#%%


# Sett opp kartet
center = (6.5, 62)
map_extent = [3, 10, 53, 60.5]

fig, ax = plt.subplots(1, 1, figsize=(12, 12), subplot_kw={'projection': ccrs.NorthPolarStereo(central_longitude=center[0])})
ax.coastlines('10m')

# Gridlines with labels, adjusted for better visibility
gl = ax.gridlines(
    crs=ccrs.PlateCarree(),
    draw_labels=True,
    linewidth=1,
    color='gray',
    alpha=0.5,
    x_inline=False,
    y_inline=False
)
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': 18, 'color': 'black', 'weight': 'bold'}
gl.ylabel_style = {'size': 18, 'color': 'black', 'weight': 'bold'}

# Plot land med skygge
land = cfeature.NaturalEarthFeature(
    category='physical',
    name='land',
    scale='10m',
    facecolor=cfeature.COLORS['land']
)
ax.add_feature(land)

mark_lon_sn, mark_lat_sn = 5.000, 56.867

center_point_vestavind_f = (4.508, 59.195)


# Sett extent
ax.set_extent(map_extent, crs=ccrs.PlateCarree())
ax.add_feature(cfeature.BORDERS, linestyle='--')

ax.plot(mark_lon_sn, mark_lat_sn, marker='o', color='red', markersize=10, transform=ccrs.PlateCarree())
ax.text(mark_lon_sn + 0.5, mark_lat_sn, 'SNII', fontsize=16, color='red', transform=ccrs.PlateCarree())

ax.plot(center_point_vestavind_f[0], center_point_vestavind_f[1], marker='o', color='red', markersize=10, transform=ccrs.PlateCarree())
ax.text(center_point_vestavind_f[0] - 1.3, center_point_vestavind_f[1], 'Utsira', fontsize=16, color='red', transform=ccrs.PlateCarree())


# Marker punktet
lat = 54.01486
lon = 6.58764
ax.plot(lon, lat, marker='o', color='red', markersize=10, transform=ccrs.PlateCarree())
ax.text(lon + 0.2, lat, 'FINO1', transform=ccrs.PlateCarree(), fontsize=16, color='red')

plt.savefig("/Users/hermanellingsrud/Documents/Skole/Masteroppgave/Koder/figures_master/LIDAR/kart_lidar.png", dpi = 300)

plt.show()

