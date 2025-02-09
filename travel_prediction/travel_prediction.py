import pygrib
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timezone, timedelta
from mpl_toolkits.basemap import Basemap, shiftgrid, maskoceans
from geopy.distance import geodesic
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm
from time import sleep
import glob
import requests
import os
import argparse

def create_image(grib_files, init, data_from, data_to, prec_threshold, temp_threshold, distance, resolution, path=None):
    total_precip = None
    min_temp = None

    for grib_file, hours in grib_files:
        print(f"Processing {grib_file}...")

        grbs = pygrib.open(grib_file)

        try:
            grb = grbs.select(name="Total Precipitation")[0]
            grb1 = grbs.select(name="Minimum temperature")[0]
            precip_data = grb.values * hours
            temp_data = grb1.values
            
            if total_precip is None:
                total_precip = precip_data
            else:
                total_precip += precip_data
            
            if min_temp is None:
                min_temp = temp_data
            else:
                min_temp = np.minimum(temp_data, min_temp)

        except Exception as e:
            print(f"Skipping {grib_file}: {e}")

    if total_precip is None:
        raise ValueError("No valid precipitation data found in the GRIB files!")

    lons = np.linspace(float(grb['longitudeOfFirstGridPointInDegrees']), 
                       float(grb['longitudeOfLastGridPointInDegrees']), int(grb['Ni']))
    lats = np.linspace(float(grb['latitudeOfFirstGridPointInDegrees']), 
                       float(grb['latitudeOfLastGridPointInDegrees']), int(grb['Nj']))

    min_temp, _ = shiftgrid(180., min_temp, lons, start=False)
    total_precip, lons = shiftgrid(180., total_precip, lons, start=False)
    grid_lon, grid_lat = np.meshgrid(lons, lats)

    fig, ax = plt.subplots(figsize=(15, 20))
    m = Basemap(projection='cyl', llcrnrlon=-10, urcrnrlon=40, 
                llcrnrlat=35, urcrnrlat=60, resolution='i')

    x, y = m(grid_lon, grid_lat)

    cmap = plt.get_cmap('inferno')
    cmap.set_bad(alpha=0)

    min_temp -= 274.15
    masked_total_precip = np.ma.masked_where(total_precip <= prec_threshold, total_precip)
    masked_min_temp = np.ma.masked_where(min_temp <= temp_threshold, min_temp)

    #masked_total_precip = maskoceans(grid_lon, grid_lat, masked_total_precip, resolution='i', inlands=True)
    masked_min_temp = maskoceans(grid_lon, grid_lat, masked_min_temp, resolution='i', inlands=True)

    cs = m.contourf(x, y, masked_total_precip, levels=5, cmap=cmap, zorder=2)
    cs1 = m.contourf(x, y, masked_min_temp, cmap='coolwarm', shading='auto', zorder=1, levels=10)

    # Add map features
    m.drawcoastlines(zorder=2)
    m.drawcountries(zorder=2)
    m.drawparallels(np.arange(35., 61., 5.), labels=[1, 0, 0, 0], zorder=2)
    m.drawmeridians(np.arange(-10., 41., 5.), labels=[0, 0, 0, 1], zorder=2)

    # Prague and Brno coordinates
    locations = {
        "Prague": (50.08, 14.43),
        "Brno": (49.20, 16.61)
    }

    circle_radius_km = distance

    def draw_geodesic_circle(m, lat, lon, radius_km, n_points=100):
        """
        Draws a geodesic circle around a given latitude and longitude.
        """
        circle_lats = []
        circle_lons = []
        for angle in np.linspace(0, 360, n_points):
            point = geodesic(kilometers=radius_km).destination((lat, lon), angle)
            circle_lats.append(point.latitude)
            circle_lons.append(point.longitude)
        
        circle_x, circle_y = m(circle_lons, circle_lats)
        plt.plot(circle_x, circle_y, linestyle="dashed", color="black", linewidth=2, zorder=3)

    for city, (lat, lon) in locations.items():
        city_x, city_y = m(lon, lat)
        
        plt.scatter(city_x, city_y, color='black', s=50, label=city, zorder=3)
        
        draw_geodesic_circle(m, lat, lon, circle_radius_km)

    plt.title(f"Travel friendly location prediction \n Init={init.strftime("%Y-%m-%d %H:%M")}, From={data_from.strftime("%Y-%m-%d %H:%M")}, To={data_to.strftime("%Y-%m-%d %H:%M")}, Dist={distance}\ntemp>={temp_threshold} degC, prec>={prec_threshold} mm")
    plt.legend()

    divider = make_axes_locatable(ax)
    cax = divider.new_horizontal(size="2.5%", pad=0.2)
    fig.add_axes(cax)
    fig.colorbar(cs, cax=cax, orientation='vertical', shrink=0.4, label="Total Precipitation (mm)")

    cax2 = divider.new_horizontal(size="2.5%", pad=0.7, pack_start=True)
    fig.add_axes(cax2)
    cb2 = fig.colorbar(cs1, cax=cax2, orientation='vertical', shrink=0.4, label="Minimum temperature (degC)")
    cb2.ax.yaxis.set_ticks_position('left')
    
    if path:
        plt.savefig(path, bbox_inches='tight')
    else:
        plt.show()
    return plt

def hour_difference(dt1, dt2):
    return abs(int((dt2 - dt1).total_seconds() // 3600))

def download_url(url, filename, max_retries=3, timeout=10):
    if os.path.exists(filename) and os.path.getsize(filename) > 0:
        print(f"Skipping downloading {filename} (already exists)")
        return True

    retries = 0
    while retries < max_retries:
        try:
            print(f"Attempting to download: {url}")
            with requests.get(url, stream=True, timeout=timeout) as response:
                response.raise_for_status()

                total_size = int(response.headers.get('content-length', 0))
                chunk_size = 8192  # 8 KB

                with open(filename, 'wb') as file, tqdm(
                    desc=f"Downloading {filename}",
                    total=total_size if total_size > 0 else None,
                    unit='B',
                    unit_scale=True,
                    unit_divisor=1024
                ) as bar:
                    for chunk in response.iter_content(chunk_size=chunk_size):  
                        if chunk:
                            file.write(chunk)
                            bar.update(len(chunk))

            return True

        except (requests.RequestException, requests.Timeout) as e:
            retries += 1
            print(f"Download failed ({retries}/{max_retries}): {e}")

    print(f"Failed to download {filename} after {max_retries} attempts.")
    return False

def delete_old_files(base_directory, current_init_date):
    threshold_date = current_init_date - timedelta(days=5)

    for filename in os.listdir(base_directory):
        if filename.startswith("gfs_") and filename.endswith(".grib2"):
            try:
                init_str = filename.split("_")[1]  # Extract YYYYMMDD
                init_date = datetime.strptime(init_str, "%Y%m%d")
                
                if init_date < threshold_date:
                    file_path = os.path.join(base_directory, filename)
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")

            except (IndexError, ValueError) as e:
                print(f"Skipping removal of {filename}: {e}") 
                       
def download_files(init_date, data_from, data_to, resolution, step, base_directory):
    res = str(resolution).replace('.', 'p')
    init = init_date.strftime("%Y%m%d")
    init_hour = init_date.strftime("%H")
    forward_from = hour_difference(init_date, data_from)
    forward_to = hour_difference(init_date, data_to)
    full = ""
    file_list = []
    if res == "0p50":
        full = "full"
    current_offset = forward_from
    while current_offset <= forward_to:
        url = f"https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_{res}.pl?dir=%2Fgfs.{init}%2F{init_hour}%2Fatmos&file=gfs.t{init_hour}z.pgrb2{full}.{res}.f{current_offset}&all_var=on&lev_2_m_above_ground=on&toplat=90&leftlon=0&rightlon=360&bottomlat=-90&lev_surface=on"
        destination = f"{base_directory}/gfs_{init}_{init_hour}_{current_offset}.grib2"
        #print(f"Starting download: {url}")
        status = download_url(url, destination)
        if status:
            file_list.append([destination, step])
        if(current_offset >= 120):
            step = 3
        current_offset += step
    
    return file_list
    
    return files_pathes

def parse_arguments():
    parser = argparse.ArgumentParser(description="Download and process GFS model data.")

    parser.add_argument("--from", dest="from_d", type=str, required=True, help="Start time in YYYY-MM-DD-HH format.")
    parser.add_argument("--to", type=str, required=True, help="End time in YYYY-MM-DD-HH format.")
    parser.add_argument("--prec_threshold", type=float, default=3, help="Precipitation threshold (default: 3).")
    parser.add_argument("--temp_threshold", type=float, default=0, help="Temperature threshold (default: 0).")
    parser.add_argument("--distance", type=int, default=1300, help="Distance threshold in km (default: 1300).")
    parser.add_argument("--resolution", type=float, default=0.25, help="Grid resolution (default: 0.25).")
    parser.add_argument("--base_directory", type=str, default="meteodata", help="Directory for storing model data (default: meteodata).")
    parser.add_argument("--step", type=int, default=3, help="Forecast step in hours (default: 3).")
    parser.add_argument("path", nargs="?", default=None, help="Optional output directory for the generated map.")

    return parser.parse_args()

def parse_datetime(datetime_str):
    return datetime.strptime(datetime_str, "%Y-%m-%d-%H")

if __name__ == "__main__":
    args = parse_arguments()

    data_from = parse_datetime(args.from_d)
    data_to = parse_datetime(args.to)
    prec_threshold = args.prec_threshold
    temp_threshold = args.temp_threshold
    resolution = args.resolution
    step = args.step
    distance = args.distance
    base_directory = args.base_directory
    
    now = datetime.now(timezone.utc)
    current_utc_hour = now.hour

    gfs_availability = {0: 4, 6: 10, 12: 16, 18: 22}
    available_runs = [run for run, available in gfs_availability.items() if available <= current_utc_hour]

    if available_runs:
        nearest_run = max(available_runs)
        forecast_date = now.date()
    else:
        nearest_run = 18
        forecast_date = now.date() - timedelta(days=1)

    init = datetime(forecast_date.year, forecast_date.month, forecast_date.day, nearest_run, 0)
    path = None
    if args.path:
        path = args.path + f"/travel_prediction_{init.strftime('%Y-%m-%d-%H')}_{data_from.strftime('%Y-%m-%d-%H')}_{data_to.strftime('%Y-%m-%d-%H')}_t{temp_threshold}_p{prec_threshold}_d{distance}_r{int(resolution*100)}.png"

    print(f"Using nearest GFS run: {nearest_run} UTC")
    print(f"Init: {init.strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"From: {data_from.strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"To: {data_to.strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"Resolution: {resolution}°")
    print(f"Step: {step} hour(s)")
    print("------")
    print("Downloading model data...")

    grib_files = download_files(init, data_from, data_to, resolution, step, base_directory)

    if len(grib_files) > 0:
        print("Deleting old grib files...")
        delete_old_files(base_directory, init)
        print("Creating map...")
        create_image(grib_files, init, data_from, data_to, prec_threshold, temp_threshold, distance, resolution, path)
        if path:
            print(f"Map saved to: {path}")
    else:
        print("No GFS data found!")
