#!/bin/bash
init="20250208"
init_hour="12"
forward_from=120
forward_to=129
step=3
res=25
full=""
for ((current_offset=forward_from; current_offset<=forward_to; current_offset+=step)); do
    url="https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p${res}.pl?dir=%2Fgfs.${init}%2F${init_hour}%2Fatmos&file=gfs.t${init_hour}z.pgrb2${full}.0p${res}.f$(printf "%03d" $current_offset)&all_var=on&lev_2_m_above_ground=on&toplat=90&leftlon=0&rightlon=360&bottomlat=-90&lev_surface=on"

    echo "Downloading: $url"
    wget "$url" -O "gfs_${init}_${current_offset}.grib2"
done

