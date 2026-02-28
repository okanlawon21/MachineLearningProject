"""
==========================================================
  ERA5 Climate Data Downloader – Abuja, Nigeria (2000–2024)
==========================================================
Downloads ERA5 reanalysis single-level data for Abuja
(Lat: 9.06 °N, Lon: 7.49 °E) for every year in 2000–2024.

Variables downloaded
---------------------
- 2 m temperature                  (2m_temperature)
- 2 m dewpoint temperature         (2m_dewpoint_temperature)
  → used to derive relative humidity
- Total precipitation              (total_precipitation)  [rainfall proxy]
- 10 m U-component of wind         (10m_u_component_of_wind)
- 10 m V-component of wind         (10m_v_component_of_wind)
  → combine U & V to get wind speed
- Surface pressure                 (surface_pressure)
- Total cloud cover                (total_cloud_cover)

Output
------
One NetCDF file per year: era5_abuja_<YEAR>.nc
All files are written to the ./era5_data/ directory.

Prerequisites
-------------
1. Install the CDS API client:
       pip install cdsapi
2. Create a CDS API credentials file at ~/.cdsapirc:
       url: https://cds.climate.copernicus.eu/api/v2
       key: <YOUR_UID>:<YOUR_API_KEY>
   See https://cds.climate.copernicus.eu/api-how-to for details.
"""

import os
import cdsapi

# ----------------------------------------------------------
# Configuration
# ----------------------------------------------------------
LATITUDE = 9.06   # Abuja, Nigeria
LONGITUDE = 7.49

# ERA5 area box [North, West, South, East] – 0.5° buffer around the point
AREA = [
    LATITUDE + 0.5,   # North
    LONGITUDE - 0.5,  # West
    LATITUDE - 0.5,   # South
    LONGITUDE + 0.5,  # East
]

START_YEAR = 2000
END_YEAR = 2024

# ERA5 variables (single levels)
VARIABLES = [
    "2m_temperature",
    "2m_dewpoint_temperature",    # used to derive relative humidity
    "total_precipitation",        # rainfall rate proxy
    "10m_u_component_of_wind",    # east–west wind component
    "10m_v_component_of_wind",    # north–south wind component
    "surface_pressure",
    "total_cloud_cover",
]

# All months, all days, every 6 hours (daily / sub-daily resolution)
MONTHS = [f"{m:02d}" for m in range(1, 13)]
DAYS = [f"{d:02d}" for d in range(1, 32)]
HOURS = ["00:00", "06:00", "12:00", "18:00"]

OUTPUT_DIR = "era5_data"

# ----------------------------------------------------------
# Download
# ----------------------------------------------------------


def download_era5(start_year: int = START_YEAR, end_year: int = END_YEAR) -> None:
    """Download ERA5 data for Abuja for each year in [start_year, end_year]."""

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    client = cdsapi.Client()

    for year in range(start_year, end_year + 1):
        output_path = os.path.join(OUTPUT_DIR, f"era5_abuja_{year}.nc")

        if os.path.exists(output_path):
            print(f"[INFO] {output_path} already exists – skipping.")
            continue

        print(f"[INFO] Requesting ERA5 data for year {year} …")

        client.retrieve(
            "reanalysis-era5-single-levels",
            {
                "product_type": "reanalysis",
                "variable": VARIABLES,
                "year": str(year),
                "month": MONTHS,
                "day": DAYS,
                "time": HOURS,
                "area": AREA,
                "format": "netcdf",
            },
            output_path,
        )

        print(f"[INFO] Saved → {output_path}")

    print("[INFO] All downloads complete.")


if __name__ == "__main__":
    download_era5()
