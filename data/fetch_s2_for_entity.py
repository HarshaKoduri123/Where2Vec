import uuid
import os
import time
import numpy as np
import psycopg2

from shapely.geometry import box
from pystac_client import Client
import planetary_computer as pc

import rasterio
from rasterio.windows import from_bounds, Window
from rasterio.warp import transform_bounds
from rasterio.enums import Resampling

CHIP_SIZE = 100          
RESOLUTION = 10         
OUT_DIR = "s2_chips"
os.makedirs(OUT_DIR, exist_ok=True)

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"

DATE_RANGE = "2022-01-01/2024-12-31"
CLOUD_THRESHOLD = 20
BANDS = ["B02", "B03", "B04", "B08"]

BATCH_LIMIT = 2000       
SLEEP_BETWEEN = 0.1      


conn = psycopg2.connect(
    dbname="geokb",
    user="geokb_user",
    password="geokb_pass",
    host="localhost",
    port=5432
)
cur = conn.cursor()

catalog = Client.open(
    STAC_URL,
    modifier=pc.sign_inplace
)


cur.execute("""
    SELECT e.entity_id,
           ST_X(e.centroid::geometry) AS lon,
           ST_Y(e.centroid::geometry) AS lat
    FROM geo_entity e
    WHERE e.entity_type IN ('water','forest','farmland')
    AND e.entity_type <> 'administrative'
    AND NOT EXISTS (
        SELECT 1 FROM s2_observation s
        WHERE s.entity_id = e.entity_id
    )

    LIMIT %s
""", (BATCH_LIMIT,))

entities = cur.fetchall()
print(f"Processing {len(entities)} entities")

for idx, (entity_id, lon, lat) in enumerate(entities, 1):
    print(f"[{idx}/{len(entities)}] Entity {entity_id}")

    try:

        half_size_deg = (CHIP_SIZE * RESOLUTION) / 111_000 / 2
        aoi = box(
            lon - half_size_deg,
            lat - half_size_deg,
            lon + half_size_deg,
            lat + half_size_deg
        )
        bbox = list(aoi.bounds)


        search = catalog.search(
            collections=["sentinel-2-l2a"],
            bbox=bbox,
            datetime=DATE_RANGE,
            query={"eo:cloud_cover": {"lt": CLOUD_THRESHOLD}},
            limit=1
        )

        items = list(search.items())
        if not items:
            continue

        item = items[0]


        arrays = []

        for b in BANDS:
            with rasterio.open(item.assets[b].href) as src:
                bbox_proj = transform_bounds(
                    "EPSG:4326",
                    src.crs,
                    *bbox,
                    densify_pts=5
                )

                window = from_bounds(*bbox_proj, transform=src.transform)
                window = window.intersection(
                    Window(0, 0, src.width, src.height)
                )

                if window.width <= 0 or window.height <= 0:
                    raise RuntimeError("Invalid raster window")

                data = src.read(
                    1,
                    window=window,
                    out_shape=(CHIP_SIZE, CHIP_SIZE),
                    resampling=Resampling.nearest
                )

                arrays.append(data)

        chip = np.stack(arrays, axis=0)  

        obs_id = str(uuid.uuid4())
        chip_path = os.path.join(OUT_DIR, f"{obs_id}.npz")
        np.savez_compressed(chip_path, chip=chip)

        cur.execute("""
            INSERT INTO s2_observation
            (obs_id, entity_id, dt, cloud_cover,
             product_id, source, geom, chip_path)
            VALUES (
                %s, %s, %s, %s,
                %s, %s,
                ST_GeomFromText(%s, 4326),
                %s
            )
        """, (
            obs_id,
            entity_id,
            item.datetime,
            item.properties["eo:cloud_cover"],
            item.id,
            "PLANETARY_COMPUTER",
            aoi.wkt,
            chip_path
        ))

        conn.commit()
        print("saved")
        time.sleep(SLEEP_BETWEEN)

    except Exception as e:
        print("error:", e)
        conn.rollback()

cur.close()
conn.close()

