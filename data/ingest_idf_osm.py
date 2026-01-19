import geopandas as gpd
import psycopg2
import uuid

gdf = gpd.read_file("idf_osm.gpkg")

print("Total rows:", len(gdf))


conn = psycopg2.connect(
    dbname="geokb",
    user="geokb_user",
    password="geokb_pass",
    host="localhost",
    port=5432
)

cur = conn.cursor()
inserted = 0


for _, r in gdf.iterrows():

    
    if r.geometry is None or r.geometry.is_empty:
        continue
    if "name" not in r or not r["name"]:
        continue

    eid = str(uuid.uuid4())

    geom = r.geometry


    if not geom.is_valid:
        geom = geom.buffer(0)

 
    geom = geom.simplify(0.001, preserve_topology=True)
    if geom.is_empty:
        continue


    entity_type = (
        r.get("place")
        or r.get("boundary")
        or r.get("natural")
        or r.get("landuse")
    )


    cur.execute("""
        INSERT INTO geo_entity
        (entity_id, canonical_name, entity_type, geometry, centroid, bbox)
        VALUES (
            %s, %s, %s,
            ST_GeomFromText(%s, 4326),   -- main geometry (POINT / POLYGON / MULTIPOLYGON)
            ST_GeogFromText(%s),         -- centroid (GEOGRAPHY)
            ST_GeomFromText(%s, 4326)    -- bbox (GEOMETRY) 
        )
    """, (
        eid,
        r["name"],
        entity_type,
        geom.wkt,
        geom.centroid.wkt,
        geom.envelope.wkt
    ))


    cur.execute("""
        INSERT INTO geo_alias (alias, entity_id, language, source)
        VALUES (%s, %s, %s, %s)
    """, (
        r["name"],
        eid,
        None,
        "OSM"
    ))

    inserted += 1


conn.commit()
cur.close()
conn.close()

print("Inserted entities:", inserted)
