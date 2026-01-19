
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";


CREATE TABLE geo_entity (
    entity_id UUID PRIMARY KEY,
    canonical_name TEXT,
    entity_type TEXT,
    geometry GEOMETRY,             
    centroid GEOGRAPHY(POINT),
    bbox GEOMETRY
);

CREATE TABLE geo_alias (
    alias TEXT,
    entity_id UUID,
    language TEXT,
    source TEXT
);

CREATE TABLE geo_source_ref (
    entity_id UUID,
    source TEXT,
    source_id TEXT
);

CREATE TABLE IF NOT EXISTS s2_observation (
  obs_id UUID PRIMARY KEY,
  entity_id UUID,
  dt TIMESTAMP,
  cloud_cover REAL,
  product_id TEXT,
  source TEXT,             
  geom GEOMETRY,           
  chip_path TEXT            
);

CREATE TABLE entity_text_template (
    template_id SERIAL PRIMARY KEY,
    entity_type TEXT NOT NULL,
    template TEXT NOT NULL
);


CREATE INDEX IF NOT EXISTS s2_obs_entity_idx ON s2_observation(entity_id);
CREATE INDEX IF NOT EXISTS s2_obs_geom_idx ON s2_observation USING GIST(geom);


CREATE INDEX geo_entity_geom_idx
    ON geo_entity
    USING GIST (geometry);

CREATE INDEX geo_alias_idx
    ON geo_alias
    USING GIN (alias gin_trgm_ops);


CREATE INDEX IF NOT EXISTS geo_entity_name_idx
ON geo_entity (canonical_name);

CREATE INDEX IF NOT EXISTS geo_entity_centroid_idx
ON geo_entity
USING GIST (centroid);
