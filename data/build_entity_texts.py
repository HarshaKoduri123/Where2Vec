import psycopg2
import random
import csv
import os

DB_CONFIG = dict(
    dbname="geokb",
    user="geokb_user",
    password="geokb_pass",
    host="localhost",
    port=5432
)

ENTITY_TYPES = ("water", "forest", "farmland")
MAX_ALIASES = 5

OUT_CSV = "multimodal_dataset.csv"


def build_entity_text(
    cur,
    entity_id,
    canonical_name,
    entity_type,
    max_aliases=2
):
    
    cur.execute("""
        SELECT template
        FROM entity_text_template
        WHERE entity_type = %s
        ORDER BY random()
        LIMIT 1
    """, (entity_type,))
    row = cur.fetchone()
    if row is None:
        return None

    template = row[0]

    cur.execute("""
        SELECT alias
        FROM geo_alias
        WHERE entity_id = %s
        ORDER BY random()
        LIMIT %s
    """, (entity_id, max_aliases))
    aliases = [r[0] for r in cur.fetchall()]

    if aliases:
        name_part = (
            f"{canonical_name} "
            f"(also known as {', '.join(aliases)})"
        )
    else:
        name_part = canonical_name

    return template.format(name=name_part)



def main():
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    cur.execute("""
        SELECT
            s.obs_id,
            s.chip_path,
            e.entity_id,
            e.canonical_name,
            e.entity_type
        FROM s2_observation s
        JOIN geo_entity e ON e.entity_id = s.entity_id
        WHERE e.entity_type IN ('water','forest','farmland')
        ORDER BY s.obs_id
    """)

    rows = cur.fetchall()
    print(f"Exporting {len(rows)} multimodal samples")

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "obs_id",
            "entity_id",
            "entity_type",
            "chip_path",
            "text"
        ])

        for obs_id, chip_path, eid, name, etype in rows:
            if not chip_path or not os.path.exists(chip_path):
                continue

            text = build_entity_text(
                cur=cur,
                entity_id=eid,
                canonical_name=name,
                entity_type=etype,
                max_aliases=MAX_ALIASES
            )

            if not text:
                continue

            writer.writerow([
                obs_id,
                eid,
                etype,
                chip_path,
                text
            ])

    cur.close()
    conn.close()

    print(f"CSV written to: {OUT_CSV}")


if __name__ == "__main__":
    main()
