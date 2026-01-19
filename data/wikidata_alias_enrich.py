import psycopg2
import requests
import time
import random

SPARQL_URL = "https://query.wikidata.org/sparql"

HEADERS = {
    "Accept": "application/sparql+json",
    "User-Agent": "where2vec/0.1 (academic research; contact: your_email@domain.com)"
}

BASE_SLEEP = 2.0


def query_wikidata(name):
    name = name.replace('"', '').strip()
    if not name:
        return []

    results = []

    sparql_query = f"""
    SELECT ?altLabel WHERE {{
      ?item rdfs:label ?label .
      FILTER(LANG(?label) IN ("fr","en")) .
      FILTER(STRSTARTS(LCASE(STR(?label)), LCASE("{name}")))

      OPTIONAL {{
        ?item skos:altLabel ?altLabel .
        FILTER(LANG(?altLabel) IN ("fr","en"))
      }}
    }}
    LIMIT 5
    """

    try:
        r = requests.get(
            SPARQL_URL,
            params={"query": sparql_query},
            headers=HEADERS,
            timeout=15
        )

        if r.status_code == 200 and r.text.strip():
            for b in r.json()["results"]["bindings"]:
                if "altLabel" in b:
                    results.append({
                        "alias": b["altLabel"]["value"],
                        "language": b["altLabel"]["xml:lang"]
                    })

            if results:
                return results

    except Exception:
        pass


    try:
        r = requests.get(
            "https://www.wikidata.org/w/api.php",
            params={
                "action": "wbsearchentities",
                "search": name,
                "language": "en",
                "format": "json",
                "limit": 5
            },
            headers=HEADERS,
            timeout=10
        )

        for item in r.json().get("search", []):
            if "label" in item:
                results.append({
                    "alias": item["label"],
                    "language": "en"
                })

    except Exception:
        pass

    return results


conn = psycopg2.connect(
    dbname="geokb",
    user="geokb_user",
    password="geokb_pass",
    host="localhost",
    port=5432
)
cur = conn.cursor()


cur.execute("""
    SELECT entity_id, canonical_name, entity_type
    FROM geo_entity
    WHERE entity_type IN ('water','forest','farmland')
""")

rows = cur.fetchall()
print(f"Processing {len(rows)} entities (water / forest / farmland)")


for i, (eid, name, etype) in enumerate(rows, 1):
    print(f"[{i}/{len(rows)}] {name} ({etype})")

    try:
        aliases = query_wikidata(name)
        inserted = 0

        for item in aliases:
          
            if not item["alias"] or not item["language"]:
                continue

            cur.execute("""
                INSERT INTO geo_alias (alias, entity_id, language, source)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT DO NOTHING
            """, (
                item["alias"],
                eid,
                item["language"],
                "WIKIDATA"
            ))

            inserted += 1

        conn.commit()
        print(f"{inserted} aliases added")

        
        time.sleep(1.5 + random.random())

    except Exception as e:
        print("skipped:", e)
        conn.rollback()
        time.sleep(5)

cur.close()
conn.close()

