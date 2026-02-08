from dotenv import load_dotenv
load_dotenv()

import os
import pickle
from pathlib import Path

import faiss
import numpy as np
import requests
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent


def _env(name: str) -> str:
    value = os.environ.get(name)
    if value:
        return value
    raise RuntimeError(f"Variable d'environnement manquante: {name}")


def _supabase_headers() -> dict:
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ.get("SUPABASE_ANON_KEY")
    if not key:
        raise RuntimeError(
            "Variable d'environnement manquante: SUPABASE_SERVICE_ROLE_KEY (ou SUPABASE_ANON_KEY)"
        )
    return {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Accept": "application/json",
    }


def _fetch_all_ptc_offers(rest_url: str, headers: dict, batch_size: int = 1000) -> list[dict]:
    offers: list[dict] = []
    offset = 0
    while True:
        params = {
            "select": "id,title,description,category,device_type,points,is_active",
            "is_active": "eq.true",
            "limit": str(batch_size),
            "offset": str(offset),
        }
        resp = requests.get(
            f"{rest_url}/ptc_offers",
            params=params,
            headers=headers,
            timeout=30,
        )
        if resp.status_code != 200:
            raise RuntimeError(
                f"Erreur Supabase ptc_offers: HTTP {resp.status_code} - {resp.text}"
            )
        batch = resp.json() or []
        offers.extend(batch)
        if len(batch) < batch_size:
            break
        offset += batch_size
    return offers


def main() -> None:
    supabase_url = _env("SUPABASE_URL").rstrip("/")
    rest_url = f"{supabase_url}/rest/v1"
    headers = _supabase_headers()

    offers_raw = _fetch_all_ptc_offers(rest_url, headers)
    if not offers_raw:
        raise RuntimeError("Aucune offre active trouvée dans ptc_offers (is_active=true).")

    meta: list[dict] = []
    texts: list[str] = []

    for o in offers_raw:
        offer_id = o.get("id")
        title = o.get("title") or ""
        description = o.get("description") or ""
        category = o.get("category") or ""
        device_type = o.get("device_type") or ""
        points = o.get("points") or 0

        text = f"{title} {description} {category} {device_type} {points}"
        texts.append(text.strip())

        meta.append(
            {
                "offer_id": offer_id,
                "title": title,
                "points": points,
                "category": category,
                "device_type": device_type,
            }
        )

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts)
    embeddings = np.asarray(embeddings, dtype=np.float32)
    faiss.normalize_L2(embeddings)

    d = int(embeddings.shape[1])
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)

    faiss.write_index(index, str(BASE_DIR / "offers.faiss"))
    with open(BASE_DIR / "offers_meta.pkl", "wb") as f:
        pickle.dump(meta, f)

    print(f"OK index offers créé: {len(meta)}")


if __name__ == "__main__":
    main()
