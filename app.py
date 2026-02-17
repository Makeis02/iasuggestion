from dotenv import load_dotenv
load_dotenv()

import os
import pickle
import json
import random
import re
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

BASE_DIR = Path(__file__).resolve().parent
resources: dict = {}








def _supabase_headers() -> dict:
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ.get("SUPABASE_ANON_KEY")
    if not key:
        raise RuntimeError("Variable d'environnement manquante: SUPABASE_SERVICE_ROLE_KEY")
    return {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Accept": "application/json",
    }


def _supabase_rest_url() -> str:
    supabase_url = os.environ.get("SUPABASE_URL")
    if not supabase_url:
        raise RuntimeError("Variable d'environnement manquante: SUPABASE_URL")
    return f"{supabase_url.rstrip('/')}/rest/v1"


def _supabase_get(table: str, params: dict, timeout_s: int = 20) -> list[dict]:
    rest_url = resources["supabase_rest_url"]
    headers = resources["supabase_headers"]
    resp = requests.get(
        f"{rest_url}/{table}",
        params=params,
        headers=headers,
        timeout=timeout_s,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"Erreur Supabase {table}: HTTP {resp.status_code} - {resp.text}")
    return resp.json() or []


def _safe_str(value) -> str:
    if value is None:
        return ""
    return str(value)


def safe_int(value, default: int = 0) -> int:
    try:
        if value is None:
            return default
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        s = str(value).strip()
        if not s:
            return default
        return int(float(s))
    except Exception:
        return default


def clamp01(value: float) -> float:
    try:
        x = float(value)
    except Exception:
        return 0.0
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def _extract_first_json_object(text: str) -> dict | None:
    s = _safe_str(text).strip()
    if not s:
        return None

    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    try:
        start = s.find("{")
        if start == -1:
            return None
        depth = 0
        for i in range(start, len(s)):
            ch = s[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = s[start : i + 1]
                    obj = json.loads(candidate)
                    if isinstance(obj, dict):
                        return obj
                    return None
    except Exception:
        return None

    return None


def _fallback_quiz(topic: str | None, difficulty: str | None) -> dict:
    topic_s = _safe_str(topic).strip() or "culture_generale"
    diff_s = _safe_str(difficulty).strip().lower() or "medium"

    bank = [
        {
            "topic": "culture_generale",
            "difficulty": "easy",
            "question": "Quel est le plus grand océan sur Terre ?",
            "answers": ["océan pacifique", "pacifique"],
            "points": 15,
        },
        {
            "topic": "culture_generale",
            "difficulty": "easy",
            "question": "Quelle planète est surnommée la planète rouge ?",
            "answers": ["mars"],
            "points": 15,
        },
        {
            "topic": "culture_generale",
            "difficulty": "medium",
            "question": "Dans quel pays se trouve la ville de Marrakech ?",
            "answers": ["maroc", "le maroc"],
            "points": 25,
        },
        {
            "topic": "culture_generale",
            "difficulty": "medium",
            "question": "Quel est l'auteur de « Les Misérables » ?",
            "answers": ["victor hugo", "hugo"],
            "points": 30,
        },
        {
            "topic": "culture_generale",
            "difficulty": "hard",
            "question": "Quelle est la capitale de la Nouvelle-Zélande ?",
            "answers": ["wellington"],
            "points": 40,
        },
    ]

    candidates = [q for q in bank if q["topic"] == topic_s and q["difficulty"] == diff_s]
    if not candidates:
        candidates = [q for q in bank if q["topic"] == topic_s] or bank

    chosen = random.choice(candidates)
    return {
        "id": str(uuid.uuid4()),
        "topic": chosen["topic"],
        "difficulty": chosen["difficulty"],
        "question": chosen["question"],
        "answers": chosen["answers"],
        "points": chosen["points"],
        "source": "fallback",
    }


def _generate_quiz_llm(topic: str | None, difficulty: str | None) -> dict | None:
    hf_key = _safe_str(os.environ.get("HUGGINGFACE_API_KEY")).strip()
    if not hf_key:
        return None
    model = _safe_str(os.environ.get("HUGGINGFACE_MODEL")).strip() or "HuggingFaceTB/SmolLM3-3B:hf-inference"

    topic_s = _safe_str(topic).strip() or "culture_generale"
    diff_s = _safe_str(difficulty).strip().lower() or "medium"

    sys_prompt = (
        "Tu génères une question de quiz de culture générale en français. "
        "Réponds UNIQUEMENT en JSON valide, sans texte autour."
    )
    user_prompt = (
        "Génère une question (une seule) et les réponses acceptées. "
        "Contraintes:\n"
        f"- topic: {topic_s}\n"
        f"- difficulté: {diff_s}\n"
        "- la question doit être claire et courte\n"
        "- réponses acceptées: 1 à 4 variantes (minuscules, sans ponctuation)\n"
        "- points: entier (easy 10-20, medium 20-35, hard 35-60)\n\n"
        "Format JSON attendu:\n"
        "{\n"
        '  "question": "…",\n'
        '  "answers": ["…"],\n'
        '  "points": 25\n'
        "}"
    )

    resp = requests.post(
        "https://router.huggingface.co/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {hf_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": 220,
            "temperature": 0.8,
            "stream": False,
        },
        timeout=35,
    )

    if not resp.ok:
        return None

    raw = resp.json() or {}
    content = ""
    try:
        content = _safe_str(((raw.get("choices") or [])[0] or {}).get("message", {}).get("content"))
    except Exception:
        content = ""

    obj = _extract_first_json_object(content)
    if not obj:
        return None

    question = _safe_str(obj.get("question")).strip()
    answers_raw = obj.get("answers")
    points_raw = obj.get("points")
    if not question:
        return None

    answers: list[str] = []
    if isinstance(answers_raw, list):
        for a in answers_raw[:4]:
            s = _safe_str(a).strip().lower()
            s = re.sub(r"\s+", " ", s)
            if s:
                answers.append(s)
    elif isinstance(answers_raw, str):
        s = _safe_str(answers_raw).strip().lower()
        s = re.sub(r"\s+", " ", s)
        if s:
            answers.append(s)
    if not answers:
        return None

    try:
        points = int(float(points_raw)) if points_raw is not None else 25
    except Exception:
        points = 25
    points = max(1, min(200, points))

    return {
        "id": str(uuid.uuid4()),
        "topic": topic_s,
        "difficulty": diff_s,
        "question": question,
        "answers": answers,
        "points": points,
        "source": "huggingface",
        "model": model,
    }


def normalize_provider(value: str | None) -> str | None:
    s = _safe_str(value).strip().lower()
    if not s:
        return None
    aliases = {
        "cpx_research": "cpx",
        "cpxresearch": "cpx",
        "cpx-research": "cpx",
        "rapido_reach": "rapidoreach",
        "rapido-reach": "rapidoreach",
        "theorem_reach": "theoremreach",
        "theorem-reach": "theoremreach",
    }
    s = aliases.get(s, s)
    return s


def _supabase_get_first_success(table: str, variants: list[dict], timeout_s: int = 20) -> list[dict]:
    last_error: Exception | None = None
    for params in variants:
        try:
            return _supabase_get(table, params=params, timeout_s=timeout_s)
        except Exception as e:
            last_error = e
            continue
    if last_error:
        raise last_error
    return []


def _to_iso(dt) -> str:
    try:
        return dt.isoformat()
    except Exception:
        return str(dt)


def _survey_provider_fallback(limit: int) -> list[dict]:
    fallback = ["cpx", "rapidoreach", "theoremreach", "bitlabs"]
    items: list[dict] = []
    for p in fallback[: max(0, int(limit))]:
        items.append({"provider": p, "score": 0.5, "reason": "fallback", "entry": p})
    return items


def _iframe_provider_fallback(limit: int) -> list[dict]:
    fallback = ["theoremreach", "timewall", "bitcotasks"]
    items: list[dict] = []
    for p in fallback[: max(0, int(limit))]:
        items.append({"provider": p, "score": 0.5, "reason": "fallback", "entry": p})
    return items


def _offerwall_provider_fallback(limit: int) -> list[dict]:
    fallback = ["revlum", "kiwiwall", "notik"]
    items: list[dict] = []
    for p in fallback[: max(0, int(limit))]:
        items.append({"provider": p, "score": 0.5, "reason": "fallback", "entry": p})
    return items


def _normalize_offerwall_provider(value: str | None) -> str | None:
    s = _safe_str(value).strip().lower()
    if not s:
        return None
    aliases = {
        "kiwi": "kiwiwall",
        "kiwi-wall": "kiwiwall",
        "kiwi_wall": "kiwiwall",
        "rev": "revlum",
        "rev-lum": "revlum",
        "rev_lum": "revlum",
    }
    s = aliases.get(s, s)
    if s in {"revlum", "kiwiwall", "notik", "mylead", "wannads"}:
        return s
    if "revlum" in s:
        return "revlum"
    if "kiwi" in s:
        return "kiwiwall"
    if "notik" in s:
        return "notik"
    if "mylead" in s:
        return "mylead"
    if "wannads" in s:
        return "wannads"
    return None


def _compute_offerwall_provider_recommendations(user_id: str, limit: int = 3) -> list[dict]:
    from datetime import datetime, timedelta, timezone

    try:
        limit_int = max(1, safe_int(limit, 3))
    except Exception:
        limit_int = 3

    candidates = {"revlum", "kiwiwall", "notik"}
    now = datetime.now(timezone.utc)
    since_impressions = _to_iso(now - timedelta(days=30))
    since_history = _to_iso(now - timedelta(days=60))

    behavior_counts: dict[str, int] = {}
    try:
        rows = _supabase_get_first_success(
            "mobile_impressions",
            variants=[
                {
                    "select": "provider,created_at",
                    "user_id": f"eq.{user_id}",
                    "created_at": f"gte.{since_impressions}",
                    "order": "created_at.desc",
                    "limit": "2000",
                },
                {
                    "select": "provider,created_at",
                    "user_id": f"eq.{user_id}",
                    "created_at": f"gte.{since_impressions}",
                    "limit": "2000",
                },
                {"select": "provider", "user_id": f"eq.{user_id}", "limit": "2000"},
            ],
            timeout_s=25,
        )
        for r in rows:
            if not isinstance(r, dict):
                continue
            p = _normalize_offerwall_provider(r.get("provider"))
            if not p or p not in candidates:
                continue
            behavior_counts[p] = behavior_counts.get(p, 0) + 1
    except Exception:
        behavior_counts = {}

    max_behavior = max(behavior_counts.values()) if behavior_counts else 0

    completed_counts: dict[str, int] = {}
    try:
        tx_rows = _supabase_get_first_success(
            "transaction_offers",
            variants=[
                {
                    "select": "status,provider,type,created_at",
                    "user_id": f"eq.{user_id}",
                    "created_at": f"gte.{since_history}",
                    "order": "created_at.desc",
                    "limit": "2000",
                },
                {
                    "select": "status,provider,type,created_at",
                    "user_id": f"eq.{user_id}",
                    "created_at": f"gte.{since_history}",
                    "limit": "2000",
                },
                {"select": "status,provider,type", "user_id": f"eq.{user_id}", "limit": "2000"},
            ],
            timeout_s=25,
        )
        for r in tx_rows:
            if not isinstance(r, dict):
                continue
            if safe_int(r.get("status"), -1) != 1:
                continue
            provider_raw = r.get("provider")
            type_raw = r.get("type")
            p = _normalize_offerwall_provider(provider_raw) or _normalize_offerwall_provider(type_raw)
            if not p or p not in candidates:
                continue
            completed_counts[p] = completed_counts.get(p, 0) + 1
    except Exception:
        completed_counts = {}

    max_completed = max(completed_counts.values()) if completed_counts else 0

    has_any_signal = bool(behavior_counts) or bool(completed_counts)
    if not has_any_signal:
        return _offerwall_provider_fallback(limit_int)

    items: list[dict] = []
    for p in sorted(candidates):
        score_behavior = (behavior_counts.get(p, 0) / max_behavior) if max_behavior > 0 else 0.0
        score_history = (completed_counts.get(p, 0) / max_completed) if max_completed > 0 else 0.0
        score = clamp01(0.65 * float(score_behavior) + 0.35 * float(score_history))

        reason = "Recommandé"
        if score_history > 0:
            reason = f"Tu convertis bien sur {p}"
        elif score_behavior > 0.2:
            reason = f"Tu l’ouvres souvent ({p})"

        items.append({"provider": p, "score": round(float(score), 4), "reason": reason, "entry": p})

    items.sort(
        key=lambda x: (
            float(x.get("score", 0.0)),
            completed_counts.get(_safe_str(x.get("provider")).strip().lower(), 0),
            behavior_counts.get(_safe_str(x.get("provider")).strip().lower(), 0),
        ),
        reverse=True,
    )
    items = items[:limit_int]

    if not items:
        return _offerwall_provider_fallback(limit_int)
    return items


def _is_completed_status(value) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, int):
        return value == 1
    if isinstance(value, float):
        return int(value) == 1
    s = _safe_str(value).strip().lower()
    if not s:
        return False
    return s in {"completed", "approved", "paid", "success", "credited", "credit"}


def _count_iframe_conversions(user_id: str, since_history: str) -> dict[str, int]:
    counts: dict[str, int] = {"timewall": 0, "theoremreach": 0, "bitcotasks": 0}

    try:
        rows = _supabase_get_first_success(
            "transaction_theoremreach",
            variants=[
                {
                    "select": "status,created_at,reward_points",
                    "user_id": f"eq.{user_id}",
                    "created_at": f"gte.{since_history}",
                    "order": "created_at.desc",
                    "limit": "2000",
                },
                {
                    "select": "status,created_at,reward_points",
                    "user_id": f"eq.{user_id}",
                    "created_at": f"gte.{since_history}",
                    "limit": "2000",
                },
                {"select": "status", "user_id": f"eq.{user_id}", "limit": "2000"},
            ],
            timeout_s=25,
        )
        counts["theoremreach"] = sum(
            1 for r in rows if isinstance(r, dict) and _is_completed_status(r.get("status"))
        )
    except Exception:
        counts["theoremreach"] = 0

    try:
        rows = _supabase_get_first_success(
            "transaction_bitcotasks",
            variants=[
                {
                    "select": "status,created_at",
                    "user_id": f"eq.{user_id}",
                    "created_at": f"gte.{since_history}",
                    "order": "created_at.desc",
                    "limit": "2000",
                },
                {
                    "select": "status,created_at",
                    "user_id": f"eq.{user_id}",
                    "created_at": f"gte.{since_history}",
                    "limit": "2000",
                },
                {"select": "status", "user_id": f"eq.{user_id}", "limit": "2000"},
            ],
            timeout_s=25,
        )
        counts["bitcotasks"] = sum(
            1 for r in rows if isinstance(r, dict) and _is_completed_status(r.get("status"))
        )
    except Exception:
        counts["bitcotasks"] = 0

    try:
        rows = _supabase_get_first_success(
            "transaction_timewall",
            variants=[
                {
                    "select": "type,created_at,status",
                    "user_id": f"eq.{user_id}",
                    "created_at": f"gte.{since_history}",
                    "order": "created_at.desc",
                    "limit": "2000",
                },
                {
                    "select": "type,created_at",
                    "user_id": f"eq.{user_id}",
                    "created_at": f"gte.{since_history}",
                    "limit": "2000",
                },
                {"select": "type", "user_id": f"eq.{user_id}", "limit": "2000"},
            ],
            timeout_s=25,
        )
        counts["timewall"] = sum(
            1
            for r in rows
            if isinstance(r, dict)
            and (
                _is_completed_status(r.get("status"))
                or _is_completed_status(r.get("type"))
                or _safe_str(r.get("type")).strip().lower() in {"credit", "credited", "paid", "approved", "success"}
            )
        )
    except Exception:
        counts["timewall"] = 0

    if counts["timewall"] <= 0:
        try:
            rows = _supabase_get_first_success(
                "transaction_offers",
                variants=[
                    {
                        "select": "status,provider,type,created_at",
                        "user_id": f"eq.{user_id}",
                        "provider": "ilike.*timewall*",
                        "created_at": f"gte.{since_history}",
                        "order": "created_at.desc",
                        "limit": "2000",
                    },
                    {
                        "select": "status,provider,type,created_at",
                        "user_id": f"eq.{user_id}",
                        "type": "ilike.*timewall*",
                        "created_at": f"gte.{since_history}",
                        "order": "created_at.desc",
                        "limit": "2000",
                    },
                    {
                        "select": "status,provider,type",
                        "user_id": f"eq.{user_id}",
                        "provider": "ilike.*timewall*",
                        "limit": "2000",
                    },
                    {
                        "select": "status,provider,type",
                        "user_id": f"eq.{user_id}",
                        "type": "ilike.*timewall*",
                        "limit": "2000",
                    },
                ],
                timeout_s=25,
            )
            counts["timewall"] = sum(1 for r in rows if isinstance(r, dict) and safe_int(r.get("status"), -1) == 1)
        except Exception:
            counts["timewall"] = 0

    return counts


def _compute_iframe_provider_recommendations(user_id: str, limit: int = 3) -> list[dict]:
    from datetime import datetime, timedelta, timezone

    try:
        limit_int = max(1, safe_int(limit, 3))
    except Exception:
        limit_int = 3

    candidates = {"timewall", "theoremreach", "bitcotasks", "bitlabs"}
    ignored_behavior = {"page_load", "unknown"}

    now = datetime.now(timezone.utc)
    since_impressions = _to_iso(now - timedelta(days=30))
    since_history = _to_iso(now - timedelta(days=60))

    behavior_counts: dict[str, int] = {}
    iframe_generic_count = 0
    try:
        impressions = _supabase_get_first_success(
            "mobile_impressions",
            variants=[
                {
                    "select": "provider,created_at",
                    "user_id": f"eq.{user_id}",
                    "created_at": f"gte.{since_impressions}",
                    "order": "created_at.desc",
                    "limit": "2000",
                },
                {
                    "select": "provider,created_at",
                    "user_id": f"eq.{user_id}",
                    "created_at": f"gte.{since_impressions}",
                    "limit": "2000",
                },
                {"select": "provider", "user_id": f"eq.{user_id}", "limit": "2000"},
            ],
            timeout_s=25,
        )
        for row in impressions:
            if not isinstance(row, dict):
                continue
            p = normalize_provider(row.get("provider"))
            if not p:
                continue
            if p in ignored_behavior:
                continue
            if p == "iframe":
                iframe_generic_count += 1
                continue
            if p not in candidates:
                continue
            behavior_counts[p] = behavior_counts.get(p, 0) + 1
    except Exception:
        behavior_counts = {}
        iframe_generic_count = 0

    if behavior_counts.get("timewall", 0) <= 0 and iframe_generic_count > 0:
        behavior_counts["timewall"] = iframe_generic_count

    max_behavior = max(behavior_counts.values()) if behavior_counts else 0

    conv_counts: dict[str, int] = {}
    try:
        conv_counts = _count_iframe_conversions(user_id=user_id, since_history=since_history)
    except Exception:
        conv_counts = {"timewall": 0, "theoremreach": 0, "bitcotasks": 0}

    max_conv = max(conv_counts.values()) if conv_counts else 0

    bonus = 0.0
    try:
        profile_rows = _supabase_get(
            "profiles",
            params={
                "select": "profile_completion_percentage,preferred_survey_topics,max_survey_duration",
                "id": f"eq.{user_id}",
                "limit": "1",
            },
            timeout_s=20,
        )
        profile = profile_rows[0] if profile_rows else {}
        completion_pct = safe_int(profile.get("profile_completion_percentage"), 100)
        if completion_pct < 60:
            bonus += 0.05
        preferred_topics = profile.get("preferred_survey_topics")
        has_topics = False
        if isinstance(preferred_topics, list):
            has_topics = len(preferred_topics) > 0
        elif isinstance(preferred_topics, str):
            has_topics = bool(preferred_topics.strip())
        if has_topics:
            bonus += 0.03
    except Exception:
        bonus = bonus

    all_candidates = set(candidates).union(set(behavior_counts.keys())).union(set(conv_counts.keys()))
    all_candidates.discard("iframe")
    all_candidates.discard("unknown")
    all_candidates.discard("page_load")

    items: list[dict] = []
    for provider in sorted(all_candidates):
        if provider not in candidates:
            continue
        score_behavior = (behavior_counts.get(provider, 0) / max_behavior) if max_behavior > 0 else 0.0
        score_conv = (conv_counts.get(provider, 0) / max_conv) if max_conv > 0 else 0.0

        score = clamp01(0.60 * float(score_conv) + 0.35 * float(score_behavior) + float(bonus))

        reason = "Découverte"
        if score_conv > 0:
            reason = f"Tu convertis bien sur {provider} (postback)"
        elif score_behavior > 0.3:
            reason = "Tu l’ouvres souvent, bon potentiel"

        items.append({"provider": provider, "score": round(float(score), 4), "reason": reason, "entry": provider})

    items.sort(
        key=lambda x: (
            float(x.get("score", 0.0)),
            conv_counts.get(_safe_str(x.get("provider")).strip().lower(), 0),
            behavior_counts.get(_safe_str(x.get("provider")).strip().lower(), 0),
        ),
        reverse=True,
    )
    items = items[:limit_int]

    has_any_signal = bool(behavior_counts) or any((conv_counts or {}).values())
    if not has_any_signal:
        return _iframe_provider_fallback(limit_int)
    if not items:
        return _iframe_provider_fallback(limit_int)
    return items


def compute_user_mode(user_id: str) -> dict:
    from datetime import datetime, timedelta, timezone

    survey_providers = {"cpx", "rapidoreach", "theoremreach", "bitlabs", "notik"}
    ignored_providers = {"page_load", "iframe", "unknown", "adgem", "wannads"}

    now = datetime.now(timezone.utc)
    since_impressions = _to_iso(now - timedelta(days=30))
    since_history = _to_iso(now - timedelta(days=60))

    survey_impressions = 0
    try:
        impressions = _supabase_get_first_success(
            "mobile_impressions",
            variants=[
                {
                    "select": "provider,created_at",
                    "user_id": f"eq.{user_id}",
                    "created_at": f"gte.{since_impressions}",
                    "order": "created_at.desc",
                    "limit": "2000",
                },
                {
                    "select": "provider,created_at",
                    "user_id": f"eq.{user_id}",
                    "created_at": f"gte.{since_impressions}",
                    "limit": "2000",
                },
                {"select": "provider", "user_id": f"eq.{user_id}", "limit": "2000"},
            ],
            timeout_s=25,
        )
        counts: dict[str, int] = {}
        for row in impressions:
            if not isinstance(row, dict):
                continue
            p = normalize_provider(row.get("provider"))
            if not p or p in ignored_providers:
                continue
            if p in survey_providers:
                counts[p] = counts.get(p, 0) + 1
        survey_impressions = sum(counts.values())
    except Exception:
        survey_impressions = 0

    completed_surveys = 0
    try:
        tx_rows = _supabase_get_first_success(
            "transaction_sondage",
            variants=[
                {
                    "select": "status,created_at",
                    "user_id": f"eq.{user_id}",
                    "created_at": f"gte.{since_history}",
                    "order": "created_at.desc",
                    "limit": "2000",
                },
                {
                    "select": "status,created_at",
                    "user_id": f"eq.{user_id}",
                    "created_at": f"gte.{since_history}",
                    "limit": "2000",
                },
                {"select": "status", "user_id": f"eq.{user_id}", "limit": "2000"},
            ],
            timeout_s=25,
        )
        completed_surveys = sum(1 for r in tx_rows if isinstance(r, dict) and safe_int(r.get("status"), -1) == 1)
    except Exception:
        completed_surveys = 0

    clicked_offers = 0
    completed_offers = 0
    try:
        offer_rows = _supabase_get_first_success(
            "transaction_offers",
            variants=[
                {
                    "select": "status,created_at",
                    "user_id": f"eq.{user_id}",
                    "created_at": f"gte.{since_history}",
                    "order": "created_at.desc",
                    "limit": "2000",
                },
                {
                    "select": "status,created_at",
                    "user_id": f"eq.{user_id}",
                    "created_at": f"gte.{since_history}",
                    "limit": "2000",
                },
                {"select": "status", "user_id": f"eq.{user_id}", "limit": "2000"},
            ],
            timeout_s=25,
        )
        for r in offer_rows:
            if not isinstance(r, dict):
                continue
            st = safe_int(r.get("status"), -1)
            if st == 0:
                clicked_offers += 1
            elif st == 1:
                completed_offers += 1
    except Exception:
        clicked_offers = 0
        completed_offers = 0

    survey_signal_raw = safe_int(survey_impressions, 0) + 10 * safe_int(completed_surveys, 0)
    offer_signal_raw = safe_int(clicked_offers, 0) + 10 * safe_int(completed_offers, 0)
    try:
        iframe_completed = _count_iframe_conversions(user_id=user_id, since_history=since_history)
        offer_signal_raw += 10 * safe_int(sum((iframe_completed or {}).values()), 0)
    except Exception:
        offer_signal_raw = offer_signal_raw

    total = float(survey_signal_raw + offer_signal_raw)
    if total <= 0:
        survey_affinity = 0.5
        offer_affinity = 0.5
    else:
        survey_affinity = float(survey_signal_raw) / total
        offer_affinity = float(offer_signal_raw) / total

    surveys_weight = clamp01(0.2 + 0.6 * float(survey_affinity))
    offers_weight = clamp01(1.0 - float(surveys_weight))

    reason = "balanced"
    if survey_affinity >= 0.6:
        reason = "survey-heavy"
    elif survey_affinity <= 0.4:
        reason = "offer-heavy"

    return {
        "survey_affinity": round(float(survey_affinity), 4),
        "offer_affinity": round(float(offer_affinity), 4),
        "surveys_weight": round(float(surveys_weight), 4),
        "offers_weight": round(float(offers_weight), 4),
        "reason": reason,
    }


def compute_mix(user_id: str) -> dict:
    return compute_user_mode(user_id=user_id)


def recommend_survey_providers(user_id: str, limit: int = 4, mix: dict | None = None) -> list[dict]:
    from datetime import datetime, timedelta, timezone

    try:
        limit_int = max(1, safe_int(limit, 4))
    except Exception:
        limit_int = 4

    ignored_providers = {"page_load", "iframe", "unknown"}
    allowed_providers = {"cpx", "rapidoreach", "theoremreach", "bitlabs", "notik"}
    fast_providers = {"cpx", "bitlabs"}
    fallback = ["cpx", "rapidoreach", "theoremreach", "bitlabs"]

    now = datetime.now(timezone.utc)
    since_impressions = _to_iso(now - timedelta(days=30))
    since_history = _to_iso(now - timedelta(days=60))

    behavior_counts: dict[str, int] = {}
    try:
        impressions = _supabase_get_first_success(
            "mobile_impressions",
            variants=[
                {
                    "select": "provider,created_at",
                    "user_id": f"eq.{user_id}",
                    "created_at": f"gte.{since_impressions}",
                    "order": "created_at.desc",
                    "limit": "2000",
                },
                {
                    "select": "provider,created_at",
                    "user_id": f"eq.{user_id}",
                    "created_at": f"gte.{since_impressions}",
                    "limit": "2000",
                },
                {"select": "provider", "user_id": f"eq.{user_id}", "limit": "2000"},
            ],
            timeout_s=25,
        )
        for row in impressions:
            if not isinstance(row, dict):
                continue
            p = normalize_provider(row.get("provider"))
            if not p or p in ignored_providers:
                continue
            if p not in allowed_providers:
                continue
            behavior_counts[p] = behavior_counts.get(p, 0) + 1
    except Exception:
        behavior_counts = {}

    max_behavior = max(behavior_counts.values()) if behavior_counts else 0

    history_counts: dict[str, int] = {}
    try:
        tx_rows = _supabase_get_first_success(
            "transaction_sondage",
            variants=[
                {
                    "select": "provider,status,created_at",
                    "user_id": f"eq.{user_id}",
                    "created_at": f"gte.{since_history}",
                    "order": "created_at.desc",
                    "limit": "2000",
                },
                {
                    "select": "provider,status,created_at",
                    "user_id": f"eq.{user_id}",
                    "created_at": f"gte.{since_history}",
                    "limit": "2000",
                },
                {"select": "provider,status", "user_id": f"eq.{user_id}", "limit": "2000"},
                {
                    "select": "status,created_at",
                    "user_id": f"eq.{user_id}",
                    "created_at": f"gte.{since_history}",
                    "order": "created_at.desc",
                    "limit": "2000",
                },
                {"select": "status", "user_id": f"eq.{user_id}", "limit": "2000"},
            ],
            timeout_s=25,
        )
        for r in tx_rows:
            if not isinstance(r, dict):
                continue
            if safe_int(r.get("status"), -1) != 1:
                continue
            provider_raw = r.get("provider")
            p = normalize_provider(provider_raw)
            if not p or p in ignored_providers:
                continue
            if p not in allowed_providers:
                continue
            history_counts[p] = history_counts.get(p, 0) + 1
    except Exception:
        history_counts = {}

    max_history = max(history_counts.values()) if history_counts else 0

    profile: dict = {}
    try:
        profile_rows = _supabase_get(
            "profiles",
            params={
                "select": "profile_completion_percentage,max_survey_duration,preferred_survey_topics,survey_frequency,device_signup,country",
                "id": f"eq.{user_id}",
                "limit": "1",
            },
            timeout_s=20,
        )
        profile = profile_rows[0] if profile_rows else {}
    except Exception:
        profile = {}

    completion_pct = safe_int(profile.get("profile_completion_percentage"), 100)
    max_duration = safe_int(profile.get("max_survey_duration"), 9999)
    preferred_topics = profile.get("preferred_survey_topics")
    has_topics = False
    if isinstance(preferred_topics, list):
        has_topics = len(preferred_topics) > 0
    elif isinstance(preferred_topics, str):
        has_topics = bool(preferred_topics.strip())

    base_bonus = 0.0
    if completion_pct < 60:
        base_bonus += 0.08
    if has_topics:
        base_bonus += 0.05

    candidates = set(behavior_counts.keys()) | set(history_counts.keys()) | set(fallback)
    candidates = {p for p in candidates if p and p not in ignored_providers}

    has_any_signal = bool(behavior_counts) or bool(history_counts)
    if not has_any_signal:
        return _survey_provider_fallback(limit_int)

    items: list[dict] = []
    for provider in candidates:
        score_behavior = (behavior_counts.get(provider, 0) / max_behavior) if max_behavior > 0 else 0.0
        score_history = (history_counts.get(provider, 0) / max_history) if max_history > 0 else 0.0

        fast_bonus = 0.05 if (max_duration <= 10 and provider in fast_providers) else 0.0
        score = clamp01(0.75 * float(score_behavior) + 0.25 * float(score_history) + float(base_bonus) + float(fast_bonus))

        reason = "Recommandé"
        if score_history > 0:
            reason = f"Tu complètes souvent sur {provider.upper()}"
        elif score_behavior > 0.2:
            reason = f"Tu l’ouvres souvent ({provider.upper()})"
        elif fast_bonus > 0:
            reason = "Sondages rapides"
        elif base_bonus >= 0.08 and completion_pct < 60:
            reason = "Améliore ton profil avec des sondages"

        items.append({"provider": provider, "score": round(float(score), 4), "reason": reason, "entry": provider})

    items.sort(
        key=lambda x: (
            float(x.get("score", 0.0)),
            history_counts.get(_safe_str(x.get("provider")).strip().lower(), 0),
            behavior_counts.get(_safe_str(x.get("provider")).strip().lower(), 0),
        ),
        reverse=True,
    )
    items = items[:limit_int]

    if not items:
        return _survey_provider_fallback(limit_int)

    return items


def recommend_offerwall_providers(user_id: str, limit: int = 3, mix: dict | None = None) -> list[dict]:
    try:
        return _compute_offerwall_provider_recommendations(user_id=user_id, limit=limit)
    except Exception:
        return _offerwall_provider_fallback(limit)


def recommend_iframe_providers(user_id: str, limit: int = 3, mix: dict | None = None) -> list[dict]:
    try:
        return _compute_iframe_provider_recommendations(user_id=user_id, limit=limit)
    except Exception:
        return _iframe_provider_fallback(limit)


def _compute_survey_provider_recommendations(user_id: str, limit: int = 4) -> list[dict]:
    return recommend_survey_providers(user_id=user_id, limit=limit, mix=None)


def _normalize_device(value: str | None) -> str | None:
    if value is None:
        return None
    v = value.strip().lower()
    return v if v else None


def _is_device_compatible(offer_device_type: str | None, effective_device: str | None) -> bool:
    if not effective_device:
        return True

    s = (offer_device_type or "").strip().lower()
    if not s:
        return True

    if "all" in s or "any" in s or "mobile" in s:
        return True

    if "android" in s and effective_device == "android":
        return True

    if "ios" in s and effective_device == "ios":
        return True

    if "," in s:
        parts = [p.strip() for p in s.split(",") if p.strip()]
        if effective_device in parts:
            return True

    return s == effective_device


def _bearer_token(request: Request) -> str:
    auth = request.headers.get("authorization")
    if not auth:
        raise HTTPException(status_code=401, detail="Authorization manquante")
    parts = auth.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(status_code=401, detail="Authorization invalide")
    token = parts[1].strip()
    if not token:
        raise HTTPException(status_code=401, detail="Token manquant")
    return token


def _require_admin(request: Request) -> str:
    token = _bearer_token(request)
    supabase_url = os.environ.get("SUPABASE_URL")
    if not supabase_url:
        raise HTTPException(status_code=500, detail="SUPABASE_URL manquant côté serveur")

    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ.get("SUPABASE_ANON_KEY")
    if not key:
        raise HTTPException(
            status_code=500,
            detail="SUPABASE_SERVICE_ROLE_KEY ou SUPABASE_ANON_KEY manquant côté serveur",
        )

    auth_resp = requests.get(
        f"{supabase_url.rstrip('/')}/auth/v1/user",
        headers={"apikey": key, "Authorization": f"Bearer {token}"},
        timeout=10,
    )
    if auth_resp.status_code != 200:
        raise HTTPException(status_code=401, detail="Session invalide ou expirée")
    user = auth_resp.json() or {}
    user_id = user.get("id")
    if not user_id:
        raise HTTPException(status_code=401, detail="Session invalide (id absent)")

    try:
        rows = _supabase_get(
            "profiles",
            params={"select": "role", "id": f"eq.{user_id}", "limit": "1"},
        )
    except Exception:
        raise HTTPException(status_code=403, detail="Accès admin refusé")

    role = _safe_str((rows[0] if rows else {}).get("role")).strip().lower()
    if role != "admin":
        raise HTTPException(status_code=403, detail="Accès admin refusé")

    return user_id


@asynccontextmanager
async def lifespan(app: FastAPI):
    resources.clear()
    try:
        resources["supabase_rest_url"] = _supabase_rest_url()
        resources["supabase_headers"] = _supabase_headers()
    except Exception as e:
        resources["startup_error"] = str(e)
        yield
        resources.clear()
        return

    yield
    resources.clear()


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/llm-test")
def llm_test():
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3.1:8b",
                "prompt": "Réponds uniquement: OK",
                "stream": False,
                "keep_alive": "10m",
                "options": {
                    "num_predict": 20
                }
            },
            timeout=180
        )
        response.raise_for_status()
        data = response.json()
        return {"answer": data.get("response", "")}
    except requests.Timeout:
        raise HTTPException(status_code=504, detail="Ollama request timed out")
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error contacting Ollama: {str(e)}")


def _fetch_all_ptc_offers(batch_size: int = 1000) -> list[dict]:
    rest_url = resources["supabase_rest_url"]
    headers = resources["supabase_headers"]

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
            raise RuntimeError(f"Erreur Supabase ptc_offers: HTTP {resp.status_code} - {resp.text}")
        batch = resp.json() or []
        offers.extend(batch)
        if len(batch) < batch_size:
            break
        offset += batch_size

    return offers


def _rebuild_index_from_supabase() -> int:
    offers_raw = _fetch_all_ptc_offers()
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

    model = resources.get("model") or SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts)
    embeddings = np.asarray(embeddings, dtype=np.float32)
    faiss.normalize_L2(embeddings)

    d = int(embeddings.shape[1])
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)

    index_path = BASE_DIR / "offers.faiss"
    meta_path = BASE_DIR / "offers_meta.pkl"

    faiss.write_index(index, str(index_path))
    with open(meta_path, "wb") as f:
        pickle.dump(meta, f)

    points_list = [o.get("points", 0) for o in meta if isinstance(o, dict)]
    max_points = max(points_list) if points_list else 1

    resources["index"] = index
    resources["offers_meta"] = meta
    resources["model"] = model
    resources["max_points"] = max_points if max_points > 0 else 1
    resources.pop("startup_error", None)

    return len(meta)


@app.get("/admin/monitoring")
def admin_monitoring(request: Request):
    _require_admin(request)

    index_path = BASE_DIR / "offers.faiss"
    meta_path = BASE_DIR / "offers_meta.pkl"

    index_loaded = "index" in resources
    meta_loaded = "offers_meta" in resources
    model_loaded = "model" in resources
    index_ntotal = int(resources["index"].ntotal) if index_loaded else 0
    meta_count = len(resources.get("offers_meta", []) or []) if meta_loaded else 0

    return {
        "status": "ok",
        "startup_error": resources.get("startup_error"),
        "index_file_exists": index_path.exists(),
        "meta_file_exists": meta_path.exists(),
        "index_loaded": index_loaded,
        "meta_loaded": meta_loaded,
        "model_loaded": model_loaded,
        "index_ntotal": index_ntotal,
        "meta_count": meta_count,
    }


@app.post("/admin/reindex")
def admin_reindex(request: Request):
    _require_admin(request)
    try:
        count = _rebuild_index_from_supabase()
        return {"status": "ok", "message": f"OK index offers créé: {count}", "count": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def recommend_offers_internal(
    user_id: str,
    limit: int,
    country: str | None = None,
    device: str | None = None,
    mix: dict | None = None,
) -> list[dict]:
    return []


@app.get("/recommendations/{user_id}")
def get_recommendations(
    user_id: str,
    limit: int = 10,
    country: str | None = None,
    device: str | None = None,
):
    startup_error = resources.get("startup_error")
    if startup_error:
        raise HTTPException(status_code=500, detail=startup_error)

    try:
        mix = compute_mix(user_id=user_id)
    except Exception:
        mix = {
            "survey_affinity": 0.5,
            "offer_affinity": 0.5,
            "surveys_weight": 0.5,
            "offers_weight": 0.5,
            "reason": "balanced",
        }

    try:
        limit_int = max(1, safe_int(limit, 10))
    except Exception:
        limit_int = 10

    surveys_weight = float(mix.get("surveys_weight", 0.5))
    offers_weight = float(mix.get("offers_weight", 0.5))

    offers_limit = int(round(float(limit_int) * float(offers_weight)))
    if offers_limit < 3:
        offers_limit = 3

    surveys_limit = int(round(4.0 * float(surveys_weight)))
    if surveys_limit < 1:
        surveys_limit = 1
    if surveys_limit > 4:
        surveys_limit = 4

    try:
        items = recommend_offers_internal(
            user_id=user_id,
            limit=offers_limit,
            device=device,
            country=country,
            mix=mix,
        )
    except Exception:
        items = []

    try:
        surveys = recommend_survey_providers(user_id=user_id, limit=surveys_limit, mix=mix)
    except Exception:
        surveys = _survey_provider_fallback(surveys_limit)

    try:
        iframes = recommend_iframe_providers(user_id=user_id, limit=3, mix=mix)
    except Exception:
        iframes = _iframe_provider_fallback(3)

    return {"user_id": user_id, "items": items, "surveys": surveys, "iframes": iframes, "mix": mix}


# NEW primary endpoint: /personalization
@app.get("/personalization/{user_id}")
def get_personalization(
    user_id: str,
    limit: int = 10,
    country: str | None = None,
    device: str | None = None,
):
    startup_error = resources.get("startup_error")

    try:
        mix = compute_mix(user_id=user_id)
    except Exception:
        mix = {
            "survey_affinity": 0.5,
            "offer_affinity": 0.5,
            "surveys_weight": 0.5,
            "offers_weight": 0.5,
            "reason": "balanced",
        }

    try:
        limit_int = max(1, safe_int(limit, 10))
    except Exception:
        limit_int = 10

    offers_weight = float(mix.get("offers_weight", 0.5))
    surveys_weight = float(mix.get("surveys_weight", 0.5))

    offers_limit = int(round(float(limit_int) * float(offers_weight)))
    if offers_limit < 3:
        offers_limit = 3

    surveys_limit = int(round(4.0 * float(surveys_weight)))
    if surveys_limit < 2:
        surveys_limit = 2

    offerwalls_limit = 3
    iframes_limit = 3

    offers: list[dict] = []
    if (not startup_error) and ("index" in resources) and ("offers_meta" in resources):
        try:
            offers = recommend_offers_internal(
                user_id=user_id,
                limit=offers_limit,
                device=device,
                country=country,
                mix=mix,
            )
        except Exception:
            offers = []

    try:
        surveys = recommend_survey_providers(user_id=user_id, limit=surveys_limit, mix=mix)
    except Exception:
        surveys = _survey_provider_fallback(surveys_limit)

    try:
        offerwalls = recommend_offerwall_providers(user_id=user_id, limit=offerwalls_limit, mix=mix)
    except Exception:
        offerwalls = _offerwall_provider_fallback(offerwalls_limit)

    try:
        iframes = recommend_iframe_providers(user_id=user_id, limit=iframes_limit, mix=mix)
    except Exception:
        iframes = _iframe_provider_fallback(iframes_limit)

    return {
        "user_id": user_id,
        "mix": mix,
        "providers": {
            "surveys": surveys,
            "offerwalls": offerwalls,
            "iframes": iframes,
        },
        "offers": offers,
    }


@app.get("/survey-recommendations/{user_id}")
def get_survey_recommendations(user_id: str, limit: int = 4):
    try:
        items = _compute_survey_provider_recommendations(user_id=user_id, limit=limit)
    except Exception:
        items = _survey_provider_fallback(limit)
    return {"user_id": user_id, "items": items}


@app.get("/iframe-recommendations/{user_id}")
def get_iframe_recommendations(user_id: str, limit: int = 3):
    try:
        items = _compute_iframe_provider_recommendations(user_id=user_id, limit=limit)
    except Exception:
        items = _iframe_provider_fallback(limit)
    return {"user_id": user_id, "items": items}


@app.get("/offerwall-provider-recommendations/{user_id}")
def get_offerwall_provider_recommendations(user_id: str, limit: int = 3):
    try:
        items = _compute_offerwall_provider_recommendations(user_id=user_id, limit=limit)
    except Exception:
        items = _offerwall_provider_fallback(limit)
    return {"user_id": user_id, "items": items}


@app.post("/admin/quiz/generate")
async def admin_quiz_generate(request: Request):
    _require_admin(request)

    try:
        body = await request.json()
    except Exception:
        body = {}

    topic = body.get("topic") if isinstance(body, dict) else None
    difficulty = body.get("difficulty") if isinstance(body, dict) else None

    try:
        llm = _generate_quiz_llm(topic=topic, difficulty=difficulty)
    except Exception:
        llm = None

    if llm:
        return llm

    return _fallback_quiz(topic=topic, difficulty=difficulty)

