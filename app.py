from dotenv import load_dotenv
load_dotenv()

import os
import pickle
import json
import random
import re
import uuid
import hashlib
from datetime import datetime, timedelta, timezone
from contextlib import asynccontextmanager
from pathlib import Path

import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

BASE_DIR = Path(__file__).resolve().parent
resources: dict = {}
SERVICE_VERSION = "2026-02-18-quiz-llm-health-v2"








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

def _supabase_post(table: str, payload, timeout_s: int = 20):
    rest_url = resources["supabase_rest_url"]
    headers = dict(resources["supabase_headers"])
    headers["Content-Type"] = "application/json"
    headers["Prefer"] = "return=representation"
    resp = requests.post(
        f"{rest_url}/{table}",
        json=payload,
        headers=headers,
        timeout=timeout_s,
    )
    if resp.status_code not in (200, 201):
        raise RuntimeError(f"Erreur Supabase POST {table}: HTTP {resp.status_code} - {resp.text}")
    return resp.json() or []

def _supabase_patch(table: str, match: dict, payload: dict, timeout_s: int = 20) -> list[dict]:
    rest_url = resources["supabase_rest_url"]
    headers = dict(resources["supabase_headers"])
    headers["Content-Type"] = "application/json"
    headers["Prefer"] = "return=representation"
    resp = requests.patch(
        f"{rest_url}/{table}",
        params=match,
        json=payload,
        headers=headers,
        timeout=timeout_s,
    )
    if resp.status_code not in (200, 201, 204):
        raise RuntimeError(f"Erreur Supabase PATCH {table}: HTTP {resp.status_code} - {resp.text}")
    if resp.status_code == 204:
        return []
    try:
        return resp.json() or []
    except Exception:
        return []

def _supabase_rpc(func: str, body: dict, timeout_s: int = 20) -> dict:
    rest_url = resources["supabase_rest_url"]
    headers = resources["supabase_headers"]
    resp = requests.post(
        f"{rest_url}/rpc/{func}",
        json=body,
        headers=headers,
        timeout=timeout_s,
    )
    if resp.status_code not in (200, 201, 204):
        raise RuntimeError(f"RPC {func}: HTTP {resp.status_code} - {resp.text}")
    if resp.status_code == 204:
        return {}
    try:
        return resp.json() or {}
    except Exception:
        return {}

def _support_should_request_human(message: str) -> bool:
    m = _safe_str(message).strip().lower()
    if not m:
        return False
    triggers = [
        "parler a un humain",
        "parler à un humain",
        "support humain",
        "agent humain",
        "un humain",
        "un agent",
        "ouvrir un ticket",
        "ouvrir ticket",
        "ticket support",
        "contact humain",
    ]
    return any(t in m for t in triggers)

def _support_is_yes(message: str) -> bool:
    m = re.sub(r"[^a-z0-9àâçéèêëîïôûùüÿñæœ\s'-]+", " ", _safe_str(message).strip().lower())
    m = re.sub(r"\s+", " ", m).strip()
    if not m:
        return False
    yes_set = {
        "oui",
        "ok",
        "okay",
        "daccord",
        "d'accord",
        "yes",
        "yep",
        "vas y",
        "vas-y",
        "go",
    }
    if m in yes_set:
        return True
    if m.startswith("oui "):
        return True
    return False

def _support_last_bot_asked_human_session(session_id: str) -> bool:
    sid = _safe_str(session_id).strip()
    if not sid:
        return False
    try:
        rows = _supabase_get(
            "support_chat_session_messages",
            params={
                "select": "sender,message,created_at",
                "session_id": f"eq.{sid}",
                "order": "created_at.desc",
                "limit": "12",
            },
            timeout_s=5,
        )
        for r in rows:
            if not isinstance(r, dict):
                continue
            if _safe_str(r.get("sender")).lower() != "bot":
                continue
            txt = _safe_str(r.get("message")).lower()
            if "agent" in txt and "humain" in txt:
                return True
            if "parler" in txt and "humain" in txt:
                return True
            if "mettre" in txt and "relation" in txt and "humain" in txt:
                return True
        return False
    except Exception:
        return False

def _support_get_session_history(session_id: str, limit: int = 10) -> tuple[str, int]:
    sid = _safe_str(session_id).strip()
    if not sid:
        return "", 0
    try:
        rows = _supabase_get(
            "support_chat_session_messages",
            params={
                "select": "sender,message,created_at",
                "session_id": f"eq.{sid}",
                "order": "created_at.asc",
                "limit": str(max(1, min(50, limit))),
            },
            timeout_s=5,
        )
        lines: list[str] = []
        user_count = 0
        for r in rows:
            if not isinstance(r, dict):
                continue
            sender = _safe_str(r.get("sender")).lower()
            msg = _safe_str(r.get("message")).strip()
            if not msg:
                continue
            if sender == "user":
                user_count += 1
                lines.append(f"Utilisateur: {msg}")
            elif sender == "bot":
                lines.append(f"Assistant: {msg}")
        return "\n".join(lines), user_count
    except Exception:
        return "", 0

def _support_user_seems_blocked(message: str) -> bool:
    m = re.sub(r"\s+", " ", _safe_str(message).lower()).strip()
    if not m:
        return False
    triggers = [
        "bloqué",
        "bloque",
        "ça marche pas",
        "ca marche pas",
        "ne marche pas",
        "ne fonctionne pas",
        "j'ai tout essay",
        "jai tout essay",
        "rien ne marche",
        "rien ne fonctionne",
        "toujours pareil",
        "impossible",
        "bug",
        "erreur",
        "ça bug",
        "ca bug",
        "rien de cela",
    ]
    return any(t in m for t in triggers)

def _support_bot_offered_human_recently(text: str) -> bool:
    s = _safe_str(text).lower()
    return (
        ("agent humain" in s)
        or ("parler à un humain" in s)
        or ("parler a un humain" in s)
        or ("mettre en relation" in s)
        or ("human agent" in s)
        or ("talk to a human" in s)
        or ("human support" in s)
        or ("agente humano" in s)
        or ("hablar con un humano" in s)
        or ("mit einem mensch" in s)
        or ("menschlichen" in s and "agent" in s)
        or ("parlare con un umano" in s)
        or ("menselijke" in s and "agent" in s)
        or ("souhaitez-vous" in s and "humain" in s)
    )

def _support_lang_from_locale(locale: str) -> str:
    loc = _safe_str(locale).strip().lower()
    if not loc:
        return ""
    primary = re.split(r"[_-]", loc)[0].strip()
    if primary in {"fr", "en", "es", "de", "it", "nl", "pt"}:
        return primary
    return ""

def _support_detect_user_language(text: str, locale: str | None = None) -> str:
    hint = _support_lang_from_locale(_safe_str(locale))
    s = re.sub(r"\s+", " ", _safe_str(text).lower()).strip()
    if not s:
        return hint or "fr"

    scores = {k: 0 for k in ["fr", "en", "es", "de", "it", "nl", "pt"]}
    if hint in scores:
        scores[hint] += 2

    fr_words = ["bonjour", "merci", "mot de passe", "compte", "je", "j'ai", "ça", "probleme", "problème", "aide", "connexion"]
    en_words = ["hello", "thanks", "password", "account", "i", "i'm", "can't", "cannot", "help", "login", "sign in"]
    es_words = ["hola", "gracias", "contraseña", "cuenta", "no puedo", "ayuda", "iniciar sesión", "iniciar sesion"]
    de_words = ["hallo", "danke", "passwort", "konto", "ich", "hilfe", "anmelden", "einloggen"]
    it_words = ["ciao", "grazie", "password", "account", "aiuto", "accesso", "accedere"]
    nl_words = ["hallo", "dank", "wachtwoord", "account", "help", "inloggen"]
    pt_words = ["olá", "ola", "obrigado", "senha", "conta", "ajuda", "entrar", "login"]

    def bump(lang: str, words: list[str]) -> None:
        for w in words:
            if w in s:
                scores[lang] += 1

    bump("fr", fr_words)
    bump("en", en_words)
    bump("es", es_words)
    bump("de", de_words)
    bump("it", it_words)
    bump("nl", nl_words)
    bump("pt", pt_words)

    best = max(scores.items(), key=lambda kv: kv[1])[0]
    if scores[best] == 0:
        return hint or "fr"
    return best

def _support_i18n(lang: str, key: str) -> str:
    l = _safe_str(lang).strip().lower() or "fr"
    t = {
        "offer_human": {
            "fr": "Je peux vous mettre en relation avec un agent humain. Souhaitez-vous ?",
            "en": "I can connect you with a human agent. Would you like that?",
            "es": "Puedo ponerte en contacto con un agente humano. ¿Quieres que lo haga?",
            "de": "Ich kann Sie mit einem menschlichen Support-Agenten verbinden. Möchten Sie das?",
            "it": "Posso metterti in contatto con un agente umano. Vuoi che lo faccia?",
            "nl": "Ik kan je in contact brengen met een menselijke agent. Wil je dat?",
            "pt": "Posso colocar você em contato com um atendente humano. Você quer?",
        },
        "human_opened": {
            "fr": "Merci, je vous mets en relation avec un agent humain. Vous recevrez une réponse ici dès que possible.",
            "en": "Thanks — I’m connecting you with a human agent. You’ll get a reply here as soon as possible.",
            "es": "Gracias — te pongo en contacto con un agente humano. Recibirás una respuesta aquí lo antes posible.",
            "de": "Danke — ich verbinde Sie mit einem menschlichen Agenten. Sie erhalten hier so schnell wie möglich eine Antwort.",
            "it": "Grazie — ti metto in contatto con un agente umano. Riceverai una risposta qui il prima possibile.",
            "nl": "Bedankt — ik breng je in contact met een menselijke agent. Je krijgt hier zo snel mogelijk antwoord.",
            "pt": "Obrigado — vou te colocar em contato com um atendente humano. Você receberá uma resposta aqui o mais rápido possível.",
        },
        "human_unavailable": {
            "fr": "Je n’arrive pas à vous mettre en relation avec un agent humain pour le moment. Réessayez plus tard.",
            "en": "I can’t connect you with a human agent right now. Please try again later.",
            "es": "Ahora mismo no puedo ponerte en contacto con un agente humano. Inténtalo más tarde.",
            "de": "Ich kann Sie im Moment nicht mit einem menschlichen Agenten verbinden. Bitte versuchen Sie es später erneut.",
            "it": "Al momento non riesco a metterti in contatto con un agente umano. Riprova più tardi.",
            "nl": "Ik kan je op dit moment niet met een menselijke agent verbinden. Probeer het later opnieuw.",
            "pt": "No momento não consigo te colocar em contato com um atendente humano. Tente novamente mais tarde.",
        },
        "ticket_limit": {
            "fr": "Vous avez déjà 3 demandes de support en attente. Merci d’attendre une réponse avant d’en ouvrir une nouvelle.",
            "en": "You already have 3 open support requests. Please wait for a reply before opening a new one.",
            "es": "Ya tienes 3 solicitudes de soporte abiertas. Espera una respuesta antes de abrir otra.",
            "de": "Sie haben bereits 3 offene Support-Anfragen. Bitte warten Sie auf eine Antwort, bevor Sie eine neue eröffnen.",
            "it": "Hai già 3 richieste di supporto aperte. Attendi una risposta prima di aprirne un’altra.",
            "nl": "Je hebt al 3 open supportverzoeken. Wacht op een antwoord voordat je een nieuwe opent.",
            "pt": "Você já tem 3 solicitações de suporte abertas. Aguarde uma resposta antes de abrir uma nova.",
        },
        "image_received": {
            "fr": "J’ai bien reçu l’image. Je ne peux pas l’analyser automatiquement ici. Si vous voulez, je peux vous mettre en relation avec un agent humain.",
            "en": "I received the image. I can’t automatically analyze it here. If you want, I can connect you with a human agent.",
            "es": "He recibido la imagen. No puedo analizarla automáticamente aquí. Si quieres, puedo ponerte en contacto con un agente humano.",
            "de": "Ich habe das Bild erhalten. Ich kann es hier nicht automatisch analysieren. Wenn Sie möchten, kann ich Sie mit einem menschlichen Agenten verbinden.",
            "it": "Ho ricevuto l’immagine. Non posso analizzarla automaticamente qui. Se vuoi, posso metterti in contatto con un agente umano.",
            "nl": "Ik heb de afbeelding ontvangen. Ik kan die hier niet automatisch analyseren. Als je wilt, kan ik je met een menselijke agent verbinden.",
            "pt": "Recebi a imagem. Não consigo analisá-la automaticamente aqui. Se você quiser, posso te colocar em contato com um atendente humano.",
        },
    }
    return (t.get(key) or {}).get(l) or (t.get(key) or {}).get("fr") or ""

def _support_last_bot_asked_human(ticket_id: str) -> bool:
    try:
        rows = _supabase_get(
            "support_chat_ticket_messages",
            params={
                "select": "sender,message,created_at",
                "ticket_id": f"eq.{ticket_id}",
                "order": "created_at.desc",
                "limit": "10",
            },
            timeout_s=5,
        )
        for r in rows:
            if not isinstance(r, dict):
                continue
            if _safe_str(r.get("sender")).lower() != "bot":
                continue
            txt = _safe_str(r.get("message")).lower()
            if "parler" in txt and "humain" in txt:
                return True
            if "ouvrir" in txt and "ticket" in txt:
                return True
        return False
    except Exception:
        return False

def _support_infer_ticket_subject(text: str) -> str:
    t = _safe_str(text).lower()
    t = re.sub(r"\s+", " ", t).strip()
    if not t:
        return "Support"

    def has(*words: str) -> bool:
        return any(w in t for w in words)

    if has("mot de passe", "password", "mdp", "connexion", "login", "se connecter", "compte", "email", "e-mail", "verification", "vérification", "2fa", "auth"):
        if has("mot de passe", "password", "mdp"):
            return "Compte - Mot de passe"
        if has("connexion", "login", "se connecter"):
            return "Compte - Connexion"
        return "Compte"

    if has("paypal", "virement", "retrait", "withdraw", "paiement", "payment"):
        return "Paiement - Retrait"

    if has("carte cadeau", "giftcard", "gift card", "code", "commande", "order", "livraison"):
        return "Boutique - Commande"

    if has("sondage", "survey", "cpx", "rapido", "rapidoreach", "theoremreach"):
        return "Sondages - Crédit manquant"

    if has("offre", "offer", "timewall", "ogads", "notik", "offery", "bitcotasks", "conversion", "postback", "tracking"):
        return "Offres - Crédit manquant"

    if has("points", "crédit", "credit", "manquant", "non reçu", "pas reçu", "en attente", "pending", "refus", "refusé", "rejet", "rejeté"):
        return "Récompenses - Problème de crédit"

    if has("bug", "erreur", "crash", "bloque", "ne marche pas", "ne fonctionne pas", "chargement", "page blanche"):
        return "Technique - Bug"

    if has("ban", "banni", "bannie", "mute", "suspendu", "suspension", "bloqué", "bloquee", "bloqué"):
        return "Compte - Restriction"

    return "Support"

def _support_suggest_subject_from_ticket(ticket_id: str, latest_message: str) -> str:
    tid = _safe_str(ticket_id).strip()
    if not tid:
        return _support_infer_ticket_subject(latest_message)
    try:
        rows = _supabase_get(
            "support_chat_ticket_messages",
            params={
                "select": "message,sender,created_at",
                "ticket_id": f"eq.{tid}",
                "order": "created_at.desc",
                "limit": "20",
            },
            timeout_s=5,
        )
        user_msgs: list[str] = []
        for r in rows:
            if not isinstance(r, dict):
                continue
            if _safe_str(r.get("sender")).lower() != "user":
                continue
            msg = _safe_str(r.get("message")).strip()
            if msg:
                user_msgs.append(msg)
            if len(user_msgs) >= 6:
                break
        combined = " | ".join(reversed(user_msgs)) if user_msgs else _safe_str(latest_message)
        return _support_infer_ticket_subject(combined)
    except Exception:
        return _support_infer_ticket_subject(latest_message)

def _support_get_ticket_state(ticket_id: str, user_id: str) -> dict | None:
    tid = _safe_str(ticket_id).strip()
    uid = _safe_str(user_id).strip()
    if not tid or not uid:
        return None
    try:
        rows = _supabase_get(
            "support_chat_tickets",
            params={
                "select": "id,user_id,needs_human,status",
                "id": f"eq.{tid}",
                "user_id": f"eq.{uid}",
                "limit": "1",
            },
            timeout_s=5,
        )
        if rows and isinstance(rows[0], dict):
            return rows[0]
    except Exception:
        return None
    return None

def _support_get_or_create_ticket(user_id: str, message: str, ticket_id: str | None) -> tuple[str | None, bool]:
    uid = _safe_str(user_id).strip()
    if not uid or uid == "anonymous":
        return None, False

    provided = _safe_str(ticket_id).strip()
    if provided:
        try:
            rows = _supabase_get(
                "support_chat_tickets",
                params={
                    "select": "id,user_id",
                    "id": f"eq.{provided}",
                    "user_id": f"eq.{uid}",
                    "limit": "1",
                },
                timeout_s=5,
            )
            if rows and isinstance(rows[0], dict) and _safe_str(rows[0].get("id")):
                return _safe_str(rows[0].get("id")), False
        except Exception:
            pass

    try:
        existing = _supabase_get(
            "support_chat_tickets",
            params={
                "select": "id,updated_at,status",
                "user_id": f"eq.{uid}",
                "needs_human": "eq.false",
                "status": "in.(ouvert,en_cours)",
                "order": "updated_at.desc",
                "limit": "1",
            },
            timeout_s=5,
        )
        if existing and isinstance(existing[0], dict) and _safe_str(existing[0].get("id")):
            return _safe_str(existing[0].get("id")), False
    except Exception:
        pass

    created = _supabase_rpc(
        "create_support_chat_ticket",
        {
            "p_user_id": uid,
            "p_subject": "Support",
            "p_first_message": _safe_str(message),
        },
        timeout_s=5,
    )
    if isinstance(created, str) and created:
        return created, True
    if isinstance(created, dict) and created.get("id"):
        return _safe_str(created.get("id")), True
    if isinstance(created, dict) and created.get("ticket_id"):
        return _safe_str(created.get("ticket_id")), True
    return None, False

def _support_get_or_create_session(user_id: str, session_id: str | None) -> str | None:
    uid = _safe_str(user_id).strip()
    if not uid or uid == "anonymous":
        return None

    provided = _safe_str(session_id).strip()
    if provided:
        try:
            rows = _supabase_get(
                "support_chat_sessions",
                params={
                    "select": "id,user_id,status",
                    "id": f"eq.{provided}",
                    "user_id": f"eq.{uid}",
                    "limit": "1",
                },
                timeout_s=5,
            )
            if rows and isinstance(rows[0], dict) and _safe_str(rows[0].get("id")):
                if _safe_str(rows[0].get("status")) == "closed":
                    return None
                return _safe_str(rows[0].get("id"))
        except Exception:
            pass

    try:
        existing = _supabase_get(
            "support_chat_sessions",
            params={
                "select": "id,updated_at,status",
                "user_id": f"eq.{uid}",
                "status": "eq.open",
                "order": "updated_at.desc",
                "limit": "1",
            },
            timeout_s=5,
        )
        if existing and isinstance(existing[0], dict) and _safe_str(existing[0].get("id")):
            return _safe_str(existing[0].get("id"))
    except Exception:
        pass

    created = _supabase_rpc(
        "create_support_chat_session",
        {"p_user_id": uid},
        timeout_s=5,
    )
    if isinstance(created, str) and created:
        return created
    if isinstance(created, dict) and created.get("id"):
        return _safe_str(created.get("id"))
    if isinstance(created, dict) and created.get("session_id"):
        return _safe_str(created.get("session_id"))
    return None

def _support_open_human_ticket_count(user_id: str) -> int:
    uid = _safe_str(user_id).strip()
    if not uid or uid == "anonymous":
        return 0
    try:
        rows = _supabase_get(
            "support_chat_tickets",
            params={
                "select": "id",
                "user_id": f"eq.{uid}",
                "needs_human": "eq.true",
                "status": "in.(ouvert,en_cours)",
            },
            timeout_s=5,
        )
        return len(rows or [])
    except Exception:
        return 0

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


def _require_internal_token(request: Request) -> None:
    expected = _safe_str(os.environ.get("IA_INTERNAL_TOKEN")).strip()
    if not expected:
        return
    provided = _safe_str(request.headers.get("x-internal-token") or request.headers.get("X-Internal-Token")).strip()
    if not provided or provided != expected:
        raise HTTPException(status_code=401, detail="Unauthorized")


def _llm_status() -> dict:
    ollama_base_url = _safe_str(os.environ.get("OLLAMA_BASE_URL")).strip().rstrip("/")
    ollama_model = _safe_str(os.environ.get("OLLAMA_MODEL")).strip() or "llama3.1:8b"
    hf_key = _safe_str(os.environ.get("HUGGINGFACE_API_KEY")).strip()
    hf_model = _safe_str(os.environ.get("HUGGINGFACE_MODEL")).strip() or "HuggingFaceTB/SmolLM3-3B:hf-inference"
    return {
        "ollama_configured": bool(ollama_base_url),
        "ollama_base_url": ollama_base_url,
        "ollama_model": ollama_model,
        "huggingface_configured": bool(hf_key),
        "huggingface_api_key_length": len(hf_key) if hf_key else 0,
        "huggingface_model": hf_model,
    }

def _is_translate_strict() -> bool:
    v = _safe_str(os.environ.get("TRANSLATE_STRICT")).strip().lower()
    return v in {"1", "true", "yes", "on"}

def _translation_cache_path() -> Path:
    return BASE_DIR / "translations_cache.json"

def _load_translation_cache() -> dict:
    try:
        p = _translation_cache_path()
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f) or {}
                if isinstance(data, dict):
                    return data
        return {}
    except Exception:
        return {}

def _save_translation_cache(cache: dict) -> None:
    try:
        p = _translation_cache_path()
        with open(p, "w", encoding="utf-8") as f:
            json.dump(cache or {}, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def _cache_key_for_translation(text: str, target_lang: str) -> str:
    s = _safe_str(text).strip()
    tl = _safe_str(target_lang).strip().lower() or "fr"
    h = hashlib.sha256(s.encode("utf-8")).hexdigest()
    return f"{tl}:{h}"

def _get_cached_translation(text: str, target_lang: str) -> str | None:
    try:
        cache = resources.get("translation_cache") or {}
        key = _cache_key_for_translation(text, target_lang)
        val = cache.get(key)
        return _safe_str(val).strip() if val else None
    except Exception:
        return None

def _set_cached_translation(text: str, target_lang: str, translated: str) -> None:
    try:
        cache = resources.setdefault("translation_cache", {})
        key = _cache_key_for_translation(text, target_lang)
        cache[key] = _safe_str(translated).strip()
        _save_translation_cache(cache)
    except Exception:
        pass


def _normalize_for_translation(text: str) -> str:
    s = _safe_str(text)
    if not s:
        return ""
    raw = s.strip()
    # Convert snake_case identifiers to words
    words = re.sub(r"_+", " ", raw).strip()
    # Remove common provider/game prefixes
    words = re.sub(r"^(se|sr|cg|bp|memo|gratto|shooter|blockpuzzle|wf|wild\s*fish)\s+", "", words, flags=re.IGNORECASE)
    # Specific event patterns -> concise English phrases
    m = re.search(r"player\s+reached\s+(\d+)\s+reachlevel", words, flags=re.IGNORECASE)
    if m:
        n = m.group(1)
        return f"Reach level {n}"
    m = re.search(r"player\s+reached\s+(\d+)\s+distancelevel", words, flags=re.IGNORECASE)
    if m:
        n = m.group(1)
        return f"Reach distance level {n}"
    m = re.search(r"player\s+reached\s+(\d+)\s+linelevel", words, flags=re.IGNORECASE)
    if m:
        n = m.group(1)
        return f"Reach line level {n}"
    m = re.search(r"reached\s+(\d+)\s+reachlevel", words, flags=re.IGNORECASE)
    if m:
        n = m.group(1)
        return f"Reach level {n}"
    m = re.search(r"reached\s+(\d+)\s+distancelevel", words, flags=re.IGNORECASE)
    if m:
        n = m.group(1)
        return f"Reach distance level {n}"
    m = re.search(r"reached\s+(\d+)\s+linelevel", words, flags=re.IGNORECASE)
    if m:
        n = m.group(1)
        return f"Reach line level {n}"
    # Fallback: return cleaned words
    return words

def _deterministic_translate(prepared_text: str, target_lang: str) -> str | None:
    tl = (_safe_str(target_lang).strip().lower() or "fr")
    s = _safe_str(prepared_text).strip()
    if not s:
        return None
    # Patterns: "Reach level N", "Reach distance level N", "Reach line level N"
    m = re.match(r"^reach\s+(level|distance level|line level)\s+(\d+)$", s, flags=re.IGNORECASE)
    if not m:
        return None
    kind = m.group(1).lower()
    n = m.group(2)
    templates = {
        "fr": {
            "level": "Atteindre le niveau {n}",
            "distance level": "Atteindre le niveau de distance {n}",
            "line level": "Atteindre le niveau de ligne {n}",
        },
        "de": {
            "level": "Erreiche Level {n}",
            "distance level": "Erreiche Distanzlevel {n}",
            "line level": "Erreiche Linienlevel {n}",
        },
        "es": {
            "level": "Alcanzar nivel {n}",
            "distance level": "Alcanzar nivel de distancia {n}",
            "line level": "Alcanzar nivel de línea {n}",
        },
        "it": {
            "level": "Raggiungi il livello {n}",
            "distance level": "Raggiungi il livello di distanza {n}",
            "line level": "Raggiungi il livello di linea {n}",
        },
        "nl": {
            "level": "Bereik level {n}",
            "distance level": "Bereik afstandsniveau {n}",
            "line level": "Bereik lijnniveau {n}",
        },
        "en": {
            "level": "Reach level {n}",
            "distance level": "Reach distance level {n}",
            "line level": "Reach line level {n}",
        },
    }
    lang_templates = templates.get(tl) or templates["en"]
    template = lang_templates.get(kind) or lang_templates["level"]
    return template.format(n=n)

def _ollama_chat(system_prompt: str, user_prompt: str, max_tokens: int, temperature: float) -> dict | None:
    base_url = _safe_str(os.environ.get("OLLAMA_BASE_URL")).strip().rstrip("/")
    if not base_url:
        return None

    model = _safe_str(os.environ.get("OLLAMA_MODEL")).strip() or "llama3.1:8b"
    last_err = ""

    try:
        resp = requests.post(
            f"{base_url}/api/chat",
            headers={"Content-Type": "application/json"},
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "stream": False,
                "options": {"temperature": float(temperature), "num_predict": int(max_tokens)},
            },
            timeout=60,
        )
        if resp.ok:
            data = resp.json() or {}
            msg = data.get("message") or {}
            content = _safe_str(msg.get("content")).strip()
            if content:
                resources.pop("last_llm_error", None)
                return {"content": content, "source": "ollama", "model": model}
        else:
            last_err = f"ollama /api/chat HTTP {resp.status_code}: {_safe_str(resp.text)[:240]}"
    except Exception:
        last_err = "ollama /api/chat error"

    try:
        prompt = f"{system_prompt}\n\n{user_prompt}".strip()
        resp = requests.post(
            f"{base_url}/api/generate",
            headers={"Content-Type": "application/json"},
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "keep_alive": "10m",
                "options": {"temperature": float(temperature), "num_predict": int(max_tokens)},
            },
            timeout=60,
        )
        if not resp.ok:
            last_err = f"ollama /api/generate HTTP {resp.status_code}: {_safe_str(resp.text)[:240]}"
            resources["last_llm_error"] = last_err
            return None
        data = resp.json() or {}
        content = _safe_str(data.get("response")).strip()
        if not content:
            last_err = "ollama /api/generate empty response"
            resources["last_llm_error"] = last_err
            return None
        resources.pop("last_llm_error", None)
        return {"content": content, "source": "ollama", "model": model}
    except Exception:
        last_err = "ollama /api/generate error"
        if last_err:
            resources["last_llm_error"] = last_err
        return None


def _huggingface_chat(system_prompt: str, user_prompt: str, max_tokens: int, temperature: float) -> dict | None:
    hf_key = _safe_str(os.environ.get("HUGGINGFACE_API_KEY")).strip()
    if not hf_key:
        return None

    configured = _safe_str(os.environ.get("HUGGINGFACE_MODEL")).strip() or "HuggingFaceTB/SmolLM3-3B:hf-inference"
    fallback_models = [
        configured,
        "meta-llama/Meta-Llama-3-8B-Instruct",
        "mistralai/Mistral-7B-Instruct",
        "Qwen/Qwen2-7B-Instruct",
        "HuggingFaceH4/zephyr-7b-beta",
    ]
    last_error = ""
    for model in fallback_models:
        try:
            resp = requests.post(
                "https://router.huggingface.co/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {hf_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "max_tokens": int(max_tokens),
                    "temperature": float(temperature),
                    "stream": False,
                },
                timeout=45,
            )
            if not resp.ok:
                last_error = f"huggingface HTTP {resp.status_code}: {_safe_str(resp.text)[:240]}"
                # Try next fallback if model unsupported/deprecated
                if resp.status_code in (404, 410):
                    continue
                # If router returns structured error with model_no_longer_supported, also continue
                try:
                    err_json = resp.json()
                    code = _safe_str(((err_json.get("error") or {}) or {}).get("code")).strip()
                    if code == "model_no_longer_supported":
                        continue
                except Exception:
                    pass
                resources["last_llm_error"] = last_error
                return None
            raw = resp.json() or {}
            content = ""
            try:
                content = _safe_str(((raw.get("choices") or [])[0] or {}).get("message", {}).get("content"))
            except Exception:
                content = ""
            content = _safe_str(content).strip()
            if not content:
                last_error = "huggingface empty response"
                continue
            resources.pop("last_llm_error", None)
            return {"content": content, "source": "huggingface", "model": model}
        except Exception as e:
            last_error = f"huggingface error: {e}"
            continue
    resources["last_llm_error"] = last_error or "huggingface: all fallback models failed"
    return None


def _chat_llm(system_prompt: str, user_prompt: str, max_tokens: int = 220, temperature: float = 0.8) -> dict | None:
    ollama_configured = bool(_safe_str(os.environ.get("OLLAMA_BASE_URL")).strip())
    hf_configured = bool(_safe_str(os.environ.get("HUGGINGFACE_API_KEY")).strip())
    if not ollama_configured and not hf_configured:
        resources["last_llm_error"] = "LLM non configuré (définir HUGGINGFACE_API_KEY ou OLLAMA_BASE_URL)"
        return None

    ollama_first = ollama_configured
    if ollama_first:
        r = _ollama_chat(system_prompt=system_prompt, user_prompt=user_prompt, max_tokens=max_tokens, temperature=temperature)
        if r:
            return r
    r = _huggingface_chat(system_prompt=system_prompt, user_prompt=user_prompt, max_tokens=max_tokens, temperature=temperature)
    if r:
        return r
    if not ollama_first:
        r = _ollama_chat(system_prompt=system_prompt, user_prompt=user_prompt, max_tokens=max_tokens, temperature=temperature)
        if r:
            return r
    return None


def _fallback_quiz(topic: str | None, difficulty: str | None) -> dict:
    topic_s = _safe_str(topic).strip() or "culture_generale"
    diff_s = _safe_str(difficulty).strip().lower() or "medium"

    points_map = {"easy": 150, "medium": 300, "hard": 500, "expert": 800}
    default_points = points_map.get(diff_s, 300)

    bank = [
        {
            "topic": "culture_generale",
            "difficulty": "easy",
            "question": "Quel est le plus grand océan sur Terre ?",
            "answers": ["océan pacifique", "pacifique"],
            "points": points_map["easy"],
        },
        {
            "topic": "culture_generale",
            "difficulty": "easy",
            "question": "Quelle planète est surnommée la planète rouge ?",
            "answers": ["mars"],
            "points": points_map["easy"],
        },
        {
            "topic": "culture_generale",
            "difficulty": "easy",
            "question": "Combien de continents y a-t-il sur Terre ?",
            "answers": ["7", "sept"],
            "points": points_map["easy"],
        },
        {
            "topic": "culture_generale",
            "difficulty": "easy",
            "question": "Quel animal est surnommé le roi de la jungle ?",
            "answers": ["lion", "le lion"],
            "points": points_map["easy"],
        },
        {
            "topic": "culture_generale",
            "difficulty": "medium",
            "question": "Dans quel pays se trouve la ville de Marrakech ?",
            "answers": ["maroc", "le maroc"],
            "points": points_map["medium"],
        },
        {
            "topic": "culture_generale",
            "difficulty": "medium",
            "question": "Quel est l'auteur de « Les Misérables » ?",
            "answers": ["victor hugo", "hugo"],
            "points": points_map["medium"],
        },
        {
            "topic": "culture_generale",
            "difficulty": "medium",
            "question": "Quelle est la capitale du Canada ?",
            "answers": ["ottawa"],
            "points": points_map["medium"],
        },
        {
            "topic": "culture_generale",
            "difficulty": "medium",
            "question": "Quel instrument mesure la pression atmosphérique ?",
            "answers": ["barometre", "baromètre"],
            "points": points_map["medium"],
        },
        {
            "topic": "culture_generale",
            "difficulty": "medium",
            "question": "Quel est le fleuve qui traverse Paris ?",
            "answers": ["seine", "la seine"],
            "points": points_map["medium"],
        },
        {
            "topic": "culture_generale",
            "difficulty": "medium",
            "question": "Dans quel pays se trouve la ville de Kyoto ?",
            "answers": ["japon", "le japon"],
            "points": points_map["medium"],
        },
        {
            "topic": "culture_generale",
            "difficulty": "medium",
            "question": "Quel est l'élément chimique dont le symbole est O ?",
            "answers": ["oxygene", "oxygène"],
            "points": points_map["medium"],
        },
        {
            "topic": "culture_generale",
            "difficulty": "hard",
            "question": "Quelle est la capitale de la Nouvelle-Zélande ?",
            "answers": ["wellington"],
            "points": points_map["hard"],
        },
        {
            "topic": "culture_generale",
            "difficulty": "hard",
            "question": "Quel physicien a formulé les lois du mouvement et la gravitation universelle ?",
            "answers": ["isaac newton", "newton"],
            "points": points_map["hard"],
        },
        {
            "topic": "culture_generale",
            "difficulty": "hard",
            "question": "Dans quel pays se trouve la région du Transylvanie ?",
            "answers": ["roumanie", "la roumanie"],
            "points": points_map["hard"],
        },
        {
            "topic": "culture_generale",
            "difficulty": "hard",
            "question": "Quel traité de 1919 a officiellement mis fin à la Première Guerre mondiale (avec l'Allemagne) ?",
            "answers": ["traite de versailles", "traité de versailles", "versailles"],
            "points": points_map["hard"],
        },
        {
            "topic": "culture_generale",
            "difficulty": "hard",
            "question": "Quel est le nom du processus par lequel une plante convertit l'énergie lumineuse en énergie chimique ?",
            "answers": ["photosynthese", "photosynthèse"],
            "points": points_map["hard"],
        },
        {
            "topic": "culture_generale",
            "difficulty": "hard",
            "question": "Quel est le nom de la mer intérieure située entre l'Europe, l'Asie et l'Afrique ?",
            "answers": ["mer mediterranee", "mer méditerranée", "mediterranee", "méditerranée"],
            "points": points_map["hard"],
        },
        {
            "topic": "culture_generale",
            "difficulty": "hard",
            "question": "Quelle est la particule porteuse de l'interaction électromagnétique ?",
            "answers": ["photon", "le photon"],
            "points": points_map["hard"],
        },
        {
            "topic": "culture_generale",
            "difficulty": "expert",
            "question": "Quel mathématicien a formulé le théorème d'incomplétude (1931) ?",
            "answers": ["kurt godel", "kurt gödel", "godel", "gödel"],
            "points": points_map["expert"],
        },
        {
            "topic": "culture_generale",
            "difficulty": "expert",
            "question": "Dans quelle ville se situe le siège de l'Organisation des Nations unies pour l'éducation, la science et la culture (UNESCO) ?",
            "answers": ["paris"],
            "points": points_map["expert"],
        },
        {
            "topic": "culture_generale",
            "difficulty": "expert",
            "question": "Quel est le nom de l'algorithme de chiffrement symétrique qui a succédé à DES comme standard (2001) ?",
            "answers": ["aes", "advanced encryption standard", "rijndael"],
            "points": points_map["expert"],
        },
        {
            "topic": "culture_generale",
            "difficulty": "expert",
            "question": "Quel est le nom de la métrique cosmologique utilisée dans le modèle standard pour un Univers homogène et isotrope ?",
            "answers": ["flrw", "friedmann lematre robertson walker", "friedmann-lamaitre-robertson-walker", "friedmann lematre robertson walker"],
            "points": points_map["expert"],
        },
        {
            "topic": "culture_generale",
            "difficulty": "expert",
            "question": "Quel est le nom du principe selon lequel un système quantique ne peut pas être décrit indépendamment de l'appareil de mesure, popularisé par l'école de Copenhague ?",
            "answers": ["complementarite", "complémentarité", "principe de complementarite", "principe de complémentarité"],
            "points": points_map["expert"],
        },
    ]

    candidates = [q for q in bank if q["topic"] == topic_s and q["difficulty"] == diff_s]
    if not candidates:
        candidates = [q for q in bank if q["topic"] == topic_s] or bank

    recent = resources.setdefault("recent_quiz_questions", [])
    if not isinstance(recent, list):
        recent = []
        resources["recent_quiz_questions"] = recent
    recent_set = {(_safe_str(x).strip()) for x in recent if _safe_str(x).strip()}
    fresh = [q for q in candidates if _safe_str(q.get("question")).strip() not in recent_set]
    chosen = random.choice(fresh or candidates)
    qtxt = _safe_str(chosen.get("question")).strip()
    if qtxt:
        recent.append(qtxt)
        if len(recent) > 25:
            del recent[:-25]
    return {
        "id": str(uuid.uuid4()),
        "topic": chosen["topic"],
        "difficulty": chosen["difficulty"],
        "question": chosen["question"],
        "answers": chosen["answers"],
        "points": safe_int(chosen.get("points"), default_points) or default_points,
        "source": "fallback",
    }


def _generate_quiz_llm(topic: str | None, difficulty: str | None) -> dict | None:
    topic_s = _safe_str(topic).strip() or "culture_generale"
    diff_s = _safe_str(difficulty).strip().lower() or "medium"
    points_map = {"easy": 150, "medium": 300, "hard": 500, "expert": 800}
    target_points = points_map.get(diff_s, 300)
    difficulty_rules = {
        "easy": "Niveau easy: question très accessible, connue du grand public, sans piège.",
        "medium": "Niveau medium: question accessible mais pas triviale, nécessite un minimum de culture générale.",
        "hard": "Niveau hard: question difficile, évite les évidences (capitale/planète/couleur).",
        "expert": "Niveau expert: question très difficile et précise, évite les évidences, privilégie histoire/sciences/art/geopolitique.",
    }
    rules = difficulty_rules.get(diff_s, difficulty_rules["medium"])

    sys_prompt = (
        "Tu génères une question de quiz de culture générale en français. "
        "Réponds UNIQUEMENT en JSON valide, sans texte autour."
    )
    recent = resources.setdefault("recent_quiz_questions", [])
    if not isinstance(recent, list):
        recent = []
        resources["recent_quiz_questions"] = recent
    recent_set = {(_safe_str(x).strip()) for x in recent if _safe_str(x).strip()}

    last_questions = [q for q in reversed(recent) if _safe_str(q).strip()][:5]
    last_questions_txt = "\n".join([f"- {_safe_str(q).strip()}" for q in last_questions if _safe_str(q).strip()])

    llm = None
    obj = None
    question = ""
    answers_raw = None
    points_raw = None
    for _ in range(3):
        avoid_block = f"\nNe répète pas ces questions:\n{last_questions_txt}\n" if last_questions_txt else "\n"
        user_prompt = (
            "Génère une question (une seule) et les réponses acceptées. "
            "Contraintes:\n"
            f"- topic: {topic_s}\n"
            f"- difficulté: {diff_s}\n"
            f"- {rules}\n"
            "- la question doit être claire et courte\n"
            "- réponses acceptées: 1 à 4 variantes (minuscules, sans ponctuation)\n"
            f"- points: entier EXACT ({target_points})\n"
            f"{avoid_block}\n"
            "Format JSON attendu:\n"
            "{\n"
            '  "question": "…",\n'
            '  "answers": ["…"],\n'
            f'  "points": {target_points}\n'
            "}"
        )

        llm = _chat_llm(system_prompt=sys_prompt, user_prompt=user_prompt, max_tokens=220, temperature=0.8)
        if not llm:
            return None

        obj = _extract_first_json_object(_safe_str(llm.get("content")))
        if not obj:
            continue

        question = _safe_str(obj.get("question")).strip()
        answers_raw = obj.get("answers")
        points_raw = obj.get("points")
        if not question:
            continue
        if question in recent_set:
            continue
        break
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
        points = target_points
    points = target_points

    qtxt = _safe_str(question).strip()
    if qtxt:
        recent.append(qtxt)
        if len(recent) > 25:
            del recent[:-25]

    return {
        "id": str(uuid.uuid4()),
        "topic": topic_s,
        "difficulty": diff_s,
        "question": question,
        "answers": answers,
        "points": points,
        "source": _safe_str(llm.get("source")) or "llm",
        "model": _safe_str(llm.get("model")) or "",
    }


def _pick_top_providers(items, limit: int) -> list[str]:
    if not isinstance(items, list):
        return []
    rows = [x for x in items if isinstance(x, dict)]
    rows.sort(key=lambda r: float(r.get("score", 0.0)), reverse=True)
    out: list[str] = []
    for r in rows[: max(0, int(limit))]:
        p = _safe_str(r.get("provider")).strip().lower()
        if p:
            out.append(p)
    return out


def _pick_top_offers(items, limit: int) -> list[dict]:
    if not isinstance(items, list):
        return []
    rows = [x for x in items if isinstance(x, dict)]
    rows.sort(key=lambda r: float(r.get("score", 0.0)), reverse=True)
    out: list[dict] = []
    for r in rows[: max(0, int(limit))]:
        offer_id = _safe_str(r.get("offer_id")).strip()
        title = _safe_str(r.get("title")).strip()
        payout_points = safe_int(r.get("payout_points"), 0)
        if offer_id or title:
            out.append({"offer_id": offer_id, "title": title, "payout_points": payout_points})
    return out


def _stable_pick_index(seed: str, n: int) -> int:
    if n <= 0:
        return 0
    h = hashlib.sha256(_safe_str(seed).encode("utf-8")).hexdigest()
    v = int(h[:12], 16) if h else 0
    return v % n


def _stable_choice(seed: str, options: list[str]) -> str:
    if not options:
        return ""
    return options[_stable_pick_index(seed, len(options))]


def _fallback_notification_message(payload: dict) -> dict:
    user_id = _safe_str(payload.get("user_id")).strip() or "unknown"
    personalization = payload.get("personalization")
    if not isinstance(personalization, dict):
        personalization = {}
    mix = personalization.get("mix")
    if not isinstance(mix, dict):
        mix = {}
    reason = _safe_str(mix.get("reason")).strip() or "balanced"
    providers = personalization.get("providers")
    if not isinstance(providers, dict):
        providers = {}
    top_surveys = _pick_top_providers((providers.get("surveys") or []), 2)
    top_offerwalls = _pick_top_providers((providers.get("offerwalls") or []), 1)
    top_iframes = _pick_top_providers((providers.get("iframes") or []), 1)
    top_offers = _pick_top_offers((personalization.get("offers") or []), 2)

    target = payload.get("target")
    if isinstance(target, dict):
        t_type = _safe_str(target.get("type")).strip().lower()
        t_title = _safe_str(target.get("title")).strip()
        t_desc = _safe_str(target.get("description")).strip()
        t_points = safe_int(target.get("points"), 0)
        if t_type == "offer" and t_title:
            body = (t_desc or t_title).replace("\n", " ").strip()
            if len(body) > 400:
                body = body[:397] + "..."
            points_bit = f" • Gagne {t_points} Points" if t_points > 0 else ""
            title_tpl = _stable_choice(
                f"{user_id}:notif:offer:title:{t_title}",
                [
                    "Offre recommandée : {t}",
                    "À ne pas rater : {t}",
                    "Bon plan du moment : {t}",
                    "Défi rapide : {t}",
                    "Gros gain possible : {t}",
                    "Suggestion du jour : {t}",
                ],
            )
            body_tpl = _stable_choice(
                f"{user_id}:notif:offer:body:{t_title}:{t_points}",
                [
                    "{d}{p} • Lance-la maintenant",
                    "{d}{p} • Objectif: valider et encaisser",
                    "{d}{p} • Prêt à booster tes points ?",
                    "{d}{p} • Clique, complète, récupère tes points",
                    "{d}{p} • Plus tu avances, plus tu gagnes",
                ],
            )
            resolved_title = (title_tpl or "Offre recommandée : {t}").format(t=t_title).strip()
            resolved_body = (body_tpl or "{d}{p}").format(d=body, p=points_bit).strip()
            return {"kind": "success", "title": resolved_title, "body": resolved_body}
        if t_type == "survey":
            title = t_title or "Sondage"
            body = (t_desc or title).replace("\n", " ").strip()
            if len(body) > 400:
                body = body[:397] + "..."
            points_bit = f" • Gagne {t_points} Points" if t_points > 0 else ""
            title_tpl = _stable_choice(
                f"{user_id}:notif:survey:title:{title}",
                [
                    "Sondage recommandé : {t}",
                    "Sondage rapide : {t}",
                    "À faire aujourd’hui : {t}",
                    "Bonus sondage : {t}",
                    "Suggestion : {t}",
                ],
            )
            body_tpl = _stable_choice(
                f"{user_id}:notif:survey:body:{title}:{t_points}",
                [
                    "{d}{p} • Réponds et gagne",
                    "{d}{p} • Quelques minutes, des points",
                    "{d}{p} • Profil complet = plus de sondages",
                    "{d}{p} • Fais-le maintenant",
                ],
            )
            resolved_title = (title_tpl or "Sondage recommandé : {t}").format(t=title).strip()
            resolved_body = (body_tpl or "{d}{p}").format(d=body, p=points_bit).strip()
            return {"kind": "success", "title": resolved_title, "body": resolved_body}

    if reason == "survey-heavy":
        body = f"Priorité: {', '.join([s.upper() for s in top_surveys])}." if top_surveys else "On a repéré que tu préfères les sondages en ce moment."
        return {"kind": "info", "title": "Sondages recommandés aujourd’hui", "body": body}

    if reason == "offer-heavy":
        if top_offers:
            body_tpl = _stable_choice(
                f"{user_id}:notif:offer-heavy:{top_offers[0].get('offer_id')}",
                [
                    "À tester aujourd’hui: {items}",
                    "Deux offres à gros potentiel: {items}",
                    "Objectif du jour: {items}",
                    "Sélection du moment: {items}",
                ],
            )
            items_txt = " • ".join([f"{o.get('title')} (+{o.get('payout_points')})" for o in top_offers if _safe_str(o.get("title")).strip()])
            body = (body_tpl or "{items}").format(items=items_txt).strip()
        else:
            body = "On a repéré que tu préfères les offres en ce moment."
        title_tpl = _stable_choice(
            f"{user_id}:notif:offer-heavy:title",
            ["Offres recommandées", "Suggestions d’offres", "Offres du moment", "Bonnes offres à faire"],
        )
        return {"kind": "success", "title": title_tpl or "Offres recommandées", "body": body or "Offres recommandées aujourd’hui."}

    bits: list[str] = []
    if top_surveys[:1]:
        bits.append(f"Sondage: {top_surveys[0].upper()}")
    if top_offers[:1]:
        o = top_offers[0]
        title = _safe_str(o.get("title")).strip()
        pts = safe_int(o.get("payout_points"), 0)
        if title:
            bits.append(f"Offre: {title} (+{pts})" if pts else f"Offre: {title}")
    if top_offerwalls[:1]:
        bits.append(f"Offerwall: {top_offerwalls[0].upper()}")
    if top_iframes[:1]:
        bits.append(f"Iframe: {top_iframes[0].upper()}")
    return {
        "kind": "info",
        "title": "Suggestions personnalisées",
        "body": " • ".join(bits) or "Va voir tes offres et sondages recommandés.",
    }


def _generate_notification_message_llm(payload: dict) -> dict | None:
    personalization = payload.get("personalization")
    if not isinstance(personalization, dict):
        personalization = {}
    mix = personalization.get("mix")
    if not isinstance(mix, dict):
        mix = {}
    reason = _safe_str(mix.get("reason")).strip() or "balanced"

    providers = personalization.get("providers")
    if not isinstance(providers, dict):
        providers = {}
    top_surveys = _pick_top_providers((providers.get("surveys") or []), 2)
    top_offerwalls = _pick_top_providers((providers.get("offerwalls") or []), 1)
    top_iframes = _pick_top_providers((providers.get("iframes") or []), 1)
    top_offers = _pick_top_offers((personalization.get("offers") or []), 2)

    action_url = _safe_str(payload.get("action_url")).strip()
    target = payload.get("target") if isinstance(payload.get("target"), dict) else {}
    explore = payload.get("explore") if isinstance(payload.get("explore"), dict) else {}

    sys_prompt = (
        "Tu écris une notification pour une app de rewards (points, offres, sondages). "
        "Langue: français. Ton: direct et motivant. "
        "Réponds UNIQUEMENT en JSON valide, sans texte autour."
    )
    user_prompt = (
        "Génère un message de notification personnalisé pour l'utilisateur.\n"
        "Contraintes:\n"
        "- JSON attendu: {\"kind\":\"info|success\",\"title\":\"...\",\"body\":\"...\"}\n"
        "- title: court (< 60 chars)\n"
        "- body: court (< 140 chars)\n"
        "- pas d'emoji\n"
        "- pas de guillemets inutiles\n\n"
        f"Décision (reason): {reason}\n"
        f"Top surveys: {', '.join(top_surveys) if top_surveys else 'none'}\n"
        f"Top offers: {', '.join([_safe_str(o.get('title')) for o in top_offers if _safe_str(o.get('title')).strip()]) or 'none'}\n"
        f"Top offerwalls: {', '.join(top_offerwalls) if top_offerwalls else 'none'}\n"
        f"Top iframes: {', '.join(top_iframes) if top_iframes else 'none'}\n"
        f"Explore survey_provider: {_safe_str(explore.get('survey_provider')).strip()}\n"
        f"Explore offer_provider: {_safe_str(explore.get('offer_provider')).strip()}\n"
        f"Target type: {_safe_str(target.get('type')).strip()}\n"
        f"Target title: {_safe_str(target.get('title')).strip()}\n"
        f"Target points: {safe_int(target.get('points'), 0)}\n"
        f"Action URL: {action_url}\n"
    )

    llm = _chat_llm(system_prompt=sys_prompt, user_prompt=user_prompt, max_tokens=180, temperature=0.7)
    if not llm:
        return None

    obj = _extract_first_json_object(_safe_str(llm.get("content")))
    if not obj:
        return None

    kind = _safe_str(obj.get("kind")).strip().lower()
    if kind not in {"info", "success", "warning", "error"}:
        kind = "info"
    title = _safe_str(obj.get("title")).strip()
    body = _safe_str(obj.get("body")).strip()
    if not title or not body:
        return None
    if len(title) > 80:
        title = title[:80].rstrip()
    if len(body) > 220:
        body = body[:220].rstrip()
    return {"kind": kind, "title": title, "body": body, "source": _safe_str(llm.get("source")), "model": _safe_str(llm.get("model"))}


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
    fallback = ["revlum", "kiwiwall", "opinionuniverse", "notik"]
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
        "opinion": "opinionuniverse",
        "opinion-universe": "opinionuniverse",
        "opinion_universe": "opinionuniverse",
    }
    s = aliases.get(s, s)
    if s in {"revlum", "kiwiwall", "opinionuniverse", "notik", "mylead", "wannads"}:
        return s
    if "revlum" in s:
        return "revlum"
    if "kiwi" in s:
        return "kiwiwall"
    if "opinion" in s:
        return "opinionuniverse"
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

    candidates = {"revlum", "kiwiwall", "opinionuniverse", "notik"}
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

    survey_providers = {"cpx", "rapidoreach", "theoremreach", "bitlabs", "notik", "opinionuniverse"}
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
    allowed_providers = {"cpx", "rapidoreach", "theoremreach", "bitlabs", "notik", "opinionuniverse"}
    fast_providers = {"cpx", "bitlabs"}
    fallback = ["cpx", "rapidoreach", "theoremreach", "bitlabs", "opinionuniverse"]

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
        resources["translation_cache"] = _load_translation_cache()
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
    return {
        "status": "ok",
        "version": SERVICE_VERSION,
        "startup_error": resources.get("startup_error"),
        "llm": _llm_status(),
        "last_llm_error": resources.get("last_llm_error") or "",
    }

@app.post("/translate")
async def translate(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}
    text = _safe_str((body or {}).get("text")).strip()
    target_lang = _safe_str((body or {}).get("target_lang")).strip().lower() or "fr"
    offer_id = _safe_str((body or {}).get("offer_id")).strip()
    field = _safe_str((body or {}).get("field")).strip() or "text"
    provider = _safe_str((body or {}).get("provider")).strip()
    if not text:
        return {"translated": ""}
    try:
        print("[IA-TRANSLATE][STATUS]", {"llm": _llm_status()})
    except Exception:
        pass
    cached = _get_cached_translation(text=text, target_lang=target_lang)
    if cached:
        try:
            print("[IA-TRANSLATE][CACHE]", {"lang": target_lang, "input": text[:120], "output": cached[:120]})
        except Exception:
            pass
        return {"translated": cached, "source": "cache", "model": "", "prepared": _normalize_for_translation(text), "reason": "cache_hit"}
    prepared = _normalize_for_translation(text)
    if offer_id:
        try:
            rows = _supabase_get(
                "offer_translations",
                {
                    "select": "translated_text",
                    "offer_id": f"eq.{offer_id}",
                    "lang": f"eq.{target_lang}",
                    "field": f"eq.{field}",
                    "original_text": f"eq.{text}",
                    "limit": "1",
                },
            )
            if rows:
                tt = _safe_str((rows[0] or {}).get("translated_text")).strip()
                if tt:
                    return {"translated": tt, "source": "db", "model": "", "prepared": prepared, "reason": "db_hit"}
        except Exception:
            pass
    sys_prompt = (
        "Tu es un traducteur pour une app de récompenses mobile (offerwall). "
        "Tu reçois des libellés de tâches et descriptions. "
        "Objectif: produire une phrase naturelle et concise dans la langue cible, à l'impératif, sans identifiants techniques. "
        "Contraintes: "
        "- Garde les nombres intacts (ex: 150). "
        "- Pas d'underscore, pas de code, pas d'emoji. "
        "- Sortie courte (< 60 caractères si possible). "
        "Exemples: "
        'Input: "player_reached_150_reachlevel" -> FR: "Atteindre le niveau 150" / DE: "Erreiche Level 150". '
        'Input: "cg_player_reached_500_distancelevel" -> FR: "Atteindre le niveau de distance 500". '
    )
    user_prompt = f"Langue cible: {target_lang}\nTexte:\n{prepared}"
    llm = _chat_llm(system_prompt=sys_prompt, user_prompt=user_prompt, max_tokens=600, temperature=0.2)
    if llm and isinstance(llm, dict):
        out = _safe_str(llm.get("content")).strip()
        if out:
            _set_cached_translation(text=text, target_lang=target_lang, translated=out)
            try:
                print("[IA-TRANSLATE][LLM]", {"lang": target_lang, "prepared": prepared[:120], "output": out[:120], "source": _safe_str(llm.get("source")), "model": _safe_str(llm.get("model"))})
            except Exception:
                pass
            if offer_id:
                try:
                    _supabase_rpc(
                        "upsert_offer_translation",
                        {
                            "p_offer_id": offer_id,
                            "p_lang": target_lang,
                            "p_field": field,
                            "p_original_text": text,
                            "p_translated_text": out,
                            "p_source": _safe_str(llm.get("source")),
                            "p_model": _safe_str(llm.get("model")),
                            "p_provider": provider,
                        },
                    )
                except Exception:
                    pass
            return {"translated": out, "source": _safe_str(llm.get("source")), "model": _safe_str(llm.get("model")), "prepared": prepared, "reason": "llm_success"}
    reason = "llm_empty_or_error"
    llm_err = _safe_str(resources.get("last_llm_error") or "")
    try:
        print("[IA-TRANSLATE][FALLBACK]", {"lang": target_lang, "prepared": prepared[:120], "reason": reason, "llm_error": llm_err})
    except Exception:
        pass
    if _is_translate_strict():
        raise HTTPException(status_code=503, detail={"error": "LLM unavailable or returned empty", "prepared": prepared, "llm_error": llm_err, "llm_status": _llm_status()})
    return {"translated": text, "source": "fallback", "model": "", "prepared": prepared, "reason": reason, "llm_error": llm_err}

@app.post("/internal/notification-message")
async def internal_notification_message(request: Request):
    _require_internal_token(request)

    try:
        body = await request.json()
    except Exception:
        body = {}

    if not isinstance(body, dict):
        body = {}

    payload = {
        "user_id": body.get("user_id"),
        "personalization": body.get("personalization") if isinstance(body.get("personalization"), dict) else {},
        "action_url": body.get("action_url"),
        "target": body.get("target") if isinstance(body.get("target"), dict) else {},
        "explore": body.get("explore") if isinstance(body.get("explore"), dict) else {},
    }

    try:
        llm = _generate_notification_message_llm(payload)
    except Exception:
        llm = None

    if llm and isinstance(llm, dict):
        return {
            "kind": _safe_str(llm.get("kind")).strip() or "info",
            "title": _safe_str(llm.get("title")).strip(),
            "body": _safe_str(llm.get("body")).strip(),
            "source": _safe_str(llm.get("source")).strip(),
            "model": _safe_str(llm.get("model")).strip(),
        }

    fallback = _fallback_notification_message(payload)
    return {
        "kind": _safe_str(fallback.get("kind")).strip() or "info",
        "title": _safe_str(fallback.get("title")).strip(),
        "body": _safe_str(fallback.get("body")).strip(),
        "source": "fallback",
        "model": "",
    }

@app.get("/llm-test")
def llm_test():
    try:
        base_url = _safe_str(os.environ.get("OLLAMA_BASE_URL")).strip().rstrip("/") or "http://localhost:11434"
        model = _safe_str(os.environ.get("OLLAMA_MODEL")).strip() or "llama3.1:8b"
        response = requests.post(
            f"{base_url}/api/generate",
            json={
                "model": model,
                "prompt": "Réponds uniquement: OK",
                "stream": False,
                "keep_alive": "10m",
                "options": {"num_predict": 20},
            },
            timeout=180,
        )
        response.raise_for_status()
        data = response.json() or {}
        return {"answer": data.get("response", ""), "model": model}
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
    try:
        limit_int = max(1, safe_int(limit, 6))
    except Exception:
        limit_int = 6

    now = datetime.now(timezone.utc)
    since_global = _to_iso(now - timedelta(days=45))
    since_user = _to_iso(now - timedelta(days=120))

    try:
        user_rows = _supabase_get_first_success(
            "transaction_offers",
            variants=[
                {
                    "select": "offer_id,created_at,status",
                    "user_id": f"eq.{user_id}",
                    "status": "eq.1",
                    "created_at": f"gte.{since_user}",
                    "order": "created_at.desc",
                    "limit": "2000",
                },
                {
                    "select": "offer_id,status",
                    "user_id": f"eq.{user_id}",
                    "status": "eq.1",
                    "limit": "2000",
                },
            ],
            timeout_s=18,
        )
    except Exception:
        user_rows = []

    already_done: set[str] = set()
    for r in user_rows:
        if not isinstance(r, dict):
            continue
        oid = _safe_str(r.get("offer_id")).strip()
        if oid:
            already_done.add(oid)

    try:
        global_rows = _supabase_get_first_success(
            "transaction_offers",
            variants=[
                {
                    "select": "offer_id,offer_name,points,provider,created_at,status",
                    "status": "eq.1",
                    "created_at": f"gte.{since_global}",
                    "order": "created_at.desc",
                    "limit": "4000",
                },
                {"select": "offer_id,offer_name,points,provider,status", "status": "eq.1", "limit": "4000"},
            ],
            timeout_s=22,
        )
    except Exception:
        global_rows = []

    agg: dict[str, dict] = {}
    for r in global_rows:
        if not isinstance(r, dict):
            continue
        oid = _safe_str(r.get("offer_id")).strip()
        if not oid:
            continue
        if oid in already_done:
            continue
        pts = safe_int(r.get("points"), 0)
        if pts <= 0:
            continue
        name = _safe_str(r.get("offer_name")).strip()
        provider = _safe_str(r.get("provider")).strip().lower()
        a = agg.get(oid)
        if not a:
            a = {
                "offer_id": oid,
                "title": name or oid,
                "provider": provider,
                "count": 0,
                "sum_points": 0,
                "max_points": 0,
            }
            agg[oid] = a
        a["count"] = int(a.get("count", 0)) + 1
        a["sum_points"] = int(a.get("sum_points", 0)) + pts
        a["max_points"] = max(int(a.get("max_points", 0)), pts)
        if name and (a.get("title") == oid or len(_safe_str(a.get("title"))) < 6):
            a["title"] = name

    if not agg:
        return []

    max_count = max(int(v.get("count", 0)) for v in agg.values()) or 1
    max_points = max(int(v.get("max_points", 0)) for v in agg.values()) or 1

    items: list[dict] = []
    for v in agg.values():
        count = int(v.get("count", 0))
        pts = int(v.get("max_points", 0)) or int(v.get("sum_points", 0) / max(1, count))
        if pts < 200:
            continue
        score_pop = float(count) / float(max_count) if max_count > 0 else 0.0
        score_pts = float(pts) / float(max_points) if max_points > 0 else 0.0
        score = clamp01(0.65 * score_pop + 0.35 * score_pts)
        reason = "Populaire en ce moment"
        if score_pts > 0.7 and score_pop > 0.35:
            reason = "Très demandé et bon gain"
        elif score_pts > 0.7:
            reason = "Bon gain"
        elif score_pop > 0.5:
            reason = "Très demandé"
        items.append(
            {
                "offer_id": _safe_str(v.get("offer_id")).strip(),
                "title": _safe_str(v.get("title")).strip(),
                "payout_points": pts,
                "score": round(float(score), 4),
                "reason": reason,
            }
        )

    items.sort(key=lambda x: (float(x.get("score", 0.0)), safe_int(x.get("payout_points"), 0)), reverse=True)

    month_key = now.strftime("%Y-%m")
    seed_int = int(hashlib.sha256(f"{user_id}:{month_key}:offers".encode("utf-8")).hexdigest()[:12], 16)
    rng = random.Random(seed_int)
    pool = items[:60]
    rng.shuffle(pool)
    return pool[:limit_int]


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
    if not startup_error:
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
        now = datetime.now(timezone.utc)
        month_key = now.strftime("%Y-%m")
        h = int(hashlib.sha256(f"{user_id}:{month_key}:offer-push".encode("utf-8")).hexdigest()[:12], 16)
        d1 = min(28, 3 + (h % 6))
        d2 = min(28, 17 + ((h // 7) % 6))
        push_days = {d1, d2}
        if offers and now.day in push_days:
            mix = dict(mix or {})
            mix["reason"] = "offer-heavy"
            mix["offers_weight"] = max(0.75, float(mix.get("offers_weight", 0.5)))
            mix["surveys_weight"] = min(0.25, float(mix.get("surveys_weight", 0.5)))
    except Exception:
        pass

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

    fallback = _fallback_quiz(topic=topic, difficulty=difficulty)
    if isinstance(fallback, dict):
        fallback["llm"] = _llm_status()
        fallback["llm_error"] = resources.get("last_llm_error") or ""
    return fallback


def _calculate_level_from_xp(xp_total: int) -> dict:
    """
    Réplique la logique de calcul de niveau du frontend (Profile.tsx).
    xpForLevel(level) = 50 + (level * 15)
    """
    level = 1
    current_xp = xp_total
    
    # Sécurité pour éviter boucle infinie si xp très grand
    while level < 1000:
        xp_needed = 50 + (level * 15)
        if current_xp >= xp_needed:
            current_xp -= xp_needed
            level += 1
        else:
            break
            
    return {"level": level, "current_xp": current_xp, "xp_needed": 50 + (level * 15)}


def _get_support_context(user_id: str) -> dict:
    context = {
        "user_id": user_id, 
        "points": 0, 
        "level": 1, 
        "xp": 0, 
        "username": "Utilisateur", 
        "transactions": [], 
        "orders": [], 
        "giftcards_sample": [],
        "debug_logs": [] # Nouveaux logs détaillés
    }
    
    def log(msg):
        context["debug_logs"].append(msg)
        print(f"[SUPPORT-CONTEXT] {msg}")

    log(f"Fetching context for user {user_id}")

    # 1. Récupérer Profil
    try:
        log("Querying 'profiles' table...")
        # Essayer différentes variantes de colonnes au cas où le schéma a changé
        profiles = _supabase_get_first_success(
            "profiles",
            variants=[
                # Variante 1: Colonnes probables d'après Profile.tsx
                {"select": "points_total,xp,username,full_name", "id": f"eq.{user_id}", "limit": "1"},
                # Variante 2: Juste les points et username
                {"select": "points,username", "id": f"eq.{user_id}", "limit": "1"},
                # Variante 3: Au moins récupérer quelque chose
                {"select": "*", "id": f"eq.{user_id}", "limit": "1"},
            ],
            timeout_s=10
        )
        
        if profiles:
            p = profiles[0]
            # Gestion flexible des noms de colonnes
            context["points"] = safe_int(p.get("points_total") or p.get("points"), 0)
            
            # Récupération de l'XP totale brute
            raw_xp = safe_int(p.get("xp"), 0)
            
            # Recalcul du niveau basé sur l'XP totale (comme le frontend)
            level_info = _calculate_level_from_xp(raw_xp)
            context["level"] = level_info["level"]
            context["xp"] = level_info["current_xp"] # XP dans le niveau actuel
            context["xp_needed"] = level_info["xp_needed"]
            
            context["username"] = _safe_str(p.get("username") or p.get("full_name") or "Utilisateur").strip()
            
            log(f"User found: {context['username']} (Pts: {context['points']}, XP Total: {raw_xp} -> Lvl {context['level']})")
        else:
             log(f"User NOT found in profiles table: {user_id}")
             
    except Exception as e:
        log(f"Error fetching profile: {str(e)}")

    # 2. Récupérer Dernières Transactions (Offres)
    try:
        log("Querying 'transaction_offers' table...")
        # Essayer transaction_offers
        tx_rows = []
        try:
            tx_rows = _supabase_get_first_success(
                "transaction_offers",
                variants=[
                    {
                        "select": "provider,reward_points,status,created_at,title", 
                        "user_id": f"eq.{user_id}", 
                        "order": "created_at.desc", 
                        "limit": "5"
                    }
                ],
                timeout_s=5
            )
        except Exception:
            pass
            
        # Essayer transaction_history si transaction_offers vide ou erreur
        if not tx_rows:
             log("transaction_offers empty or failed, trying 'transaction_history'...")
             try:
                tx_rows = _supabase_get_first_success(
                    "transaction_history", # Table hypothétique commune
                    variants=[
                        {
                            "select": "*", 
                            "user_id": f"eq.{user_id}", 
                            "order": "created_at.desc", 
                            "limit": "5"
                        }
                    ],
                    timeout_s=5
                )
             except Exception:
                pass
        
        log(f"Found {len(tx_rows)} transactions.")
        for row in tx_rows:
            if isinstance(row, dict):
                # Extraction sécurisée des champs
                provider = _safe_str(row.get("provider") or row.get("source"))
                points = safe_int(row.get("reward_points") or row.get("points"), 0)
                status = _safe_str(row.get("status"))
                date = _safe_str(row.get("created_at"))
                title = _safe_str(row.get("title") or row.get("description") or "Action")
                
                context["transactions"].append({
                    "type": "offer",
                    "provider": provider,
                    "points": points,
                    "status": status,
                    "date": date,
                    "title": title
                })
    except Exception as e:
        log(f"Error fetching transactions: {str(e)}")

    # 3. Récupérer Dernières Commandes (Boutique & PayPal)
    # Commandes Cartes Cadeaux
    try:
        log("Querying 'orders' table (gift cards)...")
        orders_rows = _supabase_get_first_success(
            "orders",
            variants=[
                # Variante 1: points_used (Trouvé dans AdminOrders.tsx)
                {
                    "select": "id,status,created_at,gift_details,points_used", 
                    "user_id": f"eq.{user_id}", 
                    "order": "created_at.desc", 
                    "limit": "10"
                },
                # Variantes de secours
                {
                    "select": "id,status,created_at,gift_details,cost_points", 
                    "user_id": f"eq.{user_id}", 
                    "order": "created_at.desc", 
                    "limit": "10"
                }
            ],
            timeout_s=5
        )
        log(f"Found {len(orders_rows)} gift card orders.")
        for row in orders_rows:
            if isinstance(row, dict):
                gift_details = row.get("gift_details") or {}
                name = _safe_str(gift_details.get("name") or "Carte Cadeau")
                
                # Récupération flexible du coût
                cost = safe_int(row.get("points_used") or row.get("cost_points") or row.get("points"), 0)
                
                context["orders"].append({
                    "id": _safe_str(row.get("id")),
                    "type": "Carte Cadeau",
                    "name": name,
                    "points": cost,
                    "status": _safe_str(row.get("status")),
                    "date": _safe_str(row.get("created_at"))
                })
    except Exception as e:
        log(f"Error fetching gift card orders: {str(e)}")

    # Virements PayPal
    try:
        log("Querying 'paypal_transfers' table...")
        paypal_rows = _supabase_get_first_success(
            "paypal_transfers",
            variants=[
                # Variante 1: 'amount_eur' et 'points_used' (Trouvé dans AdminOrders.tsx)
                {
                    "select": "id,status,created_at,amount_eur,points_used", 
                    "user_id": f"eq.{user_id}", 
                    "order": "created_at.desc", 
                    "limit": "10"
                },
                # Variante 2: 'amount' et 'points_cost' (Fallback)
                {
                    "select": "id,status,created_at,amount,points_cost", 
                    "user_id": f"eq.{user_id}", 
                    "order": "created_at.desc", 
                    "limit": "10"
                }
            ],
            timeout_s=5
        )
        log(f"Found {len(paypal_rows)} PayPal transfers.")
        for row in paypal_rows:
            if isinstance(row, dict):
                # Récupération flexible du montant
                amount = safe_int(row.get("amount_eur") or row.get("amount"), 0)
                
                # Récupération flexible des points
                points = safe_int(row.get("points_used") or row.get("points_cost"), amount * 1000)
                
                context["orders"].append({
                    "id": _safe_str(row.get("id")),
                    "type": "Virement PayPal",
                    "name": f"PayPal {amount}€",
                    "points": points,
                    "status": _safe_str(row.get("status")),
                    "date": _safe_str(row.get("created_at"))
                })
    except Exception as e:
        log(f"Error fetching paypal transfers: {str(e)}")
            
    # Trier toutes les commandes (PayPal + Cartes) par date
    try:
        context["orders"].sort(key=lambda x: x["date"], reverse=True)
        context["orders"] = context["orders"][:10] # Garder les 10 plus récentes
    except Exception:
        pass

    # 4. Récupérer Échantillon Cartes Cadeaux (Populaires)
    try:
        log("Querying 'giftcards' table...")
        gift_rows = _supabase_get_first_success(
            "giftcards",
            variants=[
                {
                    "select": "name,brand,points", 
                    "popular": "eq.true",
                    "limit": "5"
                },
                {
                    "select": "name,brand,points",
                    "limit": "5"
                }
            ],
            timeout_s=5
        )
        for row in gift_rows:
            if isinstance(row, dict):
                context["giftcards_sample"].append({
                    "name": _safe_str(row.get("name")),
                    "brand": _safe_str(row.get("brand")),
                    "points": safe_int(row.get("points"), 0)
                })
    except Exception as e:
        log(f"Error fetching giftcards: {str(e)}")

    # 5. Récupérer Stats GrattoFolie (Dernières parties)
    try:
        log("Querying 'scratch_results' table (GrattoFolie)...")
        scratch_rows = _supabase_get_first_success(
            "scratch_results",
            variants=[
                {
                    "select": "gained_points,scratched_at", 
                    "user_id": f"eq.{user_id}", 
                    "order": "scratched_at.desc", 
                    "limit": "5"
                }
            ],
            timeout_s=5
        )
        log(f"Found {len(scratch_rows)} scratch results.")
        
        context["grattofolie"] = []
        for row in scratch_rows:
            if isinstance(row, dict):
                pts = safe_int(row.get("gained_points"), 0)
                date = _safe_str(row.get("scratched_at"))
                context["grattofolie"].append({
                    "points": pts,
                    "date": date
                })
                
    except Exception as e:
        log(f"Error fetching scratch results: {str(e)}")
        context["grattofolie"] = []

    # 6. Récupérer Stats MemoRush (Dernières parties)
    try:
        log("Querying 'memory_game_results' table (MemoRush)...")
        memo_rows = _supabase_get_first_success(
            "memory_game_results",
            variants=[
                {
                    "select": "gained_points,moves,time_seconds,created_at", 
                    "user_id": f"eq.{user_id}", 
                    "order": "created_at.desc", 
                    "limit": "5"
                }
            ],
            timeout_s=5
        )
        log(f"Found {len(memo_rows)} memorush results.")
        
        context["memorush"] = []
        for row in memo_rows:
            if isinstance(row, dict):
                pts = safe_int(row.get("gained_points"), 0)
                moves = safe_int(row.get("moves"), 0)
                time_sec = safe_int(row.get("time_seconds"), 0)
                date = _safe_str(row.get("created_at"))
                context["memorush"].append({
                    "points": pts,
                    "moves": moves,
                    "time": time_sec,
                    "date": date
                })
                
    except Exception as e:
        log(f"Error fetching memorush results: {str(e)}")
        context["memorush"] = []

    # 7. Récupérer Stats ShooterRush (Progression + Dernières parties)
    try:
        log("Querying 'shooter_progress' table...")
        prog_rows = _supabase_get_first_success(
            "shooter_progress",
            variants=[
                {"select": "current_level,best_level", "user_id": f"eq.{user_id}", "limit": "1"}
            ],
            timeout_s=5
        )
        
        shooter_context = {"level": 1, "best_level": 1, "history": []}
        if prog_rows:
            p = prog_rows[0]
            shooter_context["level"] = safe_int(p.get("current_level"), 1)
            shooter_context["best_level"] = safe_int(p.get("best_level"), 1)
            
        log("Querying 'shooterush_results' table...")
        shooter_results = _supabase_get_first_success(
            "shooterush_results",
            variants=[
                {
                    "select": "score,points,level,won,created_at", 
                    "user_id": f"eq.{user_id}", 
                    "order": "created_at.desc", 
                    "limit": "5"
                },
                # Fallback au cas où la colonne date s'appelle played_at (vu dans le service)
                {
                    "select": "score,points,level,won,played_at", 
                    "user_id": f"eq.{user_id}", 
                    "order": "played_at.desc", 
                    "limit": "5"
                }
            ],
            timeout_s=5
        )
        
        for row in shooter_results:
            if isinstance(row, dict):
                score = safe_int(row.get("score"), 0)
                points = safe_int(row.get("points"), 0)
                lvl = safe_int(row.get("level"), 1)
                won = row.get("won")
                date = _safe_str(row.get("created_at") or row.get("played_at"))
                
                shooter_context["history"].append({
                    "score": score,
                    "points": points,
                    "level": lvl,
                    "won": won,
                    "date": date
                })
        
        context["shooterush"] = shooter_context
        log(f"Found ShooterRush progress: Lvl {shooter_context['level']} (Best: {shooter_context['best_level']}) and {len(shooter_context['history'])} games.")

    except Exception as e:
        log(f"Error fetching ShooterRush data: {str(e)}")
        context["shooterush"] = None

    # 8. Récupérer Stats BlockPuzzle (Progression + Dernières parties)
    try:
        log("Querying 'block_puzzle_progress' table...")
        block_prog_rows = _supabase_get_first_success(
            "block_puzzle_progress",
            variants=[
                {"select": "best_score", "user_id": f"eq.{user_id}", "limit": "1"}
            ],
            timeout_s=5
        )
        
        block_context = {"best_score": 0, "history": []}
        if block_prog_rows:
            p = block_prog_rows[0]
            block_context["best_score"] = safe_int(p.get("best_score"), 0)
            
        log("Querying 'block_puzzle_games' table...")
        block_results = _supabase_get_first_success(
            "block_puzzle_games",
            variants=[
                {
                    "select": "score,points_earned,created_at", 
                    "user_id": f"eq.{user_id}", 
                    "order": "created_at.desc", 
                    "limit": "5"
                }
            ],
            timeout_s=5
        )
        
        for row in block_results:
            if isinstance(row, dict):
                score = safe_int(row.get("score"), 0)
                points = safe_int(row.get("points_earned"), 0)
                date = _safe_str(row.get("created_at"))
                
                block_context["history"].append({
                    "score": score,
                    "points": points,
                    "date": date
                })
        
        context["blockpuzzle"] = block_context
        log(f"Found BlockPuzzle progress: Best Score: {block_context['best_score']} and {len(block_context['history'])} games.")

    except Exception as e:
        log(f"Error fetching BlockPuzzle data: {str(e)}")
        context["blockpuzzle"] = None

    # 9. Récupérer Stats Plinko (Dernières parties)
    try:
        log("Querying 'plinko_results' table...")
        plinko_results = _supabase_get_first_success(
            "plinko_results",
            variants=[
                {
                    "select": "points_won,balls_played,played_at", 
                    "user_id": f"eq.{user_id}", 
                    "order": "played_at.desc", 
                    "limit": "5"
                }
            ],
            timeout_s=5
        )
        
        context["plinko"] = []
        for row in plinko_results:
            if isinstance(row, dict):
                pts = safe_int(row.get("points_won"), 0)
                balls = safe_int(row.get("balls_played"), 1)
                date = _safe_str(row.get("played_at"))
                
                context["plinko"].append({
                    "points": pts,
                    "balls": balls,
                    "date": date
                })
        
        log(f"Found {len(context['plinko'])} Plinko games.")

    except Exception as e:
        log(f"Error fetching Plinko data: {str(e)}")
        context["plinko"] = []

    # 10. Récupérer Stats Poker (Progression)
    try:
        log("Querying 'poker_progress' table...")
        poker_prog_rows = _supabase_get_first_success(
            "poker_progress",
            variants=[
                {"select": "level,total_winnings,hands_won,games_played", "user_id": f"eq.{user_id}", "limit": "1"}
            ],
            timeout_s=5
        )
        
        context["poker"] = None
        if poker_prog_rows:
            p = poker_prog_rows[0]
            context["poker"] = {
                "level": safe_int(p.get("level"), 1),
                "winnings": safe_int(p.get("total_winnings"), 0),
                "hands_won": safe_int(p.get("hands_won"), 0),
                "games_played": safe_int(p.get("games_played"), 0)
            }
            log(f"Found Poker stats: Lvl {context['poker']['level']}, Winnings: {context['poker']['winnings']}")
        else:
            log("No Poker stats found.")

    except Exception as e:
        log(f"Error fetching Poker data: {str(e)}")
        context["poker"] = None

    # 11. Récupérer Stats SkyCatcher (Progression + Dernières parties)
    try:
        log("Querying 'sky_catcher_progress' table...")
        sky_prog_rows = _supabase_get_first_success(
            "sky_catcher_progress",
            variants=[
                {"select": "current_level,best_level", "user_id": f"eq.{user_id}", "limit": "1"}
            ],
            timeout_s=5
        )
        
        sky_context = {"level": 1, "best_level": 1, "history": []}
        if sky_prog_rows:
            p = sky_prog_rows[0]
            sky_context["level"] = safe_int(p.get("current_level"), 1)
            sky_context["best_level"] = safe_int(p.get("best_level"), 1)
            
        log("Querying 'sky_catcher_results' table...")
        sky_results = _supabase_get_first_success(
            "sky_catcher_results",
            variants=[
                {
                    "select": "score,points,level,won,created_at", 
                    "user_id": f"eq.{user_id}", 
                    "order": "created_at.desc", 
                    "limit": "5"
                }
            ],
            timeout_s=5
        )
        
        for row in sky_results:
            if isinstance(row, dict):
                score = safe_int(row.get("score"), 0)
                points = safe_int(row.get("points"), 0)
                lvl = safe_int(row.get("level"), 1)
                won = row.get("won")
                date = _safe_str(row.get("created_at"))
                
                sky_context["history"].append({
                    "score": score,
                    "points": points,
                    "level": lvl,
                    "won": won,
                    "date": date
                })
        
        context["skycatcher"] = sky_context
        log(f"Found SkyCatcher progress: Lvl {sky_context['level']} (Best: {sky_context['best_level']}) and {len(sky_context['history'])} games.")

    except Exception as e:
        log(f"Error fetching SkyCatcher data: {str(e)}")
        context["skycatcher"] = None

    # 12. Récupérer Historique des Sondages (via transaction_history)
    try:
        log("Querying 'transaction_history' for surveys...")
        survey_rows = _supabase_get_first_success(
            "transaction_history",
            variants=[
                # Variante 1: Filtrer par source contenant 'cpx' ou 'rapido' ou 'survey'
                {
                    "select": "source,points,created_at,status", 
                    "user_id": f"eq.{user_id}", 
                    "or": "(source.ilike.*cpx*,source.ilike.*rapido*,source.ilike.*survey*)",
                    "order": "created_at.desc", 
                    "limit": "5"
                },
                # Variante 2: Si le filtre OR échoue, prendre tout et filtrer en Python (moins efficace mais plus sûr)
                {
                    "select": "source,points,created_at,status",
                    "user_id": f"eq.{user_id}",
                    "order": "created_at.desc",
                    "limit": "20"
                }
            ],
            timeout_s=5
        )
        
        context["surveys"] = []
        if survey_rows:
            for row in survey_rows:
                if isinstance(row, dict):
                    source = _safe_str(row.get("source")).lower()
                    # Filtrage côté Python si la requête SQL large a été utilisée
                    if "cpx" in source or "rapido" in source or "survey" in source or "sondage" in source:
                        pts = safe_int(row.get("points"), 0)
                        date = _safe_str(row.get("created_at"))
                        status = _safe_str(row.get("status") or "completed")
                        
                        # Formatter le nom proprement
                        provider = "Sondage"
                        if "cpx" in source: provider = "CPX Research"
                        elif "rapido" in source: provider = "RapidoReach"
                        
                        context["surveys"].append({
                            "provider": provider,
                            "points": pts,
                            "date": date,
                            "status": status
                        })
                        if len(context["surveys"]) >= 5: break # Limite à 5 résultats pertinents
        
        log(f"Found {len(context['surveys'])} survey transactions.")

    except Exception as e:
        log(f"Error fetching survey transactions: {str(e)}")
        context["surveys"] = []

    try:
        log("Querying 'transaction_offers' table (offerwall)...")
        offer_tx_rows = _supabase_get_first_success(
            "transaction_offers",
            variants=[
                {
                    "select": "offer_id,offer_name,event_name,type,points,status,created_at",
                    "user_id": f"eq.{user_id}",
                    "order": "created_at.desc",
                    "limit": "10",
                }
            ],
            timeout_s=5,
        )

        context["offers"] = []
        for row in offer_tx_rows:
            if isinstance(row, dict):
                status_raw = row.get("status")
                status = "completed" if safe_int(status_raw, 0) == 1 else _safe_str(status_raw or "pending")
                context["offers"].append(
                    {
                        "offer_id": _safe_str(row.get("offer_id")),
                        "name": _safe_str(row.get("offer_name") or "Offre"),
                        "event": _safe_str(row.get("event_name") or row.get("type") or ""),
                        "type": _safe_str(row.get("type") or ""),
                        "points": safe_int(row.get("points"), 0),
                        "status": status,
                        "date": _safe_str(row.get("created_at")),
                    }
                )

        log("Querying 'offer_tracking' table (offerwall)...")
        tracking_rows = _supabase_get_first_success(
            "offer_tracking",
            variants=[
                {
                    "select": "offer_id,action_type,status,created_at,metadata",
                    "user_id": f"eq.{user_id}",
                    "order": "created_at.desc",
                    "limit": "10",
                }
            ],
            timeout_s=5,
        )

        context["offers_tracking"] = []
        for row in tracking_rows:
            if isinstance(row, dict):
                meta = row.get("metadata") or {}
                offer_name = ""
                if isinstance(meta, dict):
                    offer_name = _safe_str(meta.get("offer_name"))
                context["offers_tracking"].append(
                    {
                        "offer_id": _safe_str(row.get("offer_id")),
                        "name": offer_name or "Offre",
                        "action": _safe_str(row.get("action_type") or ""),
                        "status": _safe_str(row.get("status") or ""),
                        "date": _safe_str(row.get("created_at")),
                    }
                )

        log(
            f"Found {len(context['offers'])} offer transactions and {len(context['offers_tracking'])} offer tracking rows."
        )
    except Exception as e:
        log(f"Error fetching offerwall data: {str(e)}")
        context["offers"] = []
        context["offers_tracking"] = []
        
    return context


@app.post("/support/chat")
async def support_chat(request: Request):
    try:
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid JSON")
            
        user_id = _safe_str(body.get("user_id")).strip()
        message = _safe_str(body.get("message")).strip()
        session_id = _safe_str(body.get("session_id")).strip()
        attachments = body.get("attachments")
        locale = _safe_str(body.get("locale")).strip()
        
        if not user_id or not message:
            # Fallback si user_id manque (mode démo ou déconnecté)
            if message and not user_id:
                user_id = "anonymous"
            else:
                raise HTTPException(status_code=400, detail="user_id and message required")

        reply_lang = _support_detect_user_language(message, locale)
        resolved_session_id = _support_get_or_create_session(user_id, session_id)
        if resolved_session_id:
            try:
                _supabase_post(
                    "support_chat_session_messages",
                    {
                        "session_id": resolved_session_id,
                        "user_id": user_id,
                        "sender": "user",
                        "message": message,
                        "attachments": attachments if isinstance(attachments, list) else None,
                    },
                    timeout_s=5,
                )
            except Exception:
                pass

        try:
            wants_human = False
            if resolved_session_id:
                wants_human = _support_should_request_human(message) or (
                    _support_is_yes(message) and _support_last_bot_asked_human_session(resolved_session_id)
                )

            if resolved_session_id and wants_human:
                if _support_open_human_ticket_count(user_id) >= 3:
                    text = _support_i18n(reply_lang, "ticket_limit")
                    return {
                        "response": text,
                        "source": "human_ticket_error",
                        "session_id": resolved_session_id,
                        "suppress_bot_message": True,
                    }
                try:
                    allowed = _supabase_rpc(
                        "can_open_human_support_ticket",
                        {"p_user_id": user_id},
                        timeout_s=5,
                    )
                    if allowed is False:
                        raise RuntimeError("ticket_limit_reached")
                    if isinstance(allowed, dict) and allowed.get("can_open") is False:
                        raise RuntimeError("ticket_limit_reached")
                except Exception as e:
                    msg = _safe_str(e)
                    if "ticket_limit_reached" in msg:
                        text = _support_i18n(reply_lang, "ticket_limit")
                    else:
                        text = _support_i18n(reply_lang, "human_unavailable")
                    return {
                        "response": text,
                        "source": "human_ticket_error",
                        "session_id": resolved_session_id,
                        "suppress_bot_message": True,
                    }

                subject = _support_suggest_subject_from_ticket("", message)
                if resolved_session_id:
                    try:
                        rows = _supabase_get(
                            "support_chat_session_messages",
                            params={
                                "select": "sender,message,created_at",
                                "session_id": f"eq.{resolved_session_id}",
                                "order": "created_at.asc",
                                "limit": "50",
                            },
                            timeout_s=5,
                        )
                        combined = " | ".join([_safe_str(r.get("message")) for r in rows if isinstance(r, dict)])
                        subject = _support_infer_ticket_subject(combined) if combined else subject
                    except Exception:
                        pass

                ticket_rows = _supabase_post(
                    "support_chat_tickets",
                    {
                        "user_id": user_id,
                        "subject": subject,
                        "status": "ouvert",
                        "priority": "normale",
                        "needs_human": True,
                    },
                    timeout_s=5,
                )
                new_ticket_id = _safe_str(ticket_rows[0].get("id")) if ticket_rows and isinstance(ticket_rows[0], dict) else ""

                if new_ticket_id and resolved_session_id:
                    try:
                        session_msgs = _supabase_get(
                            "support_chat_session_messages",
                            params={
                                "select": "sender,message,created_at,attachments",
                                "session_id": f"eq.{resolved_session_id}",
                                "order": "created_at.asc",
                                "limit": "200",
                            },
                            timeout_s=5,
                        )
                        for r in session_msgs:
                            if not isinstance(r, dict):
                                continue
                            sender = _safe_str(r.get("sender")).lower()
                            msg = _safe_str(r.get("message"))
                            atts = r.get("attachments")
                            att_list = (atts if isinstance(atts, list) else [])
                            if not msg:
                                continue
                            _supabase_post(
                                "support_chat_ticket_messages",
                                {
                                    "ticket_id": new_ticket_id,
                                    "user_id": user_id,
                                    "message": msg,
                                    "is_admin": True if sender == "bot" else False,
                                    "sender": "bot" if sender == "bot" else "user",
                                    "attachments": att_list if att_list else [],
                                },
                                timeout_s=5,
                            )
                    except Exception:
                        pass

                if new_ticket_id:
                    try:
                        _supabase_post(
                            "support_chat_ticket_messages",
                            {
                                "ticket_id": new_ticket_id,
                                "user_id": user_id,
                                "message": _support_i18n(reply_lang, "human_opened"),
                                "is_admin": True,
                                "sender": "bot",
                                "attachments": [],
                            },
                            timeout_s=5,
                        )
                    except Exception:
                        pass

                if resolved_session_id:
                    try:
                        _supabase_patch(
                            "support_chat_sessions",
                            match={"id": f"eq.{resolved_session_id}", "user_id": f"eq.{user_id}"},
                            payload={"status": "closed"},
                            timeout_s=5,
                        )
                    except Exception:
                        pass

                return {
                    "response": _support_i18n(reply_lang, "human_opened"),
                    "source": "human_ticket_opened",
                    "ticket_id": new_ticket_id,
                    "suppress_bot_message": True,
                }
        except Exception:
            pass
            
        # Récupération contexte
        ctx = _get_support_context(user_id)
        
        # Construction Prompt
        tx_summary = "Aucune offre récente."
        if ctx["transactions"]:
            lines = []
            for t in ctx["transactions"]:
                lines.append(f"- [Offre] {t['date'][:10]}: {t['title']} ({t['provider']}) - {t['status']} - {t['points']} pts")
            tx_summary = "\n".join(lines)
            
        orders_summary = "Aucune commande récente."
        if ctx["orders"]:
            lines = []
            for o in ctx["orders"]:
                lines.append(f"- [Commande] {o['date'][:10]}: {o['name']} ({o['type']}) - {o['status']} - Coût: {o['points']} pts")
            orders_summary = "\n".join(lines)
            
        gifts_summary = ""
        if ctx["giftcards_sample"]:
            gifts_summary = "Exemples de cartes cadeaux dispos : " + ", ".join([f"{g['name']} ({g['points']} pts)" for g in ctx["giftcards_sample"]])
            
        grattofolie_summary = "Aucune partie récente."
        if ctx.get("grattofolie"):
            lines = []
            for g in ctx["grattofolie"]:
                date_str = g['date'][:10]
                lines.append(f"- [GrattoFolie] {date_str}: {g['points']} pts gagnés")
            grattofolie_summary = "\n".join(lines)
            
        memorush_summary = "Aucune partie récente."
        if ctx.get("memorush"):
            lines = []
            for m in ctx["memorush"]:
                date_str = m['date'][:10]
                lines.append(f"- [MemoRush] {date_str}: {m['points']} pts ({m['moves']} coups, {m['time']}s)")
            memorush_summary = "\n".join(lines)
            
        shooterush_summary = "Aucune partie récente."
        shooter_level_info = ""
        if ctx.get("shooterush"):
            shooter_data = ctx["shooterush"]
            shooter_level_info = f"(Niveau actuel: {shooter_data['level']}, Record: {shooter_data['best_level']})"
            if shooter_data.get("history"):
                lines = []
                for s in shooter_data["history"]:
                    date_str = s['date'][:10]
                    status = "Gagné" if s['won'] else "Perdu"
                    lines.append(f"- [ShooterRush] {date_str}: {status} - {s['score']} score - {s['points']} pts")
                shooterush_summary = "\n".join(lines)
                
        blockpuzzle_summary = "Aucune partie récente."
        block_best_score = ""
        if ctx.get("blockpuzzle"):
            block_data = ctx["blockpuzzle"]
            block_best_score = f"(Record: {block_data['best_score']})"
            if block_data.get("history"):
                lines = []
                for b in block_data["history"]:
                    date_str = b['date'][:10]
                    lines.append(f"- [BlockPuzzle] {date_str}: {b['score']} score - {b['points']} pts")
                blockpuzzle_summary = "\n".join(lines)
                
        plinko_summary = "Aucune partie récente."
        if ctx.get("plinko"):
            lines = []
            for p in ctx["plinko"]:
                date_str = p['date'][:10]
                lines.append(f"- [Plinko] {date_str}: {p['points']} pts ({p['balls']} billes)")
            plinko_summary = "\n".join(lines)
            
        poker_summary = "Aucune donnée de Poker."
        if ctx.get("poker"):
            p = ctx["poker"]
            poker_summary = f"Niveau: {p['level']}, Gains totaux: {p['winnings']} pts, Mains gagnées: {p['hands_won']}, Parties jouées: {p['games_played']}"
            
        skycatcher_summary = "Aucune partie récente."
        sky_level_info = ""
        if ctx.get("skycatcher"):
            sky_data = ctx["skycatcher"]
            sky_level_info = f"(Niveau actuel: {sky_data['level']}, Record: {sky_data['best_level']})"
            if sky_data.get("history"):
                lines = []
                for s in sky_data["history"]:
                    date_str = s['date'][:10]
                    status = "Gagné" if s['won'] else "Perdu"
                    lines.append(f"- [SkyCatcher] {date_str}: {status} - {s['score']} score - {s['points']} pts")
                skycatcher_summary = "\n".join(lines)
                
        surveys_summary = "Aucun sondage récent."
        if ctx.get("surveys"):
            lines = []
            for s in ctx["surveys"]:
                date_str = s['date'][:10]
                lines.append(f"- [{s['provider']}] {date_str}: {s['points']} pts ({s['status']})")
            surveys_summary = "\n".join(lines)

        offers_summary = "Aucune offre récente."
        offer_lines = []
        if ctx.get("offers_tracking"):
            for o in ctx["offers_tracking"][:3]:
                date_str = (o.get("date") or "")[:10]
                name = o.get("name") or "Offre"
                offer_lines.append(f"- [Offre] {date_str}: {name} - in_progress")
        if ctx.get("offers"):
            for o in ctx["offers"]:
                if len(offer_lines) >= 5:
                    break
                date_str = (o.get("date") or "")[:10]
                name = o.get("name") or "Offre"
                status = o.get("status") or "pending"
                pts = o.get("points") or 0
                offer_lines.append(f"- [Offre] {date_str}: {name} - {status} - {pts} pts")
        if offer_lines:
            offers_summary = "\n".join(offer_lines)
            
        # Debug logs pour l'admin
        debug_logs_str = "\n".join([f"[LOG] {l}" for l in ctx.get("debug_logs", [])])
            
        conv_history = ""
        user_turns = 0
        if resolved_session_id:
            conv_history, user_turns = _support_get_session_history(resolved_session_id, limit=12)
        is_first_turn = user_turns <= 1

        system_prompt = (
            f"Tu es le bot support de GiftPlayz. L'utilisateur {ctx.get('username', 'Inconnu')} (Niveau {ctx['level']}, {ctx['points']} pts) te parle.\n\n"
            f"Conversation en cours (dernier échanges) :\n{conv_history or 'Aucun historique.'}\n\n"
            f"Activité Récente :\n{tx_summary}\n\n"
            f"Commandes Boutique (Virements & Cartes) :\n{orders_summary}\n\n"
            f"Dernières parties GrattoFolie :\n{grattofolie_summary}\n\n"
            f"Dernières parties MemoRush :\n{memorush_summary}\n\n"
            f"Progression ShooterRush {shooter_level_info}:\n{shooterush_summary}\n\n"
            f"Progression BlockPuzzle {block_best_score}:\n{blockpuzzle_summary}\n\n"
            f"Dernières parties Plinko :\n{plinko_summary}\n\n"
            f"Stats Poker Texas Hold'em :\n{poker_summary}\n\n"
            f"Progression SkyCatcher {sky_level_info}:\n{skycatcher_summary}\n\n"
            f"Derniers Sondages Complétés :\n{surveys_summary}\n\n"
            f"Dernières Offres (TimeWall/Offerwall) :\n{offers_summary}\n\n"
            f"Boutique :\n{gifts_summary}\n\n"
            "Fonctionnalités Connues (ne pas inventer) :\n"
            "• Connexion : page Login avec lien **Mot de passe oublié ?**\n"
            "• Réinitialisation mot de passe : pages **/forgot-password** et **/reset-password** via email.\n"
            "• Changer mot de passe (si connecté) : dans le **Profil**, l'utilisateur peut modifier son mot de passe.\n\n"
            "Règles :\n"
            "1. Sois courtois, empathique et concis.\n"
            "2. Si l'utilisateur demande où sont ses points d'OFFRE, regarde 'Activité Récente'. Si 'pending', explique le délai (24-48h).\n"
            "3. VIREMENTS PAYPAL : Si 'pending', explique que le traitement peut prendre jusqu'à une semaine.\n"
            "4. CARTES CADEAUX : La livraison est généralement instantanée. Si ce n'est pas le cas (statut 'pending' ou 'processing'), explique que notre prestataire est temporairement en rupture de stock et que le système réessaie automatiquement de passer la commande chaque jour.\n"
            "5. SI L'UTILISATEUR PARLE DE COMMANDES EN COURS/NON REÇUES : Ne liste QUE les commandes avec le statut 'pending', 'processing' ou 'disputed'. Ignore les commandes 'delivered' ou 'cancelled' sauf si l'utilisateur demande explicitement l'historique complet.\n"
            "6. GRATTOFOLIE : Si l'utilisateur demande s'il a gagné récemment à GrattoFolie, regarde 'Dernières parties GrattoFolie'. Explique que c'est un jeu de grattage quotidien.\n"
            "7. MEMORUSH : Si l'utilisateur parle de 'jeu de mémoire' ou 'MemoRush', regarde 'Dernières parties MemoRush'. C'est un jeu où il faut trouver les paires de cartes.\n"
            "8. SHOOTERRUSH : Si l'utilisateur parle de 'jeu de tir', 'ShooterRush' ou 'Shooter', regarde 'Progression ShooterRush'. C'est un jeu d'arcade où il faut détruire des ennemis. Mentionne son niveau actuel et son meilleur niveau s'il demande sa progression.\n"
            "9. BLOCKPUZZLE : Si l'utilisateur parle de 'BlockPuzzle' ou 'jeu de blocs' (style Tetris), regarde 'Progression BlockPuzzle'. C'est un jeu de réflexion. Mentionne son record de score.\n"
            "10. PLINKO : Si l'utilisateur parle de 'Plinko' ou 'jeu de billes', regarde 'Dernières parties Plinko'. C'est un jeu de chance où on lâche des billes pour gagner des points.\n"
            "11. POKER : Si l'utilisateur parle de 'Poker', regarde 'Stats Poker'. C'est du Texas Hold'em. Mentionne ses gains totaux et son niveau.\n"
            "12. SKYCATCHER : Si l'utilisateur parle de 'SkyCatcher' ou 'jeu d'attrape', regarde 'Progression SkyCatcher'. C'est un jeu où on attrape des objets qui tombent. Mentionne son niveau actuel.\n"
            "13. SONDAGES & OFFRES :\n"
            "    - Nous proposons des sondages via **CPX Research** et **RapidoReach**.\n"
            "    - Pour les offres (murs d'offres), nous travaillons avec **TimeWall**.\n"
            "    - Si l'utilisateur parle de sondages, regarde 'Derniers Sondages Complétés' pour voir s'il a reçu des points récemment.\n"
            "    - Si l'utilisateur parle d'offres, regarde 'Dernières Offres (TimeWall/Offerwall)' pour voir si une conversion est en cours ou validée.\n"
            "    - Si un utilisateur a un problème avec une offre spécifique, conseille-lui de contacter directement le support du mur d'offre concerné (ex: Support TimeWall) car nous n'avons pas la main dessus.\n"
            "    - RÈGLES GÉNÉRALES SONDAGES :\n"
            "      • Les sondages dépendent de partenaires externes. Être invité ne garantit pas d'être accepté jusqu'à la fin (disqualification possible si profil non correspondant).\n"
            "      • Une récompense peut être en attente de validation. Elle peut être annulée en cas de réponse invalide ou frauduleuse.\n"
            "      • INTERDIT : VPN, proxy, multi-comptes, fausses infos, réponses trop rapides/aléatoires.\n"
            "      • En cas de souci, demander : date, heure, fournisseur, montant et capture d'écran.\n"
            "    - RÈGLES CPX RESEARCH :\n"
            "      • Exige des infos vraies et à jour. Un seul compte par personne/foyer.\n"
            "      • Refus possible si VPN, réponses rapides ou double inscription.\n"
            "      • Problème de tracking ? Signaler rapidement (max 7 jours ouvrés).\n"
            "    - RÈGLES RAPIDOREACH :\n"
            "      • La disponibilité varie selon le profil et le pays.\n"
            "      • Les erreurs techniques peuvent venir de l'initialisation ou de l'affichage.\n"
            "    - RÈGLES GÉNÉRALES OFFRES (TimeWall et autres) :\n"
            "      • Les offres dépendent de partenaires externes. Une récompense n'est JAMAIS garantie tant qu'elle n'est pas confirmée/validée.\n"
            "      • L'utilisateur doit lancer l'offre depuis notre site, respecter toutes les conditions, et rester sur le même appareil/pays/compte pendant toute l'offre.\n"
            "      • Tracking : autoriser le suivi publicitaire, éviter les adblockers agressifs, ne pas utiliser VPN/proxy/émulateur, ne pas avoir déjà installé l'app.\n"
            "      • Un seul compte par utilisateur (multi-comptes, automatisation, comportements suspects = refus/annulation possibles).\n"
            "      • En cas de crédit manquant, demander : prestataire, nom/ID de l'offre, date+heure, montant attendu, appareil, pays, capture d'écran + preuve de l'étape atteinte.\n"
            "      • Si c'est complexe/non vérifiable, rediriger vers un agent humain. Ne jamais promettre un crédit manuel.\n"
            "    - RÈGLES TIMEWALL :\n"
            "      • TimeWall applique des règles strictes (un seul compte, pas de VPN/proxy, IP/appareil déjà utilisé = possible refus).\n"
            "      • Certains crédits peuvent prendre jusqu'à 2 semaines. Certaines tâches ont un système de dispute, mais pas toutes.\n"
            "      • Après ~30 jours sans crédit, conseiller de contacter TimeWall avec l'ID de l'offre et les preuves.\n"
            "    - AUTRES PRESTATAIRES :\n"
            "      • BitcoTasks : statut approuvé/rejeté/dispute selon les tâches ; vérifier l'historique et la raison du rejet.\n"
            "      • Notik : tracking lié à l'appareil/navigateur ; rester sur le même appareil et ne pas bloquer le suivi.\n"
            "      • Offery : validation selon pays/appareil/anti-fraude ; confirmation requise avant crédit.\n"
            "14. Tu ne PEUX PAS créditer de points ni valider de commandes manuellement.\n"
            f"15. {('PREMIER MESSAGE : ne propose PAS un agent humain si l’utilisateur ne le demande pas explicitement. Concentre-toi d’abord sur 1-2 questions de diagnostic + étapes concrètes. ' if is_first_turn else '')}Si tu ne peux pas résoudre, propose simplement : « Je peux vous mettre en relation avec un agent humain. Souhaitez-vous ? » (sans parler de ticket, sans parler de bouton, sans parler de limite).\n"
            "16. Ne JAMAIS inventer une fonctionnalité (ex: « pas de page », « pas de modal », « bouton ouvrir un ticket »). Si tu n'es pas sûr, dis-le et propose un agent humain.\n"
            "17. Pour un problème de mot de passe :\n"
            "    - Si l'utilisateur n'arrive pas à se connecter : guider vers **Mot de passe oublié ?** (page /forgot-password) puis lien reçu (page /reset-password).\n"
            "    - Si l'utilisateur est connecté et veut changer : guider vers la section mot de passe du **Profil**.\n"
            "18. Réponses courtes : va droit au but. Ne récite pas tout l'historique.\n"
            f"19. Réponds dans la langue de l'utilisateur. Langue attendue: {reply_lang}.\n"
            "20. Si l'utilisateur a joint une image, ne devine pas ce qu'elle contient. Dis que tu l'as reçue et propose un agent humain si nécessaire.\n\n"
            f"--- DEBUG LOGS (Pour info technique seulement, ne pas citer à l'utilisateur sauf s'il demande des détails techniques) ---\n{debug_logs_str}"
        )
        
        # Appel LLM
        llm = _chat_llm(system_prompt=system_prompt, user_prompt=message, max_tokens=350, temperature=0.7)
        
        if not llm:
            # Vérifier l'erreur LLM
            last_error = resources.get("last_llm_error") or "Unknown LLM error"
            print(f"[SUPPORT-CHAT] LLM Error: {last_error}")
            return {
                "response": "Je rencontre une difficulté temporaire pour analyser votre demande. Veuillez réessayer dans quelques instants.",
                "source": "fallback",
                "debug_error": last_error,
                "debug_context": system_prompt
            }
            
        content = llm.get("content", "")
        if isinstance(attachments, list) and len(attachments) > 0:
            content = _support_i18n(reply_lang, "image_received") + "\n\n" + _safe_str(content).strip()
        should_force_human_offer = (not is_first_turn) and (user_turns >= 2) and _support_user_seems_blocked(message)
        if should_force_human_offer and not _support_bot_offered_human_recently(content):
            content = _safe_str(content).rstrip()
            if content:
                content += "\n\n"
            content += _support_i18n(reply_lang, "offer_human")
        if resolved_session_id:
            try:
                _supabase_post(
                    "support_chat_session_messages",
                    {
                        "session_id": resolved_session_id,
                        "user_id": user_id,
                        "sender": "bot",
                        "message": _safe_str(content),
                        "attachments": None,
                    },
                    timeout_s=5,
                )
            except Exception:
                pass

        return {
            "response": content,
            "source": llm.get("source", "llm"),
            "model": llm.get("model", ""),
            "session_id": resolved_session_id,
            "debug_context": system_prompt
        }
    except Exception as e:
        print(f"[SUPPORT-CHAT] Exception: {str(e)}")
        # Renvoyer une réponse valide pour que le frontend l'affiche
        return {
            "response": f"Une erreur technique est survenue: {str(e)}",
            "source": "error"
        }


@app.post("/internal/quiz/generate")
async def internal_quiz_generate(request: Request):
    _require_internal_token(request)

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

    fallback = _fallback_quiz(topic=topic, difficulty=difficulty)
    if isinstance(fallback, dict):
        fallback["llm"] = _llm_status()
        fallback["llm_error"] = resources.get("last_llm_error") or ""
    return fallback
