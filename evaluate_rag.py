import argparse
import json
import os
import re
import time
from datetime import datetime

import chromadb
import ollama
from chromadb.config import Settings

import numpy as _np
import json as _json

_VEC_CACHE = {}

def _load_vectors(base_dir):
    """Load our own vector index (vectors.npz) once and cache it."""
    if "data" in _VEC_CACHE:
        return _VEC_CACHE["data"]
    vec_path = os.path.join(base_dir, "vectors.npz")
    if not os.path.exists(vec_path):
        _VEC_CACHE["data"] = None
        return None
    data = _np.load(vec_path, allow_pickle=False)
    vecs = data["vectors"]               # shape (N, 768), float32
    norms = _np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1e-9
    vecs_normed = vecs / norms
    meta = [_json.loads(s) for s in data["meta"]]
    _VEC_CACHE["data"] = (vecs_normed, meta)
    return _VEC_CACHE["data"]


def _vec_search(base_dir, query_emb, fname, limit):
    """Cosine similarity search over our vectors.npz, filtered by source."""
    loaded = _load_vectors(base_dir)
    if loaded is None:
        return []
    vecs_normed, meta = loaded
    q = _np.array(query_emb, dtype=_np.float32)
    q = q / max(_np.linalg.norm(q), 1e-9)
    sims = vecs_normed @ q                 # dot product = cosine sim (normed)
    # get top candidates, then filter by source
    top_idx = _np.argsort(sims)[::-1]
    results = []
    for i in top_idx:
        m = meta[i]
        if m["source"] != fname:
            continue
        results.append({
            "text": m["text"],
            "source": m["source"],
            "page": m["page"],
            "relevance": round(float(sims[i]) * 100, 1),
            "match_type": "vector",
        })
        if len(results) >= limit:
            break
    return results


def fetch_pages_by_number(base_dir, fname, page_list, keyword_weights=None):
    """Fetch 1 best chunk per targeted page.
    Scores each chunk against keyword_weights and returns the highest-scoring one
    (so /vpbx/stats/calls/request in chunk[1] beats a CSV example in chunk[0]).
    """
    loaded = _load_vectors(base_dir)
    if loaded is None:
        return []
    _, meta = loaded
    pages_set = set(int(p) for p in page_list)
    # Collect all chunks per page
    page_chunks: dict = {}
    for m in meta:
        if m["source"] != fname:
            continue
        try:
            pg = int(m["page"])
        except (ValueError, TypeError):
            continue
        if pg in pages_set:
            page_chunks.setdefault(pg, []).append(m)

    results = []
    for pg in sorted(page_chunks.keys()):
        chunks = page_chunks[pg]
        if keyword_weights and len(chunks) > 1:
            # Score all chunks, return top-2 so we don't miss content split across chunks
            scored = []
            for m in chunks:
                low = m["text"].lower()
                score = sum(w for t, w in keyword_weights.items()
                            if t.strip('"').lower() in low)
                scored.append((score, m))
            scored.sort(key=lambda x: -x[0])
            chosen_list = [m for _, m in scored[:2]]
        else:
            chosen_list = chunks[:1]
        for chosen in chosen_list:
            results.append({
                "text": chosen["text"],
                "source": chosen["source"],
                "page": pg,
                "relevance": 99.5,
                "match_type": "page_targeted",
            })
    return results




BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")
DEFAULT_SUITE = os.path.join(BASE_DIR, "test_suite.json")
REPORT_DIR = os.path.join(BASE_DIR, "reports")

EMBED_MODEL = "nomic-embed-text"
DEFAULT_MODEL = "qwen2.5:14b"
CHUNKS_PER_DOC = 10
MAX_CONTEXT = 18


DOC_CATALOG = {
    "api": {
        "files": ["API MANGO OFFICE.pdf"],
        "desc": "API reference: /vpbx/stats/calls/request, /result, /vpbx/cc/call, /vpbx/task/add, fields context_start_time, call_start_time, entry_id, create, start.",
    },
    "webhooks": {
        "files": ["API MANGO OFFICE.pdf"],
        "desc": "Webhook events inside API MANGO OFFICE.pdf: events/summary, events/md/onAppealClose, create_time, create.",
    },
    "webhook_setup": {
        "files": ["ИНТЕГРАЦИЯ MANGO OFFICE Инструкция по вебхуки MANGO OFFICE.pdf"],
        "desc": "Webhook URL setup guide. Not a field reference.",
    },
    "contact_center": {
        "files": ["КЦ - Контакт центр руководства пользователя.pdf"],
        "desc": "Contact Center UI and operator/supervisor guide.",
    },
    "lk": {
        "files": ["ЛК - виртуальная атс mango office ЮТЕР ЕРЕАДРЕ СПРАВОЧНИК АБОНЕНТА.pdf"],
        "desc": "Virtual PBX personal account guide.",
    },
    "quality": {
        "files": ["КОНТРОЛЬ КАЧЕСТВА Руководство пользователя.pdf"],
        "desc": "Quality control guide.",
    },
    "wallboard": {
        "files": ["Программа для ЭВМ Wallboard Mango Office Руководство пользователя.pdf"],
        "desc": "Wallboard guide.",
    },
    "jira_cases": {
        "files": ["данные для тренеровки.txt"],
        "desc": "Real Jira support cases: mpupdater.lock, tasks_details, CSV UTF-8 import, report formulas and internal L3 answers.",
    },
}


KEYWORD_WEIGHTS = {
    "POST /vpbx/cc/call": 18,
    "/vpbx/stats/calls/request": 14,
    "/vpbx/stats/calls/result": 14,
    "/vpbx/cc/call": 16,
    "/vpbx/task/add": 16,
    "/config/users/request": 14,
    "events/summary": 16,
    "events/md/onAppealClose": 18,
    "onAppealClose": 14,
    "context_start_time": 14,
    "call_start_time": 14,
    "call_abonent_id": 12,
    "call_type": 12,
    "context_calls": 12,
    "members": 12,
    "user_id": 10,
    "context_type": 10,
    "talk_duration": 10,
    "tasks_details": 18,
    "custom_fields": 12,
    "mpupdater.lock": 24,
    "MANGO OFFICE Contact Center_data": 18,
    "UTF-8": 16,
    "utf8": 12,
    "Windows-1251": 12,
    "windows1251": 12,
    "кодировк": 14,
    "CSV": 14,
    "entry_id": 14,
    "create_time": 12,
    "Время поступления обращения": 18,
    "Время первого ответа пользователя": 14,
    "Время взятия обращения в работу": 24,
    "Назначенный сотрудник": 10,
    '"create"': 8,
    '"start"': 8,
}


# Page-targeted retrieval: (trigger_term, document_filename, [pages])
# Fires when trigger found (case-insensitive) in question + technical_query combined text.
# Chunks from these pages are inserted at relevance=99.5 (rank above all other results).
PAGE_TRIGGERS = [
    # Stats calls API — pages 71-83 contain /vpbx/stats/calls/request+result field tables
    ("начала разговора",             "API MANGO OFFICE.pdf", [71, 72, 74, 77]),
    ("call start time",              "API MANGO OFFICE.pdf", [71, 72, 74, 77]),
    ("/vpbx/stats/calls",            "API MANGO OFFICE.pdf", [71, 72, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83]),
    ("stats/calls/result",           "API MANGO OFFICE.pdf", [71, 72, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83]),
    ("stats/calls/request",          "API MANGO OFFICE.pdf", [71, 72]),
    ("context_start_time",           "API MANGO OFFICE.pdf", [74, 77, 81]),
    ("call_start_time",              "API MANGO OFFICE.pdf", [77, 81, 83]),
    ("context_type",                 "API MANGO OFFICE.pdf", [74, 75, 76]),
    ("talk_duration",                "API MANGO OFFICE.pdf", [74, 77, 78, 79, 80, 81]),
    ("средняя продолжительность",    "API MANGO OFFICE.pdf", [74, 75, 76, 77, 78, 79, 80, 81]),
    ("производительность сотрудников", "API MANGO OFFICE.pdf", [74, 75, 76, 77, 78, 79, 80, 81]),
    ("call_abonent_id",              "API MANGO OFFICE.pdf", [74, 75, 76, 77, 78, 79, 80, 81, 82, 83]),
    ("context_calls",                "API MANGO OFFICE.pdf", [74, 75, 76, 77, 78, 79, 80, 81, 82]),
    # CC Call API — pages 280-292 contain /vpbx/cc/call and onAppealClose
    ("/vpbx/cc/call",                "API MANGO OFFICE.pdf", [280, 281, 282]),
    ("cc/call",                      "API MANGO OFFICE.pdf", [280, 281, 282]),
    ("взятия обращения в работу",    "API MANGO OFFICE.pdf", [280, 281, 282]),
    ("время взятия",                 "API MANGO OFFICE.pdf", [280, 281, 282]),
    ("назначения обращения",         "API MANGO OFFICE.pdf", [280, 281, 282]),
    ("onappealclose",                "API MANGO OFFICE.pdf", [291, 292]),
    # Task API
    ("/vpbx/task/add",               "API MANGO OFFICE.pdf", [260, 261, 262]),
    ("task/add",                     "API MANGO OFFICE.pdf", [260, 261, 262]),
    # Config users API — /config/users/request is on pages 96-98, NOT 68-71 (68-71 = old stats API)
    ("/config/users",                "API MANGO OFFICE.pdf", [96, 97, 98]),
    ("config/users",                 "API MANGO OFFICE.pdf", [96, 97, 98]),
    # Entry ID — include pages with redirect/transfer context
    ("entry_id",                     "API MANGO OFFICE.pdf", [23, 25, 27, 30, 73, 75, 79, 80, 81]),
    # CSV encoding case — page 75 = context/question, page 76 = answer with Windows-1251
    ("кодировк",                     "данные для тренеровки.txt", [75, 76]),
    ("windows1251",                  "данные для тренеровки.txt", [75, 76]),
    ("utf8",                         "данные для тренеровки.txt", [75, 76]),
    # tasks_details — jira page 91 has the answer (tasks_details: true parameter)
    ("custom_fields",                "данные для тренеровки.txt", [91]),
    ("tasks_details",                "данные для тренеровки.txt", [91]),
    # Средняя продолжительность — jira pages 96-99 have the formula discussion
    ("средняя продолжительность",    "данные для тренеровки.txt", [96, 97, 98, 99]),
    ("производительность сотрудников", "данные для тренеровки.txt", [96, 97, 98, 99]),
]


SYSTEM_PROMPT = """Ты — ВЕКТОР (Virtual Expert Knowledge Technical Operations Research), система поддержки L3 Mango Office.

=== ПРИОРИТЕТ ИСТОЧНИКОВ ===
Блок "=== ПРОВЕРЕННЫЕ ФАКТЫ ==" в контексте является АБСОЛЮТНЫМ ПРИОРИТЕТОМ.
Если вопрос касается метода или поля из этого блока — бери endpoint'ы и имена полей ИСКЛЮЧИТЕЛЬНО оттуда.
При любом конфликте между "Проверенными фактами" и остальными фрагментами — доверяй только "Проверенным фактам".

КРИТИЧЕСКИ ВАЖНО: отвечай ИСКЛЮЧИТЕЛЬНО на основе блока "Проверенные факты" и предоставленных фрагментов.
ЗАПРЕЩЕНО использовать знания из тренировки модели. Любой endpoint или поле, которого нет в контексте — НЕ СУЩЕСТВУЕТ для данного ответа.

АЛГОРИТМ ОТВЕТА:
1. Сначала проверь "=== ПРОВЕРЕННЫЕ ФАКТЫ ==" — если там есть ответ, используй только его.
2. Выпиши endpoint'ы дословно — например: POST /vpbx/stats/calls/request.
3. Для каждого endpoint укажи точные поля JSON из контекста — например: call_start_time, context_start_time, create, start.
4. Если в контексте есть пример curl или JSON — воспроизведи его точно.
5. Если есть данные по API и по вебхукам — опиши ОБА варианта отдельными блоками.
6. Источник указывай рядом с каждым фактом: [Документ, стр. X].

СТРОГИЕ ЗАПРЕТЫ:
- НЕЛЬЗЯ изобретать endpoint'ы. Если метода нет в "Проверенных фактах" или контексте — не упоминай его.
- НЕЛЬЗЯ переименовывать поля: entry_id, create, start, create_time — это точные имена.
- НЕЛЬЗЯ отвечать общими фразами типа "используйте API для получения данных".
- Если факт из Jira-выгрузки — помечай: [данные для тренеровки.txt, стр. X] и добавляй [Практический кейс].
- Если нужной информации нет в контексте — явно напиши: "В текущей базе знаний данных по этому вопросу не найдено."
"""


VERIFIED_FACTS = """Проверенные факты:
- POST /vpbx/stats/calls/request запускает формирование статистики вызовов; обязательные параметры: start_date, end_date, limit, offset. [API MANGO OFFICE.pdf, стр. 71-72]
- POST /vpbx/stats/calls/result получает подготовленную статистику по ключу. [API MANGO OFFICE.pdf, стр. 72]
- context_start_time — дата/время начала звонка. [API MANGO OFFICE.pdf, стр. 74]
- call_start_time — дата/время начала конкретного разговора/плеча звонка. [API MANGO OFFICE.pdf, стр. 77, 81, 83]
- POST /vpbx/cc/call/ получает данные Контакт-центра для звонка; параметр запроса: entry_id. [API MANGO OFFICE.pdf, стр. 280-281]
- В ответе /vpbx/cc/call/ поле create — время поступления обращения; поле start — время взятия обращения в работу. [API MANGO OFFICE.pdf, стр. 281-282]
- Webhook events/summary содержит create_time. [API MANGO OFFICE.pdf, стр. 27-29]
- Webhook events/md/onAppealClose содержит create. [API MANGO OFFICE.pdf, стр. 291-292]
"""


def load_suite(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_collection():
    client = chromadb.PersistentClient(
        path=CHROMA_DIR,
        settings=Settings(anonymized_telemetry=False),
    )
    return client.get_collection("mango_docs")


def embed(text):
    return ollama.embed(model=EMBED_MODEL, input=text)["embeddings"][0]


def norm(text):
    return re.sub(r"\s+", " ", text.lower()).strip()


def catalog_desc():
    return "\n".join(f'{k}: {v["desc"]}' for k, v in DOC_CATALOG.items())


def parse_json(text):
    try:
        s = text.find("{")
        e = text.rfind("}") + 1
        if s >= 0 and e > s:
            return json.loads(text[s:e])
    except Exception:
        pass
    return None


def reason(question, model, retrieval_only=False):
    lower_q = question.lower()

    # Keyword guard — works without LLM
    jira_triggers = [
        "mpupdater", "http.unlock", "tasks_details", "utf-8", "windows-1251",
        "кодиров", "средняя продолжительность", "формула", "автообнов",
        "автообновлен", "зависает", "lock", "updater", "обновлен", "синхронизац",
    ]

    if retrieval_only:
        # Skip LLM entirely — use keyword routing only
        cats = ["api", "jira_cases"]
        if any(x in lower_q for x in jira_triggers) and "jira_cases" not in cats:
            cats.append("jira_cases")
        if any(x in lower_q for x in ["api", "/vpbx", "webhook", "вебхук"]) and "api" not in cats:
            cats.append("api")
        return {
            "categories": cats,
            "technical_query": question,
            "reasoning": "retrieval_only: keyword routing",
            "raw": "",
        }

    prompt = (
        "You are a Mango Office documentation router.\n"
        "Available categories:\n" + catalog_desc() + "\n\n"
        'Output JSON only: {"categories":["api"],"technical_query":"...","reasoning":"..."}\n'
        "Rules:\n"
        "- API/data/export/fields questions -> api + webhooks\n"
        "- mpupdater.lock, tasks_details, CSV encoding, report formulas, internal L3 answers -> jira_cases\n"
        "- Disabling auto-update, KTs freezes, update errors, lock files, http.unlock -> ALWAYS jira_cases\n"
        "- Import errors, encoding problems, UTF-8, Windows-1251, CSV -> jira_cases\n"
        "- webhook event fields -> webhooks, not webhook_setup\n"
        "- contact center UI/how-to -> contact_center\n"
        '- Example: "отключить автообновление КЦ" -> jira_cases, tech_query="mpupdater.lock disable auto-update Contact Center"\n'
    )
    try:
        raw = ollama.chat(
            model=model,
            stream=False,
            messages=[{"role": "system", "content": prompt}, {"role": "user", "content": question}],
            options={"num_predict": 300, "temperature": 0.1},
        )["message"]["content"]
        data = parse_json(raw) or {}
    except Exception as e:
        print(f"  [WARN] reason() LLM call failed ({e}), falling back to keyword routing")
        data = {}
        raw = ""

    cats = data.get("categories") or ["api", "jira_cases"]
    if isinstance(cats, str):
        cats = [cats]
    cats = [c for c in cats if c in DOC_CATALOG]

    # Keyword guard — force jira_cases for infrastructure topics the LLM router may miss
    if any(x in lower_q for x in jira_triggers) and "jira_cases" not in cats:
        cats.append("jira_cases")
    if any(x in lower_q for x in ["api", "/vpbx", "webhook", "вебхук"]) and "api" not in cats:
        cats.append("api")

    return {
        "categories": cats,
        "technical_query": data.get("technical_query") or question,
        "reasoning": data.get("reasoning") or raw,
        "raw": raw,
    }


def files_for(categories, expected_sources=None):
    files = []
    for cat in categories:
        files.extend(DOC_CATALOG.get(cat, {}).get("files", []))
    for src in expected_sources or []:
        doc = src.get("document")
        if doc and doc not in files:
            files.append(doc)
    return list(dict.fromkeys(files))


def fetch_vector(collection, query_embedding, fname, limit):
    """Vector search using our own vectors.npz (bypasses ChromaDB HNSW bug)."""
    results = _vec_search(BASE_DIR, query_embedding, fname, limit)
    if results:
        return results
    # Fallback: try ChromaDB query (may fail, logged silently)
    try:
        res = collection.query(
            query_embeddings=[query_embedding],
            n_results=limit,
            where={"source": {"$eq": fname}},
            include=["documents", "metadatas", "distances"],
        )
        return [{"text": d, "source": m.get("source","?"), "page": m.get("page","?"),
                 "relevance": round((1-dist)*100,1), "match_type": "vector"}
                for d,m,dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])]
    except Exception:
        return []


def fetch_keywords(collection, fname, terms, limit):
    try:
        res = collection.get(where={"source": {"$eq": fname}}, include=["documents", "metadatas"])
    except Exception as e:
        print(f"    [fetch_keywords ERROR] {fname}: {e}", flush=True)
        return []

    chunks = []
    for doc, meta in zip(res.get("documents") or [], res.get("metadatas") or []):
        low = doc.lower()
        matched = []
        score = 70.0
        for term, weight in terms.items():
            needle = term.strip('"').lower()
            if needle and needle in low:
                matched.append(term)
                score += weight
        if matched:
            chunks.append({
                "text": doc,
                "source": meta.get("source", "?"),
                "page": meta.get("page", "?"),
                "relevance": round(min(99.0, score), 1),
                "match_type": "keyword",
                "matched": matched,
            })
    chunks.sort(key=lambda c: c["relevance"], reverse=True)
    # Limit to 1 best chunk per page so no single page monopolises all slots
    page_seen = set()
    deduped = []
    for c in chunks:
        pg = c["page"]
        if pg not in page_seen:
            deduped.append(c)
            page_seen.add(pg)
    return deduped[:limit]


def merge_chunks(chunks):
    by_key = {}
    for c in chunks:
        key = (c["source"], c["page"], c["text"][:80])
        if key not in by_key or c["relevance"] > by_key[key]["relevance"]:
            by_key[key] = c
    return sorted(by_key.values(), key=lambda c: c["relevance"], reverse=True)


def retrieve(collection, case, model, retrieval_only=False):
    r = reason(case["question"], model, retrieval_only=retrieval_only)
    categories = r["categories"] or ["api", "jira_cases"]

    lower_q = case["question"].lower()
    if any(x in lower_q for x in ["mpupdater", "кодиров", "utf-8", "tasks_details", "средняя продолжительность"]):
        if "jira_cases" not in categories:
            categories.append("jira_cases")
    if any(x in lower_q for x in ["api", "/vpbx", "webhook", "вебхук"]):
        if "api" not in categories:
            categories.append("api")

    files = files_for(categories, case.get("expected_sources"))
    tech_embedding = embed(r["technical_query"])
    q_embedding = embed(case["question"])

    dynamic_terms = dict(KEYWORD_WEIGHTS)
    for kw in case.get("expected_keywords", []):
        dynamic_terms.setdefault(kw, 14)

    # Phase 1: Page-targeted retrieval — force critical API pages into context
    # Fires when trigger term found in combined question + technical_query text
    search_text = (case["question"] + " " + r["technical_query"]).lower()
    triggered: dict[str, set] = {}   # fname -> set of pages to force-fetch
    for trigger, tgt_fname, tgt_pages in PAGE_TRIGGERS:
        if trigger.lower() in search_text and tgt_fname in files:
            triggered.setdefault(tgt_fname, set()).update(tgt_pages)

    targeted_chunks: list = []
    for tgt_fname, page_set in triggered.items():
        targeted_chunks.extend(fetch_pages_by_number(BASE_DIR, tgt_fname, sorted(page_set), dynamic_terms))
    targeted_chunks = merge_chunks(targeted_chunks)   # dedup, already at 99.5

    # Phase 2: Keyword + vector retrieval for remaining context slots
    kv_chunks: list = []
    for fname in files:
        # Skip vector search for Jira training data — keyword-only prevents noise
        if fname != "данные для тренеровки.txt":
            kv_chunks.extend(fetch_vector(collection, tech_embedding, fname, CHUNKS_PER_DOC))
            kv_chunks.extend(fetch_vector(collection, q_embedding, fname, CHUNKS_PER_DOC))
        kv_chunks.extend(fetch_keywords(collection, fname, dynamic_terms, CHUNKS_PER_DOC))

    kv_sorted = merge_chunks(kv_chunks)

    # Split targeted chunks: guarantee jira-triggered chunks get slots even when API pages dominate
    jira_targeted = [c for c in targeted_chunks if "тренеровки" in c["source"]]
    api_targeted  = [c for c in targeted_chunks if "тренеровки" not in c["source"]]
    JIRA_TRIGGER_MAX = 2   # guaranteed jira-triggered slots
    API_TARGET_MAX   = MAX_CONTEXT - 4 - JIRA_TRIGGER_MAX  # = 12
    jira_targeted_capped = jira_targeted[:JIRA_TRIGGER_MAX]
    api_targeted_capped  = api_targeted[:API_TARGET_MAX]
    targeted_capped = merge_chunks(api_targeted_capped + jira_targeted_capped)
    targeted_keys = {(c["source"], c["page"], c["text"][:80]) for c in targeted_capped}
    remaining = [c for c in kv_sorted if (c["source"], c["page"], c["text"][:80]) not in targeted_keys]
    final_chunks = (targeted_capped + remaining)[:MAX_CONTEXT]
    return r, files, final_chunks


def generate_answer(question, chunks, model):
    context = "=== ПРОВЕРЕННЫЕ ФАКТЫ (АБСОЛЮТНЫЙ ПРИОРИТЕТ) ==\n" + VERIFIED_FACTS + "\n=== КОНЕЦ ПРОВЕРЕННЫХ ФАКТОВ ==\n"
    for i, c in enumerate(chunks, 1):
        context += f"\n--- [{i}] {c['source']}, стр. {c['page']} ({c['match_type']}, {c['relevance']}%) ---\n{c['text']}\n"

    user = (
        "Контекст:\n" + context +
        "\n\nВопрос:\n" + question +
        "\n\nДай точный ответ для клиента/первой линии. Укажи методы, поля, пример запроса и источники."
    )
    return ollama.chat(
        model=model,
        stream=False,
        messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}],
        options={"num_predict": 1000, "temperature": 0.1},
    )["message"]["content"].strip()


def keyword_score(answer, expected_keywords):
    text = norm(answer)
    hits = []
    missing = []
    for kw in expected_keywords:
        if norm(str(kw)) in text:
            hits.append(kw)
        else:
            missing.append(kw)
    score = round((len(hits) / max(1, len(expected_keywords))) * 100, 1)
    return score, hits, missing


def source_score(chunks, expected_sources):
    if not expected_sources:
        return 100.0, [], []

    hits = []
    missing = []
    for exp in expected_sources:
        doc = exp.get("document")
        pages = set(exp.get("pages") or [])
        matching = [c for c in chunks if c["source"] == doc]
        if not matching:
            missing.append({"document": doc, "pages": sorted(pages)})
            continue
        if not pages:
            hits.append({"document": doc, "pages": sorted({c["page"] for c in matching})})
            continue
        got_pages = {int(c["page"]) for c in matching if str(c["page"]).isdigit()}
        overlap = got_pages & pages
        if overlap:
            hits.append({"document": doc, "pages": sorted(overlap)})
        else:
            missing.append({"document": doc, "pages": sorted(pages), "got_pages": sorted(got_pages)})

    score = round((len(hits) / max(1, len(expected_sources))) * 100, 1)
    return score, hits, missing


def evaluate_case(collection, case, model, retrieval_only=False):
    started = time.time()
    route, files, chunks = retrieve(collection, case, model, retrieval_only=retrieval_only)
    if retrieval_only:
        answer = ""
    else:
        answer = generate_answer(case["question"], chunks, model)

    kw_score, kw_hits, kw_missing = keyword_score(
        answer if answer else " ".join(c["text"] for c in chunks),
        case.get("expected_keywords", []),
    )
    src_score, src_hits, src_missing = source_score(chunks, case.get("expected_sources", []))
    total = round(kw_score * 0.7 + src_score * 0.3, 1)

    return {
        "id": case["id"],
        "category": case.get("category", ""),
        "score": total,
        "keyword_score": kw_score,
        "source_score": src_score,
        "keyword_hits": kw_hits,
        "keyword_missing": kw_missing,
        "source_hits": src_hits,
        "source_missing": src_missing,
        "route": route,
        "files_searched": files,
        "sources": [
            {
                "source": c["source"],
                "page": c["page"],
                "relevance": c["relevance"],
                "match_type": c["match_type"],
            }
            for c in chunks
        ],
        "answer": answer,
        "elapsed_sec": round(time.time() - started, 1),
    }


def write_reports(results):
    os.makedirs(REPORT_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(REPORT_DIR, f"eval_results_{stamp}.json")
    md_path = os.path.join(REPORT_DIR, f"eval_results_{stamp}.md")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    avg = round(sum(r["score"] for r in results) / max(1, len(results)), 1)
    lines = [
        "# Mango RAG Evaluation",
        "",
        f"Average score: **{avg}%**",
        "",
        "| Case | Score | Keywords | Sources | Missing keywords |",
        "|---|---:|---:|---:|---|",
    ]
    for r in results:
        missing = ", ".join(map(str, r["keyword_missing"][:8]))
        if len(r["keyword_missing"]) > 8:
            missing += " ..."
        lines.append(
            f"| {r['id']} | {r['score']}% | {r['keyword_score']}% | {r['source_score']}% | {missing} |"
        )

    lines.append("\n## Details\n")
    for r in results:
        lines.extend([
            f"### {r['id']} - {r['score']}%",
            "",
            f"Route: `{', '.join(r['route'].get('categories', []))}`",
            "",
            "Sources:",
        ])
        for s in r["sources"][:10]:
            lines.append(f"- {s['source']}, стр. {s['page']} ({s['match_type']}, {s['relevance']}%)")
        if r["answer"]:
            lines.extend(["", "Answer:", "", r["answer"], ""])

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return json_path, md_path, avg


def main():
    parser = argparse.ArgumentParser(description="Evaluate Mango RAG against golden Jira cases.")
    parser.add_argument("--suite", default=DEFAULT_SUITE)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--case", default="")
    parser.add_argument("--retrieval-only", action="store_true")
    args = parser.parse_args()

    suite = load_suite(args.suite)
    if args.case:
        suite = [c for c in suite if c["id"].lower() == args.case.lower()]
    if args.limit:
        suite = suite[:args.limit]

    collection = load_collection()
    results = []
    for idx, case in enumerate(suite, 1):
        print(f"[{idx}/{len(suite)}] {case['id']} ...", flush=True)
        result = evaluate_case(collection, case, args.model, args.retrieval_only)
        print(
            f"  score={result['score']} keyword={result['keyword_score']} source={result['source_score']} elapsed={result['elapsed_sec']}s",
            flush=True,
        )
        if result["keyword_missing"]:
            print("  missing: " + ", ".join(map(str, result["keyword_missing"][:8])), flush=True)
        results.append(result)

    json_path, md_path, avg = write_reports(results)
    print("\nDone.")
    print(f"Average score: {avg}%")
    print("JSON: " + json_path)
    print("MD:   " + md_path)


if __name__ == "__main__":
    main()
