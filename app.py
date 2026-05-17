import os, json
import streamlit as st
import chromadb, ollama
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


CHROMA_DIR        = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
EMBED_MODEL       = "nomic-embed-text"
CHUNKS_STEP       = 4
MAX_FINAL         = 12
CONFIDENCE_TARGET = 70
MAX_ITERATIONS    = 3

DOC_CATALOG = {
    "api": {
        "files": ["API MANGO OFFICE.pdf"],
        "desc": "Main API reference. Contains REST API, /vpbx/stats/calls/request, /vpbx/stats/calls/result, /vpbx/cc/call, raw data export, BI, exact fields: context_start_time, call_start_time, create, start."
    },
    "api_server": {
        "files": ["MANGO OFFICE Описание API Сервер клиент\xbb.pdf"],
        "desc": "API server-client, SDK, authentication, connection"
    },
    "webhooks": {
        "files": ["API MANGO OFFICE.pdf"],
        "desc": "Webhook event reference inside API MANGO OFFICE.pdf. Contains events/summary, events/md/onAppealClose, create_time, create, start. Do NOT route event field lookup to the integration setup guide."
    },
    "webhook_setup": {
        "files": ["ИНТЕГРАЦИЯ MANGO OFFICE Инструкция по вебхуки MANGO OFFICE.pdf"],
        "desc": "Instruction for configuring external webhook URLs in the personal account / CRM actions. Does NOT contain statistics fields, events/summary, onAppealClose, create_time, create or start."
    },
    "contact_center": {
        "files": ["КЦ - Контакт центр руководства пользователя.pdf"],
        "desc": "Contact center UI, queues, operators, supervisor, reports, SLA"
    },
    "lk": {
        "files": ["ЛК - виртуальная атс mango office ЮТЕР ЕРЕАДРЕ СПРАВОЧНИК АБОНЕНТА.pdf"],
        "desc": "Personal account, virtual PBX, routing, subscriber guide"
    },
    "quality": {
        "files": ["КОНТРОЛЬ КАЧЕСТВА Руководство пользователя.pdf"],
        "desc": "Quality control, call evaluation, scoring, checklists"
    },
    "wallboard": {
        "files": ["Программа для ЭВМ Wallboard Mango Office Руководство пользователя.pdf"],
        "desc": "Wallboard display, real-time KPI panels"
    },
    "jira_cases": {
        "files": ["данные для тренеровки.txt"],
        "desc": "Jira L3 support cases with real consultations, custom workarounds and internal facts not always present in public PDF docs: mpupdater.lock, tasks_details, UTF-8 CSV import, metric formulas, customer-specific API clarifications."
    },
}

SYSTEM_PROMPT = (
    "Ты — ВЕКТОР (Virtual Expert Knowledge Technical Operations Research), система поддержки L3 Mango Office.\n\n"
    "=== ПРИОРИТЕТ ИСТОЧНИКОВ ===\n"
    "Блок '=== ПРОВЕРЕННЫЕ ФАКТЫ ==' в контексте является АБСОЛЮТНЫМ ПРИОРИТЕТОМ.\n"
    "Если вопрос касается метода или поля из этого блока — бери endpoint'ы и имена полей ИСКЛЮЧИТЕЛЬНО оттуда.\n"
    "При любом конфликте между 'Проверенными фактами' и остальными фрагментами — доверяй только 'Проверенным фактам'.\n\n"
    "КРИТИЧЕСКИ ВАЖНО: отвечай ИСКЛЮЧИТЕЛЬНО на основе блока 'Проверенные факты' и предоставленных фрагментов.\n"
    "ЗАПРЕЩЕНО использовать знания из тренировки модели. Любой endpoint или поле, которого нет в контексте — НЕ СУЩЕСТВУЕТ для данного ответа.\n\n"
    "АЛГОРИТМ ОТВЕТА:\n"
    "1. Сначала проверь '=== ПРОВЕРЕННЫЕ ФАКТЫ ==' — если там есть ответ, используй только его.\n"
    "2. Выпиши endpoint'ы дословно — например: `POST /vpbx/stats/calls/request`.\n"
    "3. Для каждого endpoint укажи точные поля JSON из контекста — например: `call_start_time`, `context_start_time`, `create`, `start`.\n"
    "4. Если в контексте есть пример curl или JSON — воспроизведи его точно.\n"
    "5. Если есть данные по API и по вебхукам — опиши ОБА варианта отдельными блоками.\n"
    "6. Источник указывай рядом с каждым фактом: [Документ, стр. X].\n\n"
    "СТРОГИЕ ЗАПРЕТЫ:\n"
    "- НЕЛЬЗЯ изобретать endpoint'ы. Если метода нет в 'Проверенных фактах' или контексте — не упоминай его.\n"
    "- НЕЛЬЗЯ переименовывать поля: entry_id, create, start, create_time — это точные имена, не синонимы.\n"
    "- НЕЛЬЗЯ писать 'p.' — только 'стр.'\n"
    "- НЕЛЬЗЯ отвечать общими фразами типа 'используйте API для получения данных'.\n"
    "- Если факт из Jira-выгрузки — помечай: [данные для тренеровки.txt, стр. X] и добавляй пометку [Практический кейс].\n"
    "- Если факт из PDF — помечай: [API MANGO OFFICE.pdf, стр. X].\n"
    "- Если нужной информации нет в контексте — явно напиши: 'В текущей базе знаний данных по этому вопросу не найдено.'"
)

VERIFIED_FACTS = (
    "Проверенные факты из API MANGO OFFICE.pdf:\n"
    "- POST /vpbx/stats/calls/request запускает формирование статистики вызовов. Обязательные параметры: start_date, end_date, limit, offset. [API MANGO OFFICE.pdf, стр. 71-72]\n"
    "- POST /vpbx/stats/calls/result получает подготовленную статистику по ключу. [API MANGO OFFICE.pdf, стр. 72]\n"
    "- В статистике вызовов поле context_start_time — дата/время начала звонка. [API MANGO OFFICE.pdf, стр. 74]\n"
    "- В детализации context_calls поле call_start_time — дата/время начала конкретного разговора/плеча звонка. [API MANGO OFFICE.pdf, стр. 77, 81, 83]\n"
    "- POST /vpbx/cc/call/ получает данные Контакт-центра для звонка; параметр запроса: entry_id. [API MANGO OFFICE.pdf, стр. 280-281]\n"
    "- В ответе /vpbx/cc/call/ поле create — время поступления обращения; поле start — время взятия обращения в работу. [API MANGO OFFICE.pdf, стр. 281-282]\n"
    "- Webhook events/summary содержит create_time — время поступления входящего вызова / начала исходящего или внутреннего вызова. [API MANGO OFFICE.pdf, стр. 27-29]\n"
    "- Webhook events/md/onAppealClose описывает событие закрытия обращения и содержит create — дату/время создания обращения. [API MANGO OFFICE.pdf, стр. 291-292]\n"
)


def catalog_desc():
    return "\n".join('  "{}": "{}"'.format(k, v["desc"]) for k, v in DOC_CATALOG.items())


REASONING_PROMPT = (
    "You are a Mango Office documentation router.\n"
    "Available categories:\n" + catalog_desc() + "\n\n"
    "Output JSON only (no markdown):\n"
    '{"intent":"...","categories":["cat1"],"technical_query":"...","reasoning":"..."}\n\n'
    "Rules:\n"
    "- API/data/export/fields questions -> always include both api AND webhooks\n"
    "- Questions about real L3 consultations, mpupdater.lock, tasks_details, CSV encoding, report formulas, or facts absent from PDF -> include jira_cases\n"
    "- Questions about disabling auto-update, КЦ freezes, update errors, mpupdater, lock files, http.unlock -> ALWAYS include jira_cases\n"
    "- Questions about import errors, encoding, UTF-8, Windows-1251, CSV -> include jira_cases\n"
    "- webhook event field questions -> webhooks, not webhook_setup\n"
    "- UI/interface questions -> contact_center\n"
    "- technical_query: rewrite in English API terminology (endpoint paths, field names)\n"
    '  Example: "время назначения обращения" -> "vpbx/cc/call assignment create_time onAppealClose"\n'
    '  Example: "время начала разговора" -> "vpbx/stats/calls call_start_time context_start_time"\n'
    '  Example: "взятие обращения сотрудником" -> "vpbx/cc/call start_time operator accept"\n'
    '  Example: "отключить автообновление КЦ" -> "mpupdater.lock disable auto-update Contact Center"\n'
)

CHECK_PROMPT = (
    "You are a RAG quality checker for Mango Office API documentation.\n"
    "Given a question and context fragments, assess if context is sufficient.\n\n"
    "Be GENEROUS in scoring:\n"
    "- If context contains relevant API endpoints OR field names -> confidence >= 75\n"
    "- If context has exact field names matching the question -> confidence >= 85\n"
    "- If context has partial info (related endpoint but not exact field) -> confidence 55-74\n"
    "- If context is unrelated -> confidence < 40\n\n"
    "Output JSON only:\n"
    '{"sufficient":true,"confidence":0-100,"reason":"...","missing":"...","fallback_categories":[]}'
)

WEBHOOK_FILES = set()


@st.cache_resource
def get_models():
    preferred = ["qwen2.5:14b", "qwen2.5:7b", "llama3.2", "qwen2.5:3b",
                 "mistral", "llama3.1", "llama3"]
    try:
        all_m = [m.model for m in ollama.list().models]
        ordered = [m for m in preferred if any(m in x for x in all_m)]
        others = [m for m in all_m
                  if not any(p in m for p in preferred) and "embed" not in m.lower()]
        result = [m for m in ordered if "embed" not in m.lower()] + others
        return result if result else preferred
    except Exception:
        return preferred


@st.cache_resource
def load_chroma():
    if not os.path.exists(CHROMA_DIR):
        return None
    client = chromadb.PersistentClient(
        path=CHROMA_DIR, settings=Settings(anonymized_telemetry=False))
    try:
        return client.get_collection("mango_docs")
    except Exception:
        return None


def embed(text):
    return ollama.embed(model=EMBED_MODEL, input=text)["embeddings"][0]


def llm(system, user, model, tokens=200):
    r = ollama.chat(
        model=model,
        messages=[{"role": "system", "content": system},
                  {"role": "user",   "content": user}],
        stream=False,
        options={"num_predict": tokens, "temperature": 0.1})
    return r["message"]["content"].strip()


def parse_json(text):
    try:
        s = text.find("{")
        e = text.rfind("}") + 1
        if s >= 0 and e > s:
            return json.loads(text[s:e])
    except Exception:
        pass
    return None


def stream_reason(question, model, on_token=None):
    raw_buf = [""]
    for chunk in ollama.chat(
            model=model,
            messages=[{"role": "system", "content": REASONING_PROMPT},
                      {"role": "user",   "content": question}],
            stream=True,
            options={"num_predict": 300, "temperature": 0.1}):
        t = chunk["message"]["content"]
        if t:
            raw_buf[0] += t
            if on_token:
                on_token(raw_buf[0])
    d = parse_json(raw_buf[0])
    if not d:
        return {"intent": question, "categories": ["api", "webhooks"],
                "technical_query": question, "reasoning": "fallback",
                "_raw": raw_buf[0]}
    d["_raw"] = raw_buf[0]
    return d


def files_for(cats):
    files = []
    for c in cats:
        if c in DOC_CATALOG:
            files.extend(DOC_CATALOG[c]["files"])
    return list(dict.fromkeys(files))


def fetch_doc(collection, emb_vec, fname, n):
    """Vector search using our own vectors.npz (bypasses ChromaDB HNSW bug)."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results = _vec_search(base_dir, emb_vec, fname, n)
    if results:
        return results
    # Fallback: ChromaDB query
    try:
        res = collection.query(
            query_embeddings=[emb_vec], n_results=n,
            where={"source": {"$eq": fname}},
            include=["documents", "metadatas", "distances"])
        return [{"text": d, "source": m.get("source","?"), "page": m.get("page","?"),
                 "relevance": round((1-dist)*100,1)}
                for d,m,dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])]
    except Exception:
        return []


KEYWORD_WEIGHTS = {
    "POST /vpbx/cc/call": 18,
    "/vpbx/task/add": 16,
    "/config/users/request": 14,
    "/vpbx/stats/calls/request": 14,
    "/vpbx/stats/calls/result": 14,
    "/vpbx/cc/call": 16,
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


def fetch_keywords(collection, fname, terms, limit=12):
    try:
        res = collection.get(
            where={"source": {"$eq": fname}},
            include=["documents", "metadatas"])
    except Exception:
        return []

    hits = []
    docs = res.get("documents") or []
    metas = res.get("metadatas") or []
    for doc, meta in zip(docs, metas):
        low = doc.lower()
        matched = []
        score = 70.0
        for term, weight in terms.items():
            needle = term.strip('"').lower()
            if needle in low:
                matched.append(term)
                score += weight
        if not matched:
            continue
        hits.append({
            "text": doc,
            "source": meta.get("source", "?"),
            "page": meta.get("page", "?"),
            "relevance": round(min(99.0, score), 1),
        })
    hits.sort(key=lambda x: x["relevance"], reverse=True)
    # Limit to 1 best chunk per page so no single page monopolises all slots
    page_seen = set()
    deduped = []
    for c in hits:
        pg = c["page"]
        if pg not in page_seen:
            deduped.append(c)
            page_seen.add(pg)
    return deduped[:limit]


def merge(pool, new_chunks):
    for c in new_chunks:
        k = (c["source"], c["page"])
        if k not in pool or c["relevance"] > pool[k]["relevance"]:
            pool[k] = c
    return pool


def check_context(question, chunks, model):
    preview = ""
    for i, c in enumerate(chunks, 1):
        preview += "[{}] {} стр. {}: {}...\n".format(
            i, c["source"], c["page"], c["text"][:500])
    raw = llm(
        CHECK_PROMPT,
        "Question: {}\n\nContext ({} fragments):\n{}".format(
            question, len(chunks), preview),
        model, tokens=200)
    d = parse_json(raw)
    if not d:
        return {"sufficient": True, "confidence": 70, "reason": "parse failed",
                "missing": "", "fallback_categories": [], "_raw": raw}
    d["_raw"] = raw
    return d


def iterative_search(collection, question, tech_query, cats, model, log_cb=None):
    if "api" in cats and "webhooks" not in cats:
        cats = list(cats) + ["webhooks"]
    api_files = set(files_for(["api"]))
    wh_files = set(files_for(["webhooks"]))
    jira_files = set(files_for(["jira_cases"]))
    target_files = files_for(cats)
    pool = {}
    confidence = 0
    check_data = {}
    iters_log = []

    emb_api     = embed(tech_query)
    emb_webhook = embed(question + " вебхук событие время поле")
    base_dir    = os.path.dirname(os.path.abspath(__file__))

    # Phase 0: Page-targeted retrieval — force critical API pages into pool
    # Jira-triggered chunks get guaranteed slots so API pages can't crowd them out
    search_text = (question + " " + tech_query).lower()
    triggered: dict = {}
    for trigger, tgt_fname, tgt_pages in PAGE_TRIGGERS:
        if trigger.lower() in search_text and tgt_fname in target_files:
            triggered.setdefault(tgt_fname, set()).update(tgt_pages)

    JIRA_TRIGGER_MAX = 2
    API_TARGET_MAX   = MAX_FINAL - JIRA_TRIGGER_MAX  # = 10
    api_targeted: list = []
    jira_targeted: list = []
    for tgt_fname, page_set in triggered.items():
        chunks = fetch_pages_by_number(base_dir, tgt_fname, sorted(page_set), KEYWORD_WEIGHTS)
        if "тренеровки" in tgt_fname:
            jira_targeted.extend(chunks)
        else:
            api_targeted.extend(chunks)
    api_targeted_capped  = sorted(api_targeted,  key=lambda x: x["relevance"], reverse=True)[:API_TARGET_MAX]
    jira_targeted_capped = sorted(jira_targeted, key=lambda x: x["relevance"], reverse=True)[:JIRA_TRIGGER_MAX]
    pool = merge(pool, api_targeted_capped + jira_targeted_capped)

    if triggered and log_cb:
        log_cb("Page-targeted: {} API + {} jira chunks from {}".format(
            len(api_targeted_capped), len(jira_targeted_capped), list(triggered.keys())))

    for iteration in range(1, MAX_ITERATIONS + 1):
        n    = CHUNKS_STEP * iteration
        n_wh = n * 2
        if log_cb:
            log_cb("Iter {}/{}: API {}ch, webhook {}ch, {} files".format(
                iteration, MAX_ITERATIONS, n, n_wh, len(target_files)))

        for fname in target_files:
            # Skip vector search for Jira training data — keyword-only prevents noise
            if fname not in jira_files:
                if fname in api_files:
                    pool = merge(pool, fetch_doc(collection, emb_api, fname, n))
                if fname in wh_files:
                    pool = merge(pool, fetch_doc(collection, emb_webhook, fname, n_wh))
            if fname in api_files or fname in wh_files or fname in jira_files:
                pool = merge(pool, fetch_keywords(collection, fname, KEYWORD_WEIGHTS, limit=n_wh))

        top = sorted(pool.values(), key=lambda x: x["relevance"], reverse=True)[:MAX_FINAL]
        check_data = check_context(question, top, model)
        confidence = check_data.get("confidence", 0)
        fallback = [f for f in check_data.get("fallback_categories", [])
                    if f in DOC_CATALOG]

        line = "Iter {}: {} chunks pooled, confidence {}%".format(
            iteration, len(pool), confidence)
        if fallback:
            line += " | fallback: " + str(fallback)
        iters_log.append(line)
        if log_cb:
            log_cb(line)

        if confidence >= CONFIDENCE_TARGET:
            break

        if fallback and iteration < MAX_ITERATIONS:
            for f in files_for(fallback):
                if f not in target_files:
                    target_files.append(f)

    final = sorted(pool.values(), key=lambda x: x["relevance"], reverse=True)[:MAX_FINAL]
    return final, target_files, confidence, check_data, iters_log


def stream_answer(question, chunks, model):
    ctx = "=== ПРОВЕРЕННЫЕ ФАКТЫ (АБСОЛЮТНЫЙ ПРИОРИТЕТ) ==\n" + VERIFIED_FACTS + "\n=== КОНЕЦ ПРОВЕРЕННЫХ ФАКТОВ ==\n"
    for i, c in enumerate(chunks, 1):
        ctx += "\n--- [{}] {}, стр. {} ---\n{}\n".format(
            i, c["source"], c["page"], c["text"])
    stream = ollama.chat(
        model=model,
        stream=True,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": (
                "Context:\n" + ctx +
                "\n\nQuestion: " + question +
                "\n\nAnswer in Russian. Include specific API endpoints "
                "(e.g. /vpbx/stats/calls/request), exact field names, "
                "and a code example (curl). Cite document + page in Russian format: [document, стр. N]."
            )},
        ])
    for chunk in stream:
        t = chunk["message"]["content"]
        if t:
            yield t


# ---------- UI ----------

st.set_page_config(page_title="Mango Office KB", page_icon="📞", layout="wide")
st.title("📞 Mango Office — База знаний")

collection = load_chroma()
models     = get_models()

if collection is None:
    st.error("DB not found. Run 2_index.bat")
    st.stop()

doc_count = collection.count()

with st.sidebar:
    st.header("Settings")
    sel_model = st.selectbox("LLM model", options=models, index=0)
    st.caption("Embeddings: " + EMBED_MODEL)
    st.caption("Pipeline: Reason(stream) -> Iter(Retrieve+Check) -> Generate(stream)")
    st.caption("Confidence target: {}% | Max iters: {} | Chunks step: {}".format(
        CONFIDENCE_TARGET, MAX_ITERATIONS, CHUNKS_STEP))
    st.divider()
    st.header("Examples")
    examples = [
        "Как получить время начала разговора через API?",
        "Какой метод API для выгрузки статистики звонков?",
        "Что такое вебхук events/summary, какие поля?",
        "Как получить время назначения обращения в работу?",
        "Какое поле содержит время взятия обращения сотрудником?",
        "Как настроить очередь операторов в КЦ?",
        "Как работает контроль качества записей?",
    ]
    for q in examples:
        if st.button(q, use_container_width=True):
            st.session_state["pending"] = q
    st.divider()
    st.success(str(doc_count) + " fragments in DB")
    if st.button("Refresh", use_container_width=True):
        st.cache_resource.clear()
        st.rerun()

st.caption("Model: **{}** | Reason(stream)->Iter->Generate(stream) | {} fragments".format(
    sel_model, doc_count))

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources", expanded=False):
                for s in msg["sources"]:
                    st.caption("- **{}** стр. **{}** {}%".format(
                        s["source"], s["page"], s["relevance"]))
        if msg.get("trace"):
            with st.expander("Pipeline trace", expanded=False):
                st.code(msg["trace"])

pending    = st.session_state.pop("pending", None)
user_input = st.chat_input("Ask a question...") or pending

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        trace = []

        # Step 1: Reasoning with real-time token streaming
        with st.status("Step 1/3: Reasoning...", expanded=True) as s1:
            reason_ph = st.empty()
            def on_tok(current_text):
                reason_ph.code(current_text, language="json")
            r = stream_reason(user_input, sel_model, on_tok)
            reason_ph.empty()
            cats       = r.get("categories", ["api", "webhooks"])
            tech_query = r.get("technical_query", user_input)
            low_input = user_input.lower()
            if any(x in low_input for x in [
                "mpupdater", "tasks_details", "utf-8", "windows-1251",
                "кодиров", "средняя продолжительность", "формула", "автообнов",
                "автообновлен", "зависает", "зависает на синхронизац", "http.unlock",
                "lock", "updater", "обновление кц", "обновлен", "синхронизац"
            ]) and "jira_cases" not in cats:
                cats = list(cats) + ["jira_cases"]
            if any(x in low_input for x in ["/vpbx", "api", "апи"]) and "api" not in cats:
                cats = list(cats) + ["api"]
            st.markdown("**Intent:** " + r.get("intent", ""))
            st.markdown("**Routing:** " + " + ".join("`{}`".format(c) for c in cats))
            st.markdown("**Tech query:** " + tech_query)
            st.caption(r.get("reasoning", ""))
            s1.update(
                label="Step 1/3: Reasoning -> [{}]".format(", ".join(cats)),
                state="complete")
            trace += [
                "=== STEP 1: REASONING ===",
                "Intent:     " + r.get("intent", ""),
                "Categories: " + str(cats),
                "Tech query: " + tech_query,
                "Logic:      " + r.get("reasoning", ""),
            ]

        # Step 2: Iterative Retrieve + Check
        with st.status("Step 2/3: Iterative Retrieve + Check...", expanded=True) as s23:
            live = st.empty()
            log_lines = []
            def on_log(msg):
                log_lines.append(msg)
                live.caption(msg)
            chunks, files, confidence, chk, iters = iterative_search(
                collection, user_input, tech_query, cats, sel_model, on_log)
            live.empty()

            for fname in files:
                dc = [c for c in chunks if c["source"] == fname]
                if dc:
                    st.caption("doc: {} -- {} frags, best {}%".format(
                        fname, len(dc), dc[0]["relevance"]))
                else:
                    st.caption("doc: {} -- searched, 0 in final set".format(fname))
            st.divider()

            for line in iters:
                try:
                    cv = int(line.split("confidence ")[1].split("%")[0])
                except Exception:
                    cv = 0
                col = "green" if cv >= CONFIDENCE_TARGET else "orange" if cv >= 50 else "red"
                st.markdown(":{}: {}".format(col, line))

            col_f = "green" if confidence >= CONFIDENCE_TARGET else "orange" if confidence >= 50 else "red"
            st.markdown("**Final confidence:** :{}: **{}%** | {} fragments".format(
                col_f, confidence, len(chunks)))
            if chk.get("reason"):
                st.caption("Checker: " + chk["reason"])
            if chk.get("missing"):
                st.caption("Missing: " + chk["missing"])

            s23.update(
                label="Step 2/3: Retrieve+Check -> {}% confidence | {} frags".format(
                    confidence, len(chunks)),
                state="complete")
            trace += [
                "\n=== STEPS 2-3: ITERATIVE RETRIEVE+CHECK ===",
                "Files: " + str(files),
            ] + iters + [
                "Final confidence: {}%".format(confidence),
                "Checker: " + chk.get("reason", ""),
            ]

        # Step 3: Generate with streaming
        trace.append("\n=== STEP 3: GENERATE ===")
        answer = ""
        ph = st.empty()
        try:
            for tok in stream_answer(user_input, chunks, sel_model):
                answer += tok
                ph.markdown(answer + " |")
            ph.markdown(answer)
        except Exception as e:
            st.warning("LLM error: " + str(e))
            answer = "**Fragments:**\n\n"
            for i, c in enumerate(chunks, 1):
                answer += "**{}. {} стр. {}** ({}%)\n\n{}...\n\n---\n".format(
                    i, c["source"], c["page"], c["relevance"], c["text"][:400])
            ph.markdown(answer)

        with st.expander("Sources", expanded=True):
            for s in chunks:
                st.caption("- **{}** стр. **{}** {}%".format(
                    s["source"], s["page"], s["relevance"]))
        with st.expander("Pipeline trace", expanded=False):
            st.code("\n".join(trace))

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": [{"source": c["source"], "page": c["page"], "relevance": c["relevance"]} for c in chunks],
    })
