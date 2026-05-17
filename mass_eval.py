"""
mass_eval.py — массовый прогон ВЕКТОР с опциональной LLM-генерацией.
Выход: MD-отчёт сгруппированный по документу + блоки для Gemini-валидации.

Режимы:
  python mass_eval.py                  # retrieval-only (быстро, ~0.1с/вопрос)
  python mass_eval.py --generate       # полный ответ через LLM (~30с/вопрос)
  python mass_eval.py --generate --model qwen2.5:14b
"""
import os, sys, json, re, time, argparse
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluate_rag import (
    load_collection, retrieve, generate_answer,
    BASE_DIR, REPORT_DIR, DEFAULT_MODEL
)

QUESTIONS_FILE = os.path.join(BASE_DIR, "..", "вопросы для тренеровки .txt")
SUITE_FILE     = os.path.join(BASE_DIR, "test_suite.json")

# Map question categories/keywords → source PDF for Gemini grouping
DOC_GROUPS = {
    "API MANGO OFFICE.pdf":       ["api", "api_webhooks", "api_call_identity", "api_metrics",
                                   "api_custom", "extracted"],
    "MANGO OFFICE Описание API Сервер клиент.pdf": ["api_server"],
    "ИНТЕГРАЦИЯ MANGO OFFICE Инструкция по вебхуки MANGO OFFICE.pdf": ["webhooks", "webhook_setup"],
    "КОНТРОЛЬ КАЧЕСТВА Руководство пользователя.pdf": ["quality"],
    "КЦ - Контакт центр руководства пользователя.pdf": ["contact_center", "contact_center_infra",
                                                          "contact_center_import"],
    "ЛК - виртуальная атс mango office ЮТЕР ЕРЕАДРЕ СПРАВОЧНИК АБОНЕНТА.pdf": ["lk"],
    "Программа для ЭВМ Wallboard Mango Office Руководство пользователя.pdf": ["wallboard"],
}

GEMINI_VALIDATION_PROMPT = """Ты — строгий технический аудитор L3-поддержки Mango Office.
Тебе предоставлен оригинальный документ: **{doc_name}**
Ниже — пары «Вопрос / Ответ» системы ВЕКТОР (локальный RAG на документации Mango Office).

ТВОЯ ЗАДАЧА: для каждой пары проверить ответ ВЕКТОР по оригинальному документу и вынести вердикт.

КРИТЕРИИ ОЦЕНКИ:
- ✅ ВЕРНО       — ответ фактически точен, информация есть в документе, ничего лишнего не придумано
- ⚠️ ЧАСТИЧНО   — часть ответа верна, но есть неточности, пропущены важные детали или ответ неполный
- ❌ НЕВЕРНО     — ответ противоречит документу или содержит фактические ошибки
- 🚫 ГАЛЛЮЦИНАЦИЯ — ВЕКТОР уверенно утверждает факты, которых НЕТ в документе (эндпоинты, параметры, цифры)
- 📭 НЕТ В ДОКЕ — документ не содержит информации для ответа на этот вопрос (нельзя ни подтвердить, ни опровергнуть)

ПРАВИЛА:
1. Если ВЕКТОР называет конкретный URL, параметр, код ошибки или цифру — обязательно проверь по документу
2. Если информации нет в документе — ставь НЕТ В ДОКЕ, даже если ответ звучит правдоподобно
3. Укажи номер страницы документа, на которой нашёл (или не нашёл) подтверждение
4. Флаг ГАЛЛЮЦИНАЦИЯ ставь только когда ВЕКТОР явно утверждает конкретный факт, которого нет в доке

ФОРМАТ ВЫВОДА (строго для каждого вопроса):
---
Вопрос N: [текст вопроса — первые 100 символов]
Вердикт: ✅ ВЕРНО / ⚠️ ЧАСТИЧНО / ❌ НЕВЕРНО / 🚫 ГАЛЛЮЦИНАЦИЯ / 📭 НЕТ В ДОКЕ
Страница в доке: [стр. X или «не найдено»]
Что верно: [если есть]
Что неверно / чего не хватает: [конкретно]
Галлюцинации: [перечисли придуманные факты, если есть — иначе «нет»]
---

Начинай сразу с «---» для первого вопроса, без вступления.

=== ВОПРОСЫ И ОТВЕТЫ ВЕКТОР ===
"""


def extract_questions_from_file(path):
    if not os.path.exists(path):
        print(f"  Файл не найден: {path}")
        return []
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    questions = []
    seen = set()

    def add(q):
        q = re.sub(r'\s+', ' ', q).strip()
        # strip surrounding quotes/guillemets
        q = q.strip('«»"\'')
        if len(q) > 30 and q not in seen:
            questions.append(q)
            seen.add(q)

    # Pattern 1: "Вопрос: ..." (inline question label, any style)
    p1 = re.compile(
        r'(?:- )?Вопрос:\s*(.+?)(?=\n\s*(?:- )?Вопрос:|\nВопрос \d+:|\nПриветствую|\nКак |\nПривет|\Z)',
        re.DOTALL
    )
    for m in p1.finditer(text):
        add(m.group(1))

    # Pattern 2: "Вопрос по <тема>: «вопрос»" (доп вопросы — single-line format)
    p2 = re.compile(r'Вопрос по[^:]+:\s*[«"](.*?)[»"]\s*(?=$|\n)', re.MULTILINE)
    for m in p2.finditer(text):
        add(m.group(1))

    # Pattern 3: Numbered "Вопрос N: ..." headings followed by body question
    p3 = re.compile(
        r'Вопрос \d+:[^\n]*\n+(.+?\?)',
        re.DOTALL
    )
    for m in p3.finditer(text):
        candidate = re.sub(r'\s+', ' ', m.group(1)).strip()
        # Only short candidates (question sentence, not full paragraph)
        if 40 < len(candidate) < 600:
            add(candidate)

    # Pattern 4: Numbered sub-items "1.  Какой..." style
    p4 = re.compile(r'^\d+\.\s{2,}([А-ЯЁ].+?\?)', re.MULTILINE | re.DOTALL)
    for m in p4.finditer(text):
        add(m.group(1))

    return questions


def load_suite_questions(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        suite = json.load(f)
    return [{"id": c["id"], "question": c["question"],
             "category": c.get("category",""), "ground_truth": c.get("ground_truth","")}
            for c in suite]


def guess_doc_group(item):
    """Guess which source PDF this question belongs to."""
    cat = item.get("category", "")
    for doc, cats in DOC_GROUPS.items():
        if cat in cats:
            return doc
    # Fallback: keyword heuristics on question text
    q = item.get("question", "").lower()
    if any(x in q for x in ["wallboard", "виджет", "метрик", "порог", "цветов"]):
        return "Программа для ЭВМ Wallboard Mango Office Руководство пользователя.pdf"
    if any(x in q for x in ["личный кабинет", "лк", "переадресац", "ivr", "sip"]):
        return "ЛК - виртуальная атс mango office ЮТЕР ЕРЕАДРЕ СПРАВОЧНИК АБОНЕНТА.pdf"
    if any(x in q for x in ["контакт-центр", "контакт центр", "оператор", "очередь", "кц"]):
        return "КЦ - Контакт центр руководства пользователя.pdf"
    if any(x in q for x in ["контроль качества", "бланк", "оценк", "критерий"]):
        return "КОНТРОЛЬ КАЧЕСТВА Руководство пользователя.pdf"
    if any(x in q for x in ["вебхук", "webhook", "событи", "уведомлени"]):
        return "ИНТЕГРАЦИЯ MANGO OFFICE Инструкция по вебхуки MANGO OFFICE.pdf"
    if any(x in q for x in ["сервер-клиент", "websocket", "salt", "транзакц", "session"]):
        return "MANGO OFFICE Описание API Сервер клиент.pdf"
    return "API MANGO OFFICE.pdf"


def run_eval(questions, model, generate=False):
    collection = load_collection()
    results = []
    total = len(questions)

    for i, item in enumerate(questions, 1):
        q    = item["question"]
        qid  = item.get("id", f"Q{i}")
        cat  = item.get("category", "")
        doc  = item.get("doc_group") or guess_doc_group(item)

        print(f"[{i}/{total}] {qid} | {q[:70]}...", flush=True)
        t0 = time.time()

        try:
            case = {"question": q, "expected_sources": [], "expected_keywords": []}
            route, files, chunks = retrieve(collection, case, model, retrieval_only=not generate)

            answer = ""
            if generate and chunks:
                print(f"  Generating...", flush=True)
                answer = generate_answer(q, chunks, model)

            elapsed = round(time.time() - t0, 2)
            top_sources = [
                {"source": c["source"], "page": c["page"],
                 "relevance": c["relevance"], "match_type": c["match_type"],
                 "preview": c["text"][:300]}
                for c in chunks[:6]
            ]
            src_str = "; ".join(
                f"{c['source']} стр.{c['page']}"
                for c in chunks[:4]
            )
            print(f"  → {src_str} | {len(chunks)} чанков | {elapsed}s", flush=True)

            results.append({
                "id": qid, "category": cat, "doc_group": doc,
                "question": q,
                "ground_truth": item.get("ground_truth", ""),
                "categories_routed": route.get("categories", []),
                "files_searched": files,
                "top_sources": top_sources,
                "answer": answer,
                "chunks_retrieved": len(chunks),
                "elapsed_sec": elapsed,
            })

        except Exception as e:
            print(f"  ERROR: {e}", flush=True)
            results.append({
                "id": qid, "question": q, "doc_group": doc,
                "error": str(e), "answer": "", "top_sources": []
            })

    return results


def write_md_report(results, generate):
    # Group by doc
    groups = {}
    for r in results:
        doc = r.get("doc_group", "Прочее")
        groups.setdefault(doc, []).append(r)

    errors   = [r for r in results if "error" in r]
    answered = [r for r in results if r.get("answer")]

    lines = [
        "# ВЕКТОР — Массовый прогон",
        "",
        f"**Дата:** {datetime.now().strftime('%d.%m.%Y %H:%M')}  ",
        f"**Режим:** {'полный (LLM)' if generate else 'retrieval-only'}  ",
        f"**Вопросов:** {len(results)} | **С ответом:** {len(answered)} | **Ошибок:** {len(errors)}",
        "",
        "## Сводка по документам",
        "",
        "| Документ | Вопросов | С ответом |",
        "|-----------|----------|-----------|",
    ]
    for doc, items in groups.items():
        ans = sum(1 for r in items if r.get("answer"))
        short = doc.replace(".pdf", "").replace("MANGO OFFICE ", "").replace("Mango Office ", "")
        short = short[:60]
        lines.append(f"| {short} | {len(items)} | {ans} |")
    lines += [
        "",
        "> **Как валидировать:** для каждого раздела ниже — открой соответствующий PDF в Gemini,",
        "> скопируй блок `<details>` и вставь в чат. Gemini проверит каждый ответ по документу.",
        "",
        "---",
        "",
    ]

    for doc, items in sorted(groups.items()):
        lines += [f"## 📄 {doc}", ""]

        # Gemini validation block — only useful when answers were generated
        answered = [r for r in items if r.get("answer")]
        gemini_items = answered if answered else items
        lines += [
            "<details>",
            f"<summary>📋 Блок для Gemini-валидации ({len(gemini_items)} вопросов) — скопируй в Gemini вместе с PDF</summary>",
            "",
            "```",
            GEMINI_VALIDATION_PROMPT.format(doc_name=doc),
        ]
        for j, r in enumerate(gemini_items, 1):
            lines.append(f"--- Вопрос {j} ---")
            lines.append(f"Q: {r['question']}")
            if r.get("answer"):
                lines.append(f"A: {r['answer']}")
            else:
                lines.append("A: [ответ не сгенерирован — retrieval-only режим]")
            src_list = ", ".join(
                f"{s['source']} стр.{s['page']} ({s['relevance']}%)"
                for s in r.get("top_sources", [])[:3]
            )
            lines.append(f"Источники ВЕКТОР: {src_list}")
            lines.append("")
        lines += ["```", "", "</details>", ""]

        # Detailed results
        for j, r in enumerate(items, 1):
            if "error" in r:
                lines.append(f"### ❌ {r['id']} — Ошибка")
                lines.append(f"**Вопрос:** {r['question'][:200]}")
                lines.append(f"**Ошибка:** {r['error']}")
            else:
                lines.append(f"### {r['id']} ({r.get('category','')}) — {r['chunks_retrieved']} чанков")
                lines.append(f"**Вопрос:** {r['question'][:350]}")
                lines.append(f"**Роутинг:** `{', '.join(r['categories_routed'])}`")
                lines.append("")
                lines.append("**Источники:**")
                for s in r["top_sources"][:5]:
                    lines.append(f"- `{s['source']}` стр.{s['page']} "
                                 f"({s['relevance']}%, {s['match_type']})")
                    lines.append(f"  > {s['preview'][:200]}...")
                if r.get("answer"):
                    lines += ["", "**Ответ ВЕКТОР:**", "", r["answer"][:800], ""]
                if r.get("ground_truth"):
                    lines += ["**Эталон:**", r["ground_truth"][:300], ""]
            lines += ["", "---", ""]

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate", action="store_true",
                        help="Generate full LLM answers (slow, needs Ollama)")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    # Load all questions
    file_qs = extract_questions_from_file(QUESTIONS_FILE)
    suite_qs = load_suite_questions(SUITE_FILE)

    all_questions = suite_qs + [
        {"id": f"VQ{i+1}", "question": q, "category": "extracted"}
        for i, q in enumerate(file_qs)
    ]

    print(f"Вопросов из test_suite.json:        {len(suite_qs)}")
    print(f"Вопросов из 'вопросы для тренеровки': {len(file_qs)}")
    print(f"Всего: {len(all_questions)}")
    print(f"Режим: {'GENERATE (LLM)' if args.generate else 'RETRIEVAL-ONLY'}")
    print()

    results = run_eval(all_questions, args.model, generate=args.generate)

    os.makedirs(REPORT_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode  = "full" if args.generate else "retrieval"
    json_path = os.path.join(REPORT_DIR, f"mass_eval_{mode}_{stamp}.json")
    md_path   = os.path.join(REPORT_DIR, f"mass_eval_{mode}_{stamp}.md")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(write_md_report(results, args.generate))

    ok = [r for r in results if "error" not in r]
    print(f"\nГотово. Обработано: {len(results)} | Ошибок: {len(results)-len(ok)}")
    print("JSON: " + json_path)
    print("MD:   " + md_path)


if __name__ == "__main__":
    main()
