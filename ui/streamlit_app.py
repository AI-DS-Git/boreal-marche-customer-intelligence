"""Boréal Marché Customer Intelligence -- Streamlit front end (five sections, EN/FR).

Talks to the FastAPI service only (API_URL); it holds no model or data of its own.
"""
import json
import os
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

API = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")
EXAMPLE = json.loads((Path(__file__).parent / "example_customer.json").read_text())

T = {
    "en": {
        "title": "Boréal Marché Customer Intelligence", "nav": ["Prediction", "Basket & recommendations", "Assistant", "Monitoring", "About"],
        "lang": "Language", "api_down": "The API is not reachable right now.",
        "pred_h": "Retention prediction", "pred_help": "Edit the customer's profile; the probability updates live.",
        "recency": "Days since last order", "orders": "Number of orders", "spend": "Total spend (CAD)",
        "retain": "Probability the customer is retained", "risk": "Churn risk", "decision": "Recommended action",
        "target": "Send a retention offer (a staff member reviews it)", "none": "No action needed",
        "threshold": "Decision threshold", "model": "Model version",
        "why_h": "Main factors behind this score (for staff, not a customer-facing reason)", "why_up": "raises retention", "why_dn": "lowers retention",
        "reco_h": "Basket and recommendations", "add": "Add product code", "add_btn": "Add", "clear": "Clear basket",
        "basket": "Basket", "empty": "The basket is empty; showing the most popular items.", "because": "because you added",
        "chat_h": "Assistant", "provider": "Provider", "ask": "Ask about policies, products or recommendations",
        "sources": "Sources", "tools": "Tools used",
        "mon_h": "Monitoring", "overall": "Overall status", "window": "Window (hours)", "refresh": "Refresh",
        "volume": "Requests", "errs": "Error rate", "lat": "Latency (ms)", "fire": "Recommendation rule fire-rate",
        "drift_h": "Input drift (PSI and KS vs training)", "pred_drift": "Prediction drift (PSI)", "gt_h": "Delayed ground truth",
        "alerts_h": "Recent alerts", "thr": "Thresholds: PSI amber >= 0.10, red >= 0.25; error rate red >= 5%; p95 latency red >= 500 ms.",
        "about_h": "About", "card": "Model card", "privacy": "Privacy notice", "prov": "Data provenance", "arch": "Architecture",
    },
    "fr": {
        "title": "Intelligence client Boréal Marché", "nav": ["Prédiction", "Panier et recommandations", "Assistant", "Surveillance", "À propos"],
        "lang": "Langue", "api_down": "L'API est injoignable pour le moment.",
        "pred_h": "Prédiction de rétention", "pred_help": "Modifiez le profil du client; la probabilité se met à jour en direct.",
        "recency": "Jours depuis la dernière commande", "orders": "Nombre de commandes", "spend": "Dépenses totales (CAD)",
        "retain": "Probabilité que le client soit retenu", "risk": "Risque de départ", "decision": "Action recommandée",
        "target": "Envoyer une offre de rétention (révisée par un membre du personnel)", "none": "Aucune action requise",
        "threshold": "Seuil de décision", "model": "Version du modèle",
        "why_h": "Principaux facteurs de ce score (pour le personnel, pas une raison à donner au client)", "why_up": "augmente la rétention", "why_dn": "diminue la rétention",
        "reco_h": "Panier et recommandations", "add": "Ajouter un code produit", "add_btn": "Ajouter", "clear": "Vider le panier",
        "basket": "Panier", "empty": "Le panier est vide; affichage des articles les plus populaires.", "because": "parce que vous avez ajouté",
        "chat_h": "Assistant", "provider": "Fournisseur", "ask": "Posez une question sur les politiques, les produits ou les recommandations",
        "sources": "Sources", "tools": "Outils utilisés",
        "mon_h": "Surveillance", "overall": "État global", "window": "Fenêtre (heures)", "refresh": "Actualiser",
        "volume": "Requêtes", "errs": "Taux d'erreur", "lat": "Latence (ms)", "fire": "Taux d'activation des règles",
        "drift_h": "Dérive des entrées (PSI et KS vs entraînement)", "pred_drift": "Dérive des prédictions (PSI)", "gt_h": "Vérité terrain différée",
        "alerts_h": "Alertes récentes", "thr": "Seuils : PSI ambre >= 0,10, rouge >= 0,25; taux d'erreur rouge >= 5 %; latence p95 rouge >= 500 ms.",
        "about_h": "À propos", "card": "Fiche du modèle", "privacy": "Avis de confidentialité", "prov": "Provenance des données", "arch": "Architecture",
    },
}
LIGHT = {"green": "🟢", "amber": "🟠", "red": "🔴", "insufficient_data": "⚪"}

st.set_page_config(page_title="Boréal Marché", page_icon="🛒", layout="wide")
lang = st.sidebar.radio("Language / Langue", ["en", "fr"], format_func=lambda x: {"en": "English", "fr": "Français"}[x], horizontal=True)
t = T[lang]
st.sidebar.title(t["title"])
page = st.sidebar.radio("", t["nav"], label_visibility="collapsed")
idx = t["nav"].index(page)


def call(method, path, **kw):
    try:
        r = requests.request(method, API + path, timeout=20, **kw)
        return r.status_code, r.json()
    except Exception:
        return 0, None


def stream_chat(message, language, provider):
    """Yields the server-sent events of POST /chat/stream as they arrive (true token streaming)."""
    try:
        with requests.post(API + "/chat/stream", json={"message": message, "language": language,
                                                       "provider": provider},
                           stream=True, timeout=(5, 120)) as r:
            if r.status_code != 200:
                yield {"type": "error", "detail": r.text[:300]}
                return
            for line in r.iter_lines(chunk_size=None, decode_unicode=True):
                if line and line.startswith("data: "):
                    yield json.loads(line[6:])
    except requests.RequestException as exc:
        yield {"type": "error", "detail": type(exc).__name__}


def need_api():
    code, _ = call("GET", "/health")
    if code != 200:
        st.error(t["api_down"])
        st.stop()


# ---- 1. Prediction with a live what-if control ---------------------------------------
if idx == 0:
    need_api()
    st.header(t["pred_h"])
    st.caption(t["pred_help"])
    f = dict(EXAMPLE)
    c1, c2, c3 = st.columns(3)
    f["recency_days"] = c1.slider(t["recency"], 0, 700, int(EXAMPLE["recency_days"]))
    f["frequency_orders"] = c2.slider(t["orders"], 1, 60, int(EXAMPLE["frequency_orders"]))
    f["monetary_total"] = float(c3.slider(t["spend"], 0, 20000, int(EXAMPLE["monetary_total"]), step=50))
    # Keep dates and derived fields consistent with what the user edited: recency is measured
    # from a fixed snapshot date, so moving it moves the last-purchase date and the lifetime.
    first = date.fromisoformat(EXAMPLE["first_purchase_date"])
    snapshot = date.fromisoformat(EXAMPLE["last_purchase_date"]) + timedelta(days=int(EXAMPLE["recency_days"]))
    last = max(first, snapshot - timedelta(days=f["recency_days"]))
    f["last_purchase_date"] = last.isoformat()
    offset = int(EXAMPLE["customer_lifetime_days"]) - (date.fromisoformat(EXAMPLE["last_purchase_date"]) - first).days
    f["customer_lifetime_days"] = max(0, (last - first).days + offset)
    f["monetary_mean_order"] = f["revenue_per_order"] = round(f["monetary_total"] / f["frequency_orders"], 2)
    f["transaction_count"] = max(f["transaction_count"], f["frequency_orders"])
    code, res = call("POST", "/predict", json={"features": f})
    if code == 200:
        a, b, c = st.columns(3)
        a.metric(t["retain"], f"{res['retain_probability']:.1%}")
        b.metric(t["risk"], f"{res['churn_risk']:.1%}")
        c.metric(t["threshold"], f"{res['threshold']:.2f}")
        (st.warning if res["decision"] == "target_with_offer" else st.success)(f"**{t['decision']}:** {t['target'] if res['decision']=='target_with_offer' else t['none']}")
        st.caption(f"{t['model']}: {res['model_version']} · request {res['request_id']}")
        ecode, ex = call("POST", "/explain", json={"features": f, "k": 3})
        if ecode == 200 and ex["factors"]:
            st.markdown(f"**{t['why_h']}**")
            for fa in ex["factors"]:
                up = fa["direction"] == "raises_retention"
                st.write(f"{'▲' if up else '▼'} **{fa['group']}** ({fa['detail']}) → {t['why_up'] if up else t['why_dn']} ({fa['effect'] * 100:+.1f} pts)")
    else:
        st.error(json.dumps(res))

# ---- 2. Basket and recommendations (state persisted in the session) --------------------
elif idx == 1:
    need_api()
    st.header(t["reco_h"])
    st.session_state.setdefault("basket", [])
    c1, c2 = st.columns([3, 1])
    code_in = c1.text_input(t["add"], placeholder="85123A").strip().upper()
    if c2.button(t["add_btn"]) and code_in and code_in not in st.session_state.basket:
        st.session_state.basket.append(code_in)
    if st.button(t["clear"]):
        st.session_state.basket = []
    st.subheader(t["basket"])
    st.write(", ".join(st.session_state.basket) or "—")
    code, res = call("POST", "/recommend", json={"basket": st.session_state.basket, "k": 5})
    if code == 200:
        if not st.session_state.basket:
            st.info(t["empty"])
        for r in res["recommendations"]:
            why = f" — {t['because']} **{r['because_of_description'] or r['because_of']}**" if r.get("because_of") else ""
            extra = f" (lift {r['lift']:.1f}, confidence {r['confidence']:.0%})" if r.get("lift") else ""
            st.markdown(f"- **{r['item']}** {r.get('description') or ''}{why}{extra}")

# ---- 3. Assistant (streamed rendering, citations, provider selector) -------------------
elif idx == 2:
    need_api()
    st.header(t["chat_h"])
    provider = st.sidebar.selectbox(t["provider"], ["rule", "openai", "ollama"])
    st.session_state.setdefault("chat", [])
    for m in st.session_state.chat:
        with st.chat_message(m["role"]):
            st.write(m["text"])
            if m.get("cites"):
                st.caption(t["sources"] + ": " + "; ".join(f"{c['title']} ({c['score']:.2f})" for c in m["cites"]))
            if m.get("tools"):
                st.caption(t["tools"] + ": " + ", ".join(x["tool"] for x in m["tools"]))
    q = st.chat_input(t["ask"])
    if q:
        st.session_state.chat.append({"role": "user", "text": q})
        with st.chat_message("user"):
            st.write(q)
        with st.chat_message("assistant"):
            box, shown, final = st.empty(), "", None
            for ev in stream_chat(q, "auto", provider):  # answer in the language the user typed, not the UI language
                if ev["type"] == "token":
                    shown += ev["text"]
                    box.write(shown + " \u258c")
                elif ev["type"] == "replace":  # a guardrail or the answer verifier overrode the model
                    shown = ev["text"]
                    box.write(shown)
                elif ev["type"] == "final":
                    final = ev
                elif ev["type"] == "error":
                    st.error(ev["detail"])
            if final:
                box.write(final["answer"])
                if final["citations"]:
                    st.caption(t["sources"] + ": " + "; ".join(f"{c['title']} ({c['score']:.2f})" for c in final["citations"]))
                if final["tool_calls"]:
                    st.caption(t["tools"] + ": " + ", ".join(x["tool"] for x in final["tool_calls"]))
                st.session_state.chat.append({"role": "assistant", "text": final["answer"],
                                              "cites": final["citations"], "tools": final["tool_calls"]})

# ---- 4. Monitoring --------------------------------------------------------------------
elif idx == 3:
    need_api()
    st.header(t["mon_h"])
    hours = st.sidebar.number_input(t["window"], 1, 720, 24)
    st.sidebar.button(t["refresh"])
    code, s = call("GET", "/monitoring/summary", params={"hours": hours})
    if code != 200:
        st.error(json.dumps(s))
        st.stop()
    st.subheader(f"{t['overall']}: {LIGHT[s['overall_status']]} {s['overall_status']}")
    o = s["ops"]
    a, b, c, d = st.columns(4)
    a.metric(t["volume"], o["request_count"])
    b.metric(t["errs"], f"{o['error_rate']:.1%}", help=LIGHT[o["error_status"]])
    lat = o["latency_ms"]
    c.metric(t["lat"] + " p50/p95/p99", " / ".join("–" if lat[k] is None else f"{lat[k]:.0f}" for k in ("p50", "p95", "p99")), help=LIGHT[o["latency_status"]])
    d.metric(t["fire"], "–" if o["rule_fire_rate"] is None else f"{o['rule_fire_rate']:.0%}")
    st.caption(t["thr"])

    st.subheader(f"{t['drift_h']} {LIGHT[s['drift']['status']]}")
    st.caption(f"n = {s['drift']['n_predictions']}")
    feats = s["drift"].get("features", [])
    if feats:
        df = pd.DataFrame(feats).sort_values("psi", ascending=False)
        df.insert(0, "", df["drift_status"].map(LIGHT))
        st.dataframe(df[["", "feature", "psi", "ks_statistic", "ks_pvalue", "out_of_range_share", "null_rate_delta"]], hide_index=True, use_container_width=True)
        st.metric(t["pred_drift"], s["drift"]["prediction_psi"], help=LIGHT[s["drift"]["prediction_status"]])
    else:
        st.info("Not enough traffic yet for drift statistics (minimum 30 predictions in the window).")

    g = s["ground_truth"]
    st.subheader(t["gt_h"])
    st.write(g["note"])
    st.write(f"First true accuracy signal: **{g['first_true_accuracy_signal_on'] or 'n/a (no predictions yet)'}** · labelled so far: {g['accuracy']['n_labelled'] if g['accuracy'] else 0}")
    st.subheader(t["alerts_h"])
    if s["alerts"]:
        st.dataframe(pd.DataFrame(s["alerts"])[["ts", "severity", "message"]], hide_index=True, use_container_width=True)
    else:
        st.write("—")

# ---- 5. About -------------------------------------------------------------------------
else:
    st.header(t["about_h"])
    code, v = call("GET", "/version")
    if code == 200:
        st.caption(f"Deployed commit {v['git_sha']} · model {v['model_version']} · {v['rules_count']} rules")
    about = Path(__file__).parent / f"about_{lang}.md"
    tabs = st.tabs([t["card"], t["privacy"], t["prov"], t["arch"]])
    raw = about.read_text(encoding="utf-8").lstrip() if about.exists() else ""
    sections = [s for s in ("\n" + raw).split("\n## ") if s.strip()]  # one entry per "## heading" block
    for i, tab in enumerate(tabs):
        with tab:
            st.markdown(("## " + sections[i]) if i < len(sections) else "_Content pending._")
