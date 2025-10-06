# server_local.py
# API para tu mapa actual, SIN tocar el front:
# - /approx_temp_local  (GET)  -> serie diaria con T, Viento, Precip/Nieve, Radiación
# - /predict            (POST) -> resumen semanal (números simples) con lluvia, viento y radiación
#
# Base: NASA POWER. Para fechas futuras (<= 2 años) aplica ajuste leve con GPT si hay OPENAI_API_KEY.
#
# Requisitos:
#   pip install flask flask-cors pandas numpy python-dotenv requests openai

import os, math, time, logging, json
from datetime import datetime, timedelta, date

import numpy as np
import pandas as pd
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

# OpenAI (Responses API)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # por si no está instalado

load_dotenv()

PORT = int(os.getenv("PORT", "8000"))
POWER_BASE = os.getenv("NASA_BASE", "https://power.larc.nasa.gov/api/temporal/daily/point")
POWER_COMMUNITY = os.getenv("NASA_COMMUNITY", "ag")   # AG -> radiación en MJ/m²/día
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

client = OpenAI(api_key=OPENAI_API_KEY) if (OpenAI and OPENAI_API_KEY) else None

# Parámetros POWER
POWER_PARAMS = [
    "T2M","T2M_MIN","T2M_MAX",
    "WS10M","WD10M",
    "PRECTOTCORR",            # mm/día
    "PRECSNO",                # mm/día (si existe)
    "ALLSKY_SFC_SW_DWN",      # MJ/m²/día
]

# Defaults del método
DEFAULT_WINDOW_DOY   = 10
DEFAULT_HORIZON_DAYS = 7
DEFAULT_HIST_YEARS   = 25

# Rieles físicos
MIN_PHYS, MAX_PHYS = -80.0, 60.0

# Cache simple
CACHE_TTL = 900
_cache = {}

app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)

# ---------------- utilidades ----------------

def clamp(x: float | None, lo=MIN_PHYS, hi=MAX_PHYS):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return None
    return float(np.clip(x, lo, hi))

def robust_p(arr, p_low=10, p_high=90):
    arr = np.array(arr, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return (None, None)
    return float(np.percentile(arr, p_low)), float(np.percentile(arr, p_high))

def is_leap(y:int) -> bool:
    return (y%4==0 and y%100!=0) or (y%400==0)

def doy_no_leap(dt:date) -> int:
    doy = dt.timetuple().tm_yday
    if dt.month == 2 and dt.day == 29:
        return 59
    if is_leap(dt.year) and doy > 59:
        return doy - 1
    return doy

def yyyymmdd(d:date) -> str:
    return d.strftime("%Y%m%d")

def backoff_get(url, params, tries=4, timeout=30):
    delay, last = 1.0, None
    for _ in range(tries):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code == 200:
                return r
            last = r
        except Exception as e:
            last = e
        time.sleep(delay)
        delay = min(8.0, delay*2)
    if isinstance(last, requests.Response):
        raise RuntimeError(f"HTTP {last.status_code}: {last.text[:300]}")
    raise RuntimeError(f"Request failed: {last}")

def circ_mean_deg(angles_deg: np.ndarray, weights: np.ndarray | None = None):
    ang = np.deg2rad(angles_deg)
    if weights is None:
        weights = np.ones_like(ang)
    S = np.sum(weights * np.sin(ang))
    C = np.sum(weights * np.cos(ang))
    if S == 0 and C == 0:
        return None
    return float((math.degrees(math.atan2(S, C)) + 360.0) % 360.0)

# ------------- POWER fetch -------------

def fetch_power_daily(lat: float, lon: float, start: date, end: date) -> pd.DataFrame:
    params = {
        "latitude": f"{lat:.5f}",
        "longitude": f"{lon:.5f}",
        "start": yyyymmdd(start),
        "end": yyyymmdd(end),
        "parameters": ",".join(POWER_PARAMS),
        "format": "JSON",
        "community": POWER_COMMUNITY,
        "user": "app",
    }

    cache_key = ("POWER", tuple(sorted(params.items())))
    now = time.time()
    cached = _cache.get(cache_key)
    if cached and (now - cached[0]) < CACHE_TTL:
        data = cached[1]
    else:
        r = backoff_get(POWER_BASE, params)
        data = r.json()
        _cache[cache_key] = (now, data)

    par = data.get("properties", {}).get("parameter", {})
    dates = sorted(set().union(*[set(v.keys()) for v in par.values()])) if par else []

    rows = []
    for ds in dates:
        dt = datetime.strptime(ds, "%Y%m%d").date()
        row = {"date": dt, "year": dt.year, "doy": doy_no_leap(dt)}
        for p in POWER_PARAMS:
            v = par.get(p, {}).get(ds, None)
            if v is None:
                row[p] = np.nan
                continue

            try:
                fv = float(v)
            except Exception:
                fv = np.nan
            else:
                # POWER usa valores negativos muy grandes como sentinela de "sin dato"
                if abs(fv) >= 900 or fv in (-8888.0, -7777.0, -6999.0, -6666.0):
                    fv = np.nan

            if p.startswith("T2M") and not np.isnan(fv):
                fv = float(np.clip(fv, MIN_PHYS, MAX_PHYS))

            row[p] = fv
        rows.append(row)

    df = pd.DataFrame(rows)

    # Si falta T2M, intenta promedio min/max
    if "T2M" in df.columns and df["T2M"].isna().all():
        if "T2M_MIN" in df.columns and "T2M_MAX" in df.columns:
            df["T2M"] = (pd.to_numeric(df["T2M_MIN"], errors="coerce") + pd.to_numeric(df["T2M_MAX"], errors="coerce"))/2.0

    # numéricos y limpieza
    for c in ["T2M","T2M_MIN","T2M_MAX","WS10M","WD10M","PRECTOTCORR","PRECSNO","ALLSKY_SFC_SW_DWN"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["T2M"])
    return df

# --------- base: climatología + tendencia ---------

def lin_pred_with_resid(years: np.ndarray, vals: np.ndarray, target_year: int):
    years = np.asarray(years, dtype=float)
    vals  = np.asarray(vals,  dtype=float)
    ok = ~np.isnan(vals)
    years, vals = years[ok], vals[ok]
    if vals.size < 8:
        if vals.size == 0: return None, None, None
        mu = float(np.mean(vals))
        p10, p90 = robust_p(vals, 10, 90)
        return mu, p10, p90
    a, b = np.polyfit(years, vals, 1)
    pred = float(a*target_year + b)
    resid = vals - (a*years + b)
    p10r, p90r = robust_p(resid, 10, 90)
    p10 = None if p10r is None else pred + p10r
    p90 = None if p90r is None else pred + p90r
    return pred, p10, p90

def estimate_week(df_all: pd.DataFrame, target_start: date, horizon_days=DEFAULT_HORIZON_DAYS, window_days=DEFAULT_WINDOW_DOY):
    out=[]
    m = pd.to_datetime(df_all["date"])
    df = df_all[~((m.dt.month==2) & (m.dt.day==29))].copy()

    for i in range(horizon_days):
        d = target_start + timedelta(days=i)
        doy = doy_no_leap(d)
        lo, hi = max(1, doy-window_days), min(365, doy+window_days)
        sub = df[(df["doy"]>=lo) & (df["doy"]<=hi)].copy()

        day = {"date": d.isoformat()}

        # Temperaturas
        for col in ["T2M","T2M_MIN","T2M_MAX"]:
            if col in sub:
                years = sub["year"].values
                vals  = sub[col].values
                pred, p10, p90 = lin_pred_with_resid(years, vals, d.year)
                pred = clamp(pred); p10 = None if p10 is None else clamp(p10); p90 = None if p90 is None else clamp(p90)
                day[col+"_est"], day[col+"_p10"], day[col+"_p90"] = pred, p10, p90

        # Viento (velocidad)
        if "WS10M" in sub:
            ws_pred, ws_p10, ws_p90 = lin_pred_with_resid(sub["year"].values, sub["WS10M"].values, d.year)
            ws_pred = None if ws_pred is None else float(max(ws_pred, 0.0))
            ws_p10  = None if ws_p10  is None else float(max(ws_p10,  0.0))
            ws_p90  = None if ws_p90  is None else float(max(ws_p90,  0.0))
            day["WS10M_est"], day["WS10M_p10"], day["WS10M_p90"] = ws_pred, ws_p10, ws_p90

        # Dirección (media circular ponderada)
        if "WD10M" in sub:
            ang = sub["WD10M"].dropna().values
            w   = sub["WS10M"].reindex(sub["WD10M"].dropna().index).fillna(1.0).values if "WS10M" in sub else None
            wd = circ_mean_deg(ang, w)
            day["WD10M_mean"] = wd

        # Precipitación (empírico, cero-inflado)
        if "PRECTOTCORR" in sub:
            pr = sub["PRECTOTCORR"].dropna().values
            if pr.size:
                pr_pred = float(np.mean(pr))
                pr_p10, pr_p90 = robust_p(pr, 10, 90)
                pr_pred = max(pr_pred, 0.0)
                pr_p10  = None if pr_p10 is None else max(pr_p10, 0.0)
                pr_p90  = None if pr_p90 is None else max(pr_p90, 0.0)
                p_precip = float(np.mean(pr>0.0))
            else:
                pr_pred=pr_p10=pr_p90=p_precip=None
            day["PRECTOTCORR_est"], day["PRECTOTCORR_p10"], day["PRECTOTCORR_p90"] = pr_pred, pr_p10, pr_p90
            day["P_PRECIP"] = p_precip

        # Nieve (POWER si hay PRECSNO; si no, heurística T<=0)
        if "PRECSNO" in sub and not sub["PRECSNO"].dropna().empty:
            sn = sub["PRECSNO"].dropna().values
            sn_pred = float(np.mean(sn))
            sn_p10, sn_p90 = robust_p(sn, 10, 90)
            sn_pred = max(sn_pred, 0.0)
            sn_p10  = None if sn_p10 is None else max(sn_p10, 0.0)
            sn_p90  = None if sn_p90 is None else max(sn_p90, 0.0)
            day["SNOW_mm_est"], day["SNOW_mm_p10"], day["SNOW_mm_p90"] = sn_pred, sn_p10, sn_p90
            day["SNOW_flag"] = "power"
        else:
            t = day.get("T2M_est", None)
            pr_pred = day.get("PRECTOTCORR_est", None)
            snow = (pr_pred if (t is not None and pr_pred is not None and t<=0.0) else 0.0) if pr_pred is not None else None
            day["SNOW_mm_est"] = snow
            day["SNOW_flag"] = "heuristic_t<=0C"

        # Radiación (MJ/m²/día). También devolvemos kWh/m²/día derivado
        if "ALLSKY_SFC_SW_DWN" in sub:
            rad_pred, rad_p10, rad_p90 = lin_pred_with_resid(sub["year"].values, sub["ALLSKY_SFC_SW_DWN"].values, d.year)
            day["ALLSKY_SFC_SW_DWN_est"] = None if rad_pred is None else float(rad_pred)
            day["ALLSKY_SFC_SW_DWN_p10"] = rad_p10 if rad_p10 is None else float(rad_p10)
            day["ALLSKY_SFC_SW_DWN_p90"] = rad_p90 if rad_p90 is None else float(rad_p90)
            # conversión a kWh/m²/día
            if day["ALLSKY_SFC_SW_DWN_est"] is not None:
                day["RAD_kWhm2_est"] = day["ALLSKY_SFC_SW_DWN_est"] / 3.6
            if day["ALLSKY_SFC_SW_DWN_p10"] is not None:
                day["RAD_kWhm2_p10"] = day["ALLSKY_SFC_SW_DWN_p10"] / 3.6
            if day["ALLSKY_SFC_SW_DWN_p90"] is not None:
                day["RAD_kWhm2_p90"] = day["ALLSKY_SFC_SW_DWN_p90"] / 3.6

        # Rango resumen en T
        lo, hi = day.get("T2M_p10"), day.get("T2M_p90")
        day["summary_range"] = [lo, hi] if (lo is not None and hi is not None) else None

        out.append(day)

    return out

# --------- refinamiento GPT (solo futuro) ---------

def refine_with_gpt(lat, lon, week_days, meta):
    if client is None:
        return week_days, {"gpt_used": False, "reason": "OPENAI_API_KEY missing"}

    baseline = [{"date": d["date"],
                 "T2M_est": d.get("T2M_est"),
                 "T2M_p10": d.get("T2M_p10"),
                 "T2M_p90": d.get("T2M_p90"),
                 "WS10M_est": d.get("WS10M_est"),
                 "WS10M_p10": d.get("WS10M_p10"),
                 "WS10M_p90": d.get("WS10M_p90"),
                 "WD10M_mean": d.get("WD10M_mean"),
                 "PRECTOTCORR_est": d.get("PRECTOTCORR_est"),
                 "SNOW_mm_est": d.get("SNOW_mm_est"),
                 "ALLSKY_SFC_SW_DWN_est": d.get("ALLSKY_SFC_SW_DWN_est"),
                 "RAD_kWhm2_est": d.get("RAD_kWhm2_est")} for d in week_days]

    rules = {
        "temperature_delta_celsius_range": [-1.5, 1.5],
        "wind_speed_scale_range": [0.85, 1.15],
        "precip_scale_range": [0.70, 1.30],
        "radiation_scale_range": [0.85, 1.15],
        "non_negative": ["WS10M_est","PRECTOTCORR_est","SNOW_mm_est","ALLSKY_SFC_SW_DWN_est","RAD_kWhm2_est"],
        "keep_dates_order": True,
        "clamp_temp_c": [MIN_PHYS, MAX_PHYS]
    }

    prompt = (
        "Eres un asistente meteorológico. Recibirás una serie diaria base para una semana futura."
        "Devuelve SOLO JSON con el MISMO esquema de 'days', aplicando ajustes leves plausibles y coherentes:\n"
        f"- Lat: {lat}, Lon: {lon}\n"
        f"- Reglas: {json.dumps(rules)}\n"
        "1) Ajusta T manteniendo p10 ≤ est ≤ p90.\n"
        "2) Escala velocidad del viento; deja dirección igual.\n"
        "3) Escala precip y nieve sin negativos.\n"
        "4) Escala radiación (MJ y kWh derivado si está).\n"
        "5) Conserva fechas y claves. Sin texto fuera del JSON.\n"
        "Formato:\n"
        "{ \"days\": [ { ... } ], \"note\": \"<200c>\" }"
    )

    try:
        resp = client.responses.create(
            model="gpt-4o-mini",
            temperature=0.2,
            input=prompt + "\n\n### BASELINE\n" + json.dumps({"days": baseline}, ensure_ascii=False)
        )
        txt = resp.output_text.strip()
        data = json.loads(txt)
        if not isinstance(data, dict) or "days" not in data or not isinstance(data["days"], list):
            raise ValueError("Estructura inesperada")
        if len(data["days"]) != len(week_days):
            raise ValueError("Longitud no coincide")
        for a, b in zip(data["days"], week_days):
            if a.get("date") != b.get("date"):
                raise ValueError("Fechas no coinciden")

        adj = []
        for a in data["days"]:
            a["T2M_est"] = clamp(a.get("T2M_est"))
            a["T2M_p10"] = None if a.get("T2M_p10") is None else clamp(a["T2M_p10"])
            a["T2M_p90"] = None if a.get("T2M_p90") is None else clamp(a["T2M_p90"])
            for k in ["WS10M_est","WS10M_p10","WS10M_p90","PRECTOTCORR_est","SNOW_mm_est",
                      "ALLSKY_SFC_SW_DWN_est","RAD_kWhm2_est"]:
                if k in a and a[k] is not None:
                    a[k] = float(max(0.0, a[k]))
            adj.append(a)

        return adj, {"gpt_used": True, "note": data.get("note", "")}
    except Exception as e:
        app.logger.warning(f"GPT refine fallback: {e}")
        return week_days, {"gpt_used": False, "reason": str(e)}

# ---------------- endpoints ----------------

# === ENDPOINT: CHATBOT (usa Responses API para mayor compatibilidad) ===
@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json(force=True)
        user_message = (data.get("message") or "").strip()
        if not user_message:
            return jsonify({"reply": "No recibí ningún mensaje."}), 400

        if client is None:
            return jsonify({"reply": "Falta la API key de OpenAI en el entorno."}), 500

        system = "Sos un asistente meteorológico del proyecto Plani 🌎. Respondé en español con claridad y datos breves."
        response = client.responses.create(
            model="gpt-4o-mini",
            temperature=0.3,
            input=f"{system}\n\nUsuario: {user_message}\nAsistente:"
        )
        reply = response.output_text.strip()
        return jsonify({"reply": reply})
    except Exception as e:
        return jsonify({"reply": f"Error interno: {str(e)}"}), 500

@app.get("/")
def root():
    return jsonify({"ok": True, "msg": "API NASA POWER + (opcional) GPT. Endpoints: /approx_temp_local (GET), /predict (POST)"}), 200

@app.get("/approx_temp_local")
def approx_temp_local():
    # coords
    try:
        lat = float(request.args.get("lat"))
        lon = float(request.args.get("lon"))
    except Exception:
        return jsonify({"error": "lat/lon inválidos"}), 400

    start_str = request.args.get("start")
    if not start_str:
        return jsonify({"error": "falta 'start' (YYYY-MM-DD)"}), 400
    try:
        target_start = datetime.strptime(start_str, "%Y-%m-%d").date()
    except Exception:
        return jsonify({"error": "formato de 'start' inválido (YYYY-MM-DD)"}), 400

    today = date.today()
    if not (today <= target_start <= today + timedelta(days=730)):
        return jsonify({"error": "la fecha debe estar entre hoy y 2 años desde hoy"}), 400

    days = int(request.args.get("days", str(DEFAULT_HORIZON_DAYS)))
    window = int(request.args.get("window", str(DEFAULT_WINDOW_DOY)))
    hist_years = int(request.args.get("hist_years", str(DEFAULT_HIST_YEARS)))
    radius = request.args.get("radius")  # ignorado (compat UI)
    k = request.args.get("k")            # ignorado (compat UI)
    drift = request.args.get("drift")    # ignorado (compat UI)

    start_hist = date(today.year - hist_years, 1, 1) - timedelta(days=40)
    end_hist   = today

    try:
        df = fetch_power_daily(lat, lon, start_hist, end_hist)
        if df.empty:
            return jsonify({"error": "POWER sin datos para ese punto/periodo"}), 502

        week = estimate_week(df, target_start, horizon_days=days, window_days=window)

        gpt_meta = {}
        if target_start > today:
            week, gpt_meta = refine_with_gpt(lat, lon, week, {"lat": lat, "lon": lon, "start": start_str})

        years_present = sorted(pd.to_datetime(df["date"]).dt.year.unique().tolist())
        meta = {
            "lat": lat, "lon": lon,
            "target_start": start_str,
            "horizon_days": days,
            "window_days": window,
            "hist_years_requested": hist_years,
            "hist_years": years_present,
            "source": "NASA POWER (temporal/daily/point)",
            "vars": POWER_PARAMS,
            "units": {
                "T2M": "°C",
                "WS10M": "m/s",
                "WD10M": "grados",
                "PRECTOTCORR": "mm/día",
                "PRECSNO": "mm/día",
                "ALLSKY_SFC_SW_DWN": "MJ/m²/día",
                "RAD_kWhm2": "kWh/m²/día"
            },
            "note": "Aproximación estadística; no es pronóstico determinista.",
            "compat": {"radius_ignored": radius, "k_ignored": k, "drift_ignored": drift},
            "gpt": gpt_meta
        }

        return jsonify({"meta": meta, "days": week}), 200

    except Exception as e:
        app.logger.exception("approx_temp_local failed")
        return jsonify({"error": str(e)}), 500

# --- endpoint compat para tu mapa: POST /predict ---------------------------
@app.route("/predict", methods=["POST", "OPTIONS"])
def predict_compat():
    # Preflight CORS
    if request.method == "OPTIONS":
        return ("", 204)

    try:
        data = request.get_json(silent=True) or {}
        # JSON esperado: { lat, lon, fecha, radio? }
        lat = float(data.get("lat"))
        lon = float(data.get("lon"))
        start_str = data.get("fecha") or data.get("start") or data.get("date")
        if not start_str:
            return jsonify({"error": "Falta 'fecha' (YYYY-MM-DD)"}), 400

        # Permitimos pasado/presente/futuro; si es futuro (<=2 años) aplicamos GPT
        target_start = datetime.strptime(start_str, "%Y-%m-%d").date()
        today = date.today()
        if target_start > today + timedelta(days=730):
            return jsonify({"error": "la fecha no puede exceder 2 años"}), 400

        # Históricos
        hist_years = DEFAULT_HIST_YEARS
        start_hist = date(today.year - hist_years, 1, 1) - timedelta(days=40)
        end_hist = today if target_start > today else target_start
        df = fetch_power_daily(lat, lon, start_hist, end_hist)
        if df.empty:
            return jsonify({"error": "POWER sin datos para ese punto/periodo"}), 502

        # Serie base semana
        week = estimate_week(df, target_start, horizon_days=7, window_days=DEFAULT_WINDOW_DOY)

        # Ajuste GPT si futuro
        gpt_meta = {}
        if target_start > today:
            week, gpt_meta = refine_with_gpt(lat, lon, week, {"lat": lat, "lon": lon, "start": start_str})

        # Agregados semanales (para tu UI)
        def agg_mean(key):
            vals = [d.get(key) for d in week if d.get(key) is not None]
            return round(float(np.mean(vals)), 2) if vals else None

        def agg_min(key):
            vals = [d.get(key) for d in week if d.get(key) is not None]
            return round(float(np.min(vals)), 2) if vals else None

        def agg_max(key):
            vals = [d.get(key) for d in week if d.get(key) is not None]
            return round(float(np.max(vals)), 2) if vals else None

        t_mean = agg_mean("T2M_est")
        t_min  = agg_min("T2M_MIN_est") or agg_min("T2M_est")
        t_max  = agg_max("T2M_MAX_est") or agg_max("T2M_est")
        viento = agg_mean("WS10M_est")
        viento_p10 = agg_mean("WS10M_p10")
        viento_p90 = agg_mean("WS10M_p90")
        lluvia = agg_mean("PRECTOTCORR_est")
        lluvia_p10 = agg_mean("PRECTOTCORR_p10")
        lluvia_p90 = agg_mean("PRECTOTCORR_p90")
        nieve  = agg_mean("SNOW_mm_est")
        rad_mj = agg_mean("ALLSKY_SFC_SW_DWN_est")

        # W/m² promedio diario (>=0)
        rad_wm2 = None
        if rad_mj is not None:
            rad_wm2 = max(0.0, round(rad_mj / 0.0864, 2))  # 1 W/m² = 0.0864 MJ/m²/día

        # Evitar negativos por seguridad
        if lluvia is not None and lluvia < 0: lluvia = 0.0
        if nieve  is not None and nieve  < 0: nieve  = 0.0
        if viento is not None and viento < 0: viento = 0.0

        resp = {
            "lat": lat,
            "lon": lon,
            "desde": start_str,
            "hasta": (target_start + timedelta(days=7)).isoformat(),

            # TEMP como número simple (tu UI lo imprime directo)
            "temperatura": round(t_mean, 2) if t_mean is not None else None,

            # ✅ añadidos: lluvia, viento, radiación para el panel
            "lluvia": lluvia,            # mm/día promedio semana
            "viento": viento,            # m/s promedio semana
            "radiacion": rad_wm2,        # W/m² promedio semana (no negativo)
            "nieve": nieve,              # mm/día promedio semana

            # Extras no disruptivos (por si tu chatbot los usa)
            "extra": {
                "temperatura_min": t_min,
                "temperatura_max": t_max,
                "viento_p10": viento_p10,
                "viento_p90": viento_p90,
                "lluvia_p10": lluvia_p10,
                "lluvia_p90": lluvia_p90,
                "radiacion_MJm2": rad_mj,
                "radiacion_kWhm2": None if rad_mj is None else round(rad_mj/3.6, 2)
            },

            "modo": "proyección GPT" if gpt_meta.get("gpt_used") else ("histórico" if target_start <= today else "estadístico"),
            "nota": gpt_meta.get("note", "Estimación NASA POWER; ajustes leves si futuro.")
        }
        return jsonify(resp), 200

    except Exception as e:
        app.logger.exception("predict_compat failed")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=False)
