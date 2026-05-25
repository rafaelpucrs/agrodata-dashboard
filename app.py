import os
import io
import hmac
import uuid
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt



# CONFIG

st.set_page_config(page_title="AgroData — Irrigação", layout="wide")



# CONSTANTES

APP_VERSION = "1.0.2"
APP_TITLE = "AgroData — Irrigação (BI + Data Science + IA)"
APP_SUBTITLE = (
    "Protótipo acadêmico para o TCC: monitoramento operacional da irrigação com KPIs, "
    "alertas e recomendações explicáveis + simulação (baseline vs otimizado) + modelo econômico."
)

DATA_CSV_PATH = os.path.join("data", "dados_irrigacao.csv")
ARQ_IMG_PATH = os.path.join("data", "arquitetura_irrigacao.png")

LOG_DIR = "logs"
ACCESS_LOG_PATH = os.path.join(LOG_DIR, "access_log.csv")
REC_LOG_PATH = os.path.join(LOG_DIR, "recommendations_log.csv")
RES_LOG_PATH = os.path.join(LOG_DIR, "resultados_piloto.csv")
ECO_LOG_PATH = os.path.join(LOG_DIR, "modelo_economico.csv")
ECO_CENARIOS_LOG_PATH = os.path.join(LOG_DIR, "cenarios_vpl_fenologia.csv")
DQ_LOG_PATH = os.path.join(LOG_DIR, "data_quality_log.csv")

# Relatórios do avaliador
EVAL_UC_REPORT_PREFIX = os.path.join(LOG_DIR, "relatorio_avaliador_uc")
EVAL_GRAF_RS_PREFIX = os.path.join(LOG_DIR, "grafico_economia_rs")
EVAL_GRAF_KWH_PREFIX = os.path.join(LOG_DIR, "grafico_economia_kwh")

# Defaults (tarifas)
DEFAULT_TARIFA_FIXA = 0.82  # R$/kWh
DEFAULT_TARIFA_FP = 0.75
DEFAULT_TARIFA_PONTA = 1.45
DEFAULT_PONTA_INICIO = 18
DEFAULT_PONTA_FIM = 21

# Defaults (econômico)
DEFAULT_TAXA_DESCONTO = 0.12
DEFAULT_HORIZONTE_ANOS = 5
DEFAULT_ALPHA_RECEITA = 0.20      # % da economia bruta (no período) que vira receita
DEFAULT_PCT_CAPTURA = 0.20        # % da economia mensal para precificação por valor

# Defaults (unit economics)
DEFAULT_OPEX_MENSAL = 9000.0      # custos fixos (mês) do projeto (equipe, cloud, etc.)
DEFAULT_CV_POR_CLIENTE = 180.0    # custo variável por cliente/mês
DEFAULT_OVERHEAD_PCT = 0.10       # impostos/despesas administrativas (% do preço)
DEFAULT_N_CLIENTES = 25           # base ativa (cenário)

# Simulação
DEFAULT_N_DIAS = 90
DEFAULT_UCS = ["UC01", "UC02", "UC03", "UC04", "UC05", "UC06"]



# SESSÃO — ID de execução (evidência)

if "run_id" not in st.session_state:
    st.session_state["run_id"] = str(uuid.uuid4())[:8]
RUN_ID = st.session_state["run_id"]



# UTIL: Credenciais 

def get_settings():

    app_user = str(st.secrets.get("APP_USER", os.getenv("APP_USER", "admin")))
    app_pass = str(st.secrets.get("APP_PASSWORD", os.getenv("APP_PASSWORD", "admin")))
    evaluator = str(st.secrets.get("APP_EVALUATOR_NAME", os.getenv("APP_EVALUATOR_NAME", "Avaliador")))
    return app_user, app_pass, evaluator


def _safe_mkdir(path: str):
    os.makedirs(path, exist_ok=True)


def _now_iso():
    return datetime.now().isoformat(timespec="seconds")


def log_access_event(event: str, username: str, evaluator: str):
    _safe_mkdir(LOG_DIR)
    ts = _now_iso()
    header = "timestamp,event,user,evaluator,app_version,run_id\n"
    row = f"{ts},{event},{username},{evaluator},{APP_VERSION},{RUN_ID}\n"

    if not os.path.exists(ACCESS_LOG_PATH):
        with open(ACCESS_LOG_PATH, "w", encoding="utf-8") as f:
            f.write(header)

    with open(ACCESS_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(row)


def log_recommendation_event(
    nivel: str,
    mensagens: list,
    username: str,
    evaluator: str,
    periodo_label: str,
    fase: str,
    manejo: str,
    tipo_solo: str,
    risco_frio: bool,
    dias_pos_floracao: int,
    meta: dict,
    uc_id: str,
):
    _safe_mkdir(LOG_DIR)
    ts = _now_iso()
    mensagens_txt = " | ".join([str(m).replace("\n", " ").strip() for m in mensagens])

    row = {
        "timestamp": ts,
        "nivel": nivel,
        "mensagens": mensagens_txt,
        "user": username,
        "evaluator": evaluator,
        "uc_id": str(uc_id),
        "periodo": periodo_label,
        "fase": fase,
        "manejo": manejo,
        "tipo_solo": tipo_solo,
        "risco_frio": int(bool(risco_frio)),
        "dias_pos_floracao": int(dias_pos_floracao),
        "app_version": APP_VERSION,
        "run_id": RUN_ID,
        # estado atual
        "bomba_atual": int(meta.get("bomba_atual", 0)),
        "lamina_atual_cm": float(meta.get("lamina_atual", np.nan)),
        "chuva_24h_mm": float(meta.get("chuva_24h", 0.0)),
        # mini-resumo 6h
        "bomba_horas_6h": int(meta.get("bomba_horas_6h", 0)),
        "energia_6h_kwh": float(meta.get("energia_6h", 0.0)),
        "volume_6h_m3": float(meta.get("volume_6h", 0.0)),
        "ef_6h_kwh_m3": float(meta.get("eficiencia_6h", np.nan)) if meta.get("eficiencia_6h") is not None else np.nan,
        # mini-resumo 24h
        "bomba_horas_24h": int(meta.get("bomba_horas_24h", 0)),
        "energia_24h_kwh": float(meta.get("energia_24h", 0.0)),
        "volume_24h_m3": float(meta.get("volume_24h", 0.0)),
        "ef_24h_kwh_m3": float(meta.get("eficiencia_24h", np.nan)) if meta.get("eficiencia_24h") is not None else np.nan,
        # variação de lâmina em 24h
        "lamina_min_24h_cm": float(meta.get("lamina_min_24h", np.nan)),
        "lamina_max_24h_cm": float(meta.get("lamina_max_24h", np.nan)),
        # baseline eficiência
        "baseline_ef_kwh_m3": float(meta.get("baseline_ef", np.nan)) if meta.get("baseline_ef") is not None else np.nan,
    }

    df_row = pd.DataFrame([row])
    if not os.path.exists(REC_LOG_PATH):
        df_row.to_csv(REC_LOG_PATH, index=False, encoding="utf-8")
    else:
        df_row.to_csv(REC_LOG_PATH, mode="a", header=False, index=False, encoding="utf-8")

    return REC_LOG_PATH


def log_data_quality_event(username: str, evaluator: str, periodo_label: str, metrics: dict, uc_id: str):
    _safe_mkdir(LOG_DIR)
    ts = _now_iso()
    row = {
        "timestamp": ts,
        "user": username,
        "evaluator": evaluator,
        "uc_id": str(uc_id),
        "periodo": periodo_label,
        "app_version": APP_VERSION,
        "run_id": RUN_ID,
        **metrics,
    }
    df_row = pd.DataFrame([row])
    if not os.path.exists(DQ_LOG_PATH):
        df_row.to_csv(DQ_LOG_PATH, index=False, encoding="utf-8")
    else:
        df_row.to_csv(DQ_LOG_PATH, mode="a", header=False, index=False, encoding="utf-8")
    return DQ_LOG_PATH


def clear_logs():
    _safe_mkdir(LOG_DIR)
    removed = []
    for p in [
        ACCESS_LOG_PATH,
        REC_LOG_PATH,
        RES_LOG_PATH,
        ECO_LOG_PATH,
        ECO_CENARIOS_LOG_PATH,
        DQ_LOG_PATH,
    ]:
        if os.path.exists(p):
            os.remove(p)
            removed.append(os.path.basename(p))

    for fn in os.listdir(LOG_DIR):
        if fn.startswith("relatorio_avaliador_uc_") or fn.startswith("grafico_economia_") or fn.startswith("detalhe_uc_"):
            try:
                os.remove(os.path.join(LOG_DIR, fn))
                removed.append(fn)
            except Exception:
                pass

    for k in list(st.session_state.keys()):
        if str(k).startswith("log_rec::"):
            st.session_state.pop(k, None)

    return removed


def check_login():
    if st.session_state.get("authenticated", False):
        return True

    user_ok, pass_ok, evaluator = get_settings()

    st.title("Acesso restrito")
    st.caption("Informe usuário e senha para acessar o protótipo do TCC.")

    with st.form("login_form", clear_on_submit=False):
        user = st.text_input("Usuário")
        password = st.text_input("Senha", type="password")
        submit = st.form_submit_button("Entrar")

    if submit:
        ok_user = hmac.compare_digest(user.strip(), user_ok)
        ok_pass = hmac.compare_digest(password, pass_ok)

        if ok_user and ok_pass:
            st.session_state["authenticated"] = True
            st.session_state["login_user"] = user_ok
            st.session_state["evaluator_name"] = evaluator

            if not st.session_state.get("logged_access", False):
                log_access_event("LOGIN_SUCCESS", user_ok, evaluator)
                st.session_state["logged_access"] = True

            st.rerun()
        else:
            st.session_state["authenticated"] = False
            st.error("Usuário ou senha inválidos.")

    return False


if not check_login():
    st.stop()



# DADOS —

def ensure_datetime_sorted(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return df


def add_dt_hours(df: pd.DataFrame) -> pd.DataFrame:
    """
    dt_horas = diferença para o próximo ponto (em horas).
    Para o último ponto, usa mediana do dt (fallback 1h).
    """
    df = df.copy()
    dt = df["timestamp"].shift(-1) - df["timestamp"]
    dt_h = dt.dt.total_seconds() / 3600.0
    med = float(np.nanmedian(dt_h.to_numpy())) if np.isfinite(np.nanmedian(dt_h.to_numpy())) else 1.0
    df["dt_horas"] = dt_h.fillna(med).clip(lower=0.0)
    return df


def ajustar_energia_se_acumulada(df: pd.DataFrame, col: str = "energia_kwh") -> pd.DataFrame:
    """
    Se detectar padrão de energia acumulada, converte para energia por intervalo (delta).
    Regra simples: se a série é majoritariamente crescente e range é significativo.
    """
    df = df.copy()
    if col not in df.columns:
        return df

    s = pd.to_numeric(df[col], errors="coerce")
    if s.isna().all():
        return df

    dif = s.diff()
    pct_pos = float((dif.fillna(0) >= 0).mean())

    if pct_pos > 0.95 and float(s.max() - s.min()) > 0:
        df[col] = dif.clip(lower=0).fillna(0.0)
    return df


def compute_energy_and_volume(df: pd.DataFrame):
    """
    Retorna energia_total_kwh, volume_total_m3 usando:
    - energia_kwh: assume energia por amostra (já integrada no intervalo)
    - se não existir energia_kwh, mas existir potencia_kw: integra potencia_kw * dt_horas
    - volume: integra vazao_m3h * dt_horas
    """
    df = add_dt_hours(df)

    if "energia_kwh" in df.columns:
        energia_series = pd.to_numeric(df["energia_kwh"], errors="coerce").fillna(0.0)
        energia_total = float(energia_series.sum())
    elif "potencia_kw" in df.columns:
        energia_series = (
            pd.to_numeric(df["potencia_kw"], errors="coerce").fillna(0.0)
            * pd.to_numeric(df["dt_horas"], errors="coerce").fillna(0.0)
        )
        energia_total = float(energia_series.sum())
    else:
        energia_series = pd.Series([0.0] * len(df))
        energia_total = 0.0

    if "vazao_m3h" in df.columns:
        volume_series = (
            pd.to_numeric(df["vazao_m3h"], errors="coerce").fillna(0.0)
            * pd.to_numeric(df["dt_horas"], errors="coerce").fillna(0.0)
        )
        volume_total = float(volume_series.sum())
    else:
        volume_series = pd.Series([0.0] * len(df))
        volume_total = 0.0

    return energia_total, volume_total, energia_series, volume_series, df


def data_quality_metrics(df: pd.DataFrame) -> dict:
    """
    Métricas simples (auditáveis) de qualidade do dado — para escala.
    """
    df = df.copy()
    n = int(len(df))

    def pct_missing(col):
        if col not in df.columns:
            return 100.0
        return float(pd.to_numeric(df[col], errors="coerce").isna().mean() * 100.0)

    metrics = {
        "n_registros": n,
        "missing_timestamp_pct": float(df["timestamp"].isna().mean() * 100.0) if "timestamp" in df.columns else 100.0,
        "missing_lamina_pct": pct_missing("lamina_cm"),
        "missing_vazao_pct": pct_missing("vazao_m3h"),
        "missing_energia_pct": pct_missing("energia_kwh"),
        "missing_chuva_pct": pct_missing("chuva_mm"),
        "missing_bomba_pct": pct_missing("bomba_ligada"),
    }

    if n >= 3 and "timestamp" in df.columns:
        df2 = ensure_datetime_sorted(df)
        df2 = add_dt_hours(df2)
        metrics["dt_mediana_h"] = float(np.nanmedian(df2["dt_horas"].to_numpy()))
        metrics["dt_max_h"] = float(np.nanmax(df2["dt_horas"].to_numpy()))
        metrics["dt_min_h"] = float(np.nanmin(df2["dt_horas"].to_numpy()))
    else:
        metrics["dt_mediana_h"] = np.nan
        metrics["dt_max_h"] = np.nan
        metrics["dt_min_h"] = np.nan

    if "lamina_cm" in df.columns:
        lam = pd.to_numeric(df["lamina_cm"], errors="coerce")
        metrics["lamina_outlier_pct"] = float(((lam < 0) | (lam > 25)).mean() * 100.0)
    else:
        metrics["lamina_outlier_pct"] = 0.0

    if "energia_kwh" in df.columns:
        e = pd.to_numeric(df["energia_kwh"], errors="coerce")
        metrics["energia_outlier_pct"] = float((e < 0).mean() * 100.0)
    else:
        metrics["energia_outlier_pct"] = 0.0

    return metrics



# SIMULAÇÃO REALISTA (90 dias) + Multi-UC

def _sim_rain_events(rng: np.random.Generator, n_horas: int) -> np.ndarray:
    chuva = np.zeros(n_horas, dtype=float)
    n_eventos = int(rng.integers(10, 20))
    for _ in range(n_eventos):
        idx = int(rng.integers(0, n_horas))
        dur = int(rng.integers(2, 10))
        intensidade = float(rng.uniform(1.0, 7.0))  # mm/h
        chuva[idx: min(n_horas, idx + dur)] += intensidade
    chuva += rng.gamma(shape=0.4, scale=0.15, size=n_horas)
    return np.clip(chuva, 0, None)


def _sim_et0_proxy(rng: np.random.Generator, n_horas: int) -> np.ndarray:
    horas = np.arange(n_horas)
    ciclo = 0.10 + 0.08 * np.sin(2 * np.pi * (horas % 24 - 14) / 24)
    ruido = rng.normal(0, 0.01, n_horas)
    return np.clip(ciclo + ruido, 0.01, 0.25)  # mm/h


def _bomba_schedule_baseline(h: int, evitar_ponta: bool, ponta_inicio: int, ponta_fim: int) -> int:
    on = 0
    if (h >= 22 or h <= 6) or (8 <= h <= 11) or (14 <= h <= 16):
        on = 1
    if evitar_ponta and (ponta_inicio <= h < ponta_fim):
        on = 0
    return on


def gerar_dados_exemplo_multi_uc(
    ucs: list,
    n_dias: int = DEFAULT_N_DIAS,
    seed: int = 42,
    evitar_ponta: bool = True,
    ponta_inicio: int = DEFAULT_PONTA_INICIO,
    ponta_fim: int = DEFAULT_PONTA_FIM,
):
    rng = np.random.default_rng(seed)
    n_horas = int(n_dias * 24)
    now = datetime.now().replace(minute=0, second=0, microsecond=0)
    ts = pd.date_range(end=now, periods=n_horas, freq="h")

    frames = []
    for i, uc in enumerate(ucs):
        rng_uc = np.random.default_rng(seed + 1000 + i * 17)

        chuva = _sim_rain_events(rng_uc, n_horas)
        et0 = _sim_et0_proxy(rng_uc, n_horas)

        vazao_nom = float(rng_uc.uniform(55, 95))  # m³/h
        kw_nom = float(rng_uc.uniform(35, 75))     # kW
        eficiencia_ruido = float(rng_uc.uniform(0.85, 1.15))

        bomba = np.zeros(n_horas, dtype=int)
        for t in range(n_horas):
            hora = int(ts[t].hour)
            bomba[t] = _bomba_schedule_baseline(
                hora, evitar_ponta=evitar_ponta, ponta_inicio=ponta_inicio, ponta_fim=ponta_fim
            )
            if chuva[t] > 5.0 and rng_uc.random() < 0.6:
                bomba[t] = 0

        vazao = np.where(
            bomba == 1,
            rng_uc.normal(vazao_nom, vazao_nom * 0.12, n_horas),
            rng_uc.normal(2.0, 1.0, n_horas),
        )
        vazao = np.clip(vazao, 0, None)

        potencia_kw = np.where(
            bomba == 1,
            rng_uc.normal(kw_nom, kw_nom * 0.10, n_horas) * eficiencia_ruido,
            rng_uc.normal(0.8, 0.4, n_horas),
        )
        potencia_kw = np.clip(potencia_kw, 0, None)
        energia_kwh = potencia_kw * 1.0  # 1h

        lamina = np.zeros(n_horas, dtype=float)
        lamina[0] = float(rng_uc.uniform(6.5, 9.0))

        for t in range(1, n_horas):
            ganho_irrig_cm = (vazao[t] / 1200.0)          # proxy
            ganho_chuva_cm = (chuva[t] / 10.0) / 10.0     # mm -> cm + infiltração
            perda_et_cm = (et0[t] / 10.0)                 # mm -> cm
            infiltracao_cm = float(rng_uc.normal(0.015, 0.008))
            lamina[t] = lamina[t - 1] + ganho_irrig_cm + ganho_chuva_cm - perda_et_cm - infiltracao_cm

        lamina = np.clip(lamina, 0.0, 22.0)

        df_uc = pd.DataFrame(
            {
                "timestamp": ts,
                "uc_id": str(uc),
                "lamina_cm": lamina,
                "vazao_m3h": vazao,
                "energia_kwh": energia_kwh,
                "chuva_mm": chuva,
                "et0_mm": et0,
                "bomba_ligada": bomba,
                "potencia_kw": potencia_kw,
            }
        )

        # simula intervalo irregular removendo parte das linhas
        drop_pct = float(rng_uc.uniform(0.01, 0.04))
        n_drop = int(len(df_uc) * drop_pct)
        if n_drop > 0:
            drop_idx = rng_uc.choice(df_uc.index.to_numpy(), size=n_drop, replace=False)
            df_uc = df_uc.drop(drop_idx).reset_index(drop=True)

        frames.append(df_uc)

    df_all = pd.concat(frames, ignore_index=True)
    return ensure_datetime_sorted(df_all)


@st.cache_data(show_spinner=False)
def carregar_dados_cached(csv_path: str, file_sig: str, usar_simulacao: bool, seed: int, n_dias: int,
                          evitar_ponta: bool, ponta_inicio: int, ponta_fim: int):
    if (not usar_simulacao) and os.path.exists(csv_path):
        df = pd.read_csv(csv_path)

        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        else:
            for c in df.columns:
                if "data" in c.lower() or "hora" in c.lower() or "time" in c.lower():
                    df["timestamp"] = pd.to_datetime(df[c], errors="coerce")
                    break

        if "uc_id" not in df.columns:
            df["uc_id"] = "UC01"

        df = ensure_datetime_sorted(df)
        df = ajustar_energia_se_acumulada(df, "energia_kwh")
        return df

    ucs = DEFAULT_UCS
    return gerar_dados_exemplo_multi_uc(
        ucs=ucs,
        n_dias=int(n_dias),
        seed=int(seed),
        evitar_ponta=bool(evitar_ponta),
        ponta_inicio=int(ponta_inicio),
        ponta_fim=int(ponta_fim),
    )


def carregar_dados(usar_simulacao: bool, seed: int, n_dias: int, evitar_ponta: bool, ponta_inicio: int, ponta_fim: int):
    file_sig = "no_file"
    if os.path.exists(DATA_CSV_PATH):
        file_sig = str(os.path.getmtime(DATA_CSV_PATH))
    return carregar_dados_cached(DATA_CSV_PATH, file_sig, usar_simulacao, seed, n_dias, evitar_ponta, ponta_inicio, ponta_fim)



# TARIFA HORO-SAZONAL (PONTA/FP) + CUSTOS

def add_tarifa_horaria(df: pd.DataFrame, tarifa_fp: float, tarifa_ponta: float, ponta_inicio: int, ponta_fim: int) -> pd.DataFrame:
    df = df.copy()
    h = df["timestamp"].dt.hour
    is_ponta = (h >= int(ponta_inicio)) & (h < int(ponta_fim))
    df["is_ponta"] = is_ponta.astype(int)
    df["tarifa_rs_kwh"] = np.where(is_ponta, float(tarifa_ponta), float(tarifa_fp))
    return df


def calc_custo_energia(df: pd.DataFrame, usar_variavel: bool, tarifa_kwh_fixa: float, tarifa_fp: float, tarifa_ponta: float,
                       ponta_inicio: int, ponta_fim: int, energia_col: str):
    df = df.copy()
    if energia_col not in df.columns:
        return 0.0, np.nan, np.nan

    e = pd.to_numeric(df[energia_col], errors="coerce").fillna(0.0)

    if not usar_variavel:
        custo = float(e.sum()) * float(tarifa_kwh_fixa)
        return custo, np.nan, np.nan

    df2 = add_tarifa_horaria(df, tarifa_fp, tarifa_ponta, ponta_inicio, ponta_fim)
    t = pd.to_numeric(df2["tarifa_rs_kwh"], errors="coerce").fillna(float(tarifa_fp))
    custo_total = float((e * t).sum())

    is_ponta = df2["is_ponta"].astype(int).to_numpy()
    custo_ponta = float((e.to_numpy() * t.to_numpy() * (is_ponta == 1)).sum())
    custo_fp = float((e.to_numpy() * t.to_numpy() * (is_ponta == 0)).sum())
    return custo_total, custo_ponta, custo_fp



# KPIs (BI)

def kpis_basicos(df: pd.DataFrame):
    df = ensure_datetime_sorted(df)
    energia_total_kwh, volume_total_m3, _, _, df_dt = compute_energy_and_volume(df)

    lamina_media = (
        float(pd.to_numeric(df_dt.get("lamina_cm", pd.Series(dtype=float)), errors="coerce").mean())
        if "lamina_cm" in df_dt.columns else np.nan
    )
    horas_bomba = (
        int(pd.to_numeric(df_dt.get("bomba_ligada", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
        if "bomba_ligada" in df_dt.columns else 0
    )

    ult_24h = df_dt[df_dt["timestamp"] >= (df_dt["timestamp"].max() - pd.Timedelta(hours=24))]
    chuva_24h = (
        float(pd.to_numeric(ult_24h.get("chuva_mm", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum())
        if "chuva_mm" in ult_24h.columns else 0.0
    )

    eficiencia = (energia_total_kwh / volume_total_m3) if volume_total_m3 > 0 else None

    return {
        "lamina_media": lamina_media,
        "energia_total_kwh": energia_total_kwh,
        "volume_total_m3": volume_total_m3,
        "eficiencia_kwh_m3": eficiencia,
        "horas_bomba": horas_bomba,
        "chuva_24h": chuva_24h,
    }



# RECOMENDAÇÃO (IA EXPLICÁVEL por regras)

def recomendacao_ia(df, fase: str, manejo: str, tipo_solo: str, risco_frio: bool, dias_pos_floracao: int):
    df = ensure_datetime_sorted(df)
    agora = df["timestamp"].max()

    ult_6h = df[df["timestamp"] >= (agora - pd.Timedelta(hours=6))]
    ult_24h = df[df["timestamp"] >= (agora - pd.Timedelta(hours=24))]

    chuva_24h = float(pd.to_numeric(ult_24h.get("chuva_mm", 0.0), errors="coerce").fillna(0.0).sum()) if "chuva_mm" in df.columns else 0.0
    lamina_atual = float(pd.to_numeric(df.iloc[-1].get("lamina_cm", np.nan), errors="coerce")) if "lamina_cm" in df.columns else np.nan
    bomba_atual = int(pd.to_numeric(df.iloc[-1].get("bomba_ligada", 0), errors="coerce")) if "bomba_ligada" in df.columns else 0

    e6, v6, _, _, ult_6h_dt = compute_energy_and_volume(ult_6h) if len(ult_6h) else (0.0, 0.0, None, None, ult_6h)
    eficiencia_6h = (e6 / v6) if v6 > 0 else None
    bomba_horas_6h = int(pd.to_numeric(ult_6h_dt.get("bomba_ligada", 0), errors="coerce").fillna(0).sum()) if "bomba_ligada" in ult_6h_dt.columns else 0

    e24, v24, _, _, ult_24h_dt = compute_energy_and_volume(ult_24h) if len(ult_24h) else (0.0, 0.0, None, None, ult_24h)
    eficiencia_24h = (e24 / v24) if v24 > 0 else None
    bomba_horas_24h = int(pd.to_numeric(ult_24h_dt.get("bomba_ligada", 0), errors="coerce").fillna(0).sum()) if "bomba_ligada" in ult_24h_dt.columns else 0

    lamina_min_24h = float(pd.to_numeric(ult_24h.get("lamina_cm", np.nan), errors="coerce").min()) if len(ult_24h) and "lamina_cm" in ult_24h.columns else np.nan
    lamina_max_24h = float(pd.to_numeric(ult_24h.get("lamina_cm", np.nan), errors="coerce").max()) if len(ult_24h) and "lamina_cm" in ult_24h.columns else np.nan

    base_ef = None
    if "bomba_ligada" in df.columns:
        df_ligada = df[pd.to_numeric(df["bomba_ligada"], errors="coerce").fillna(0).astype(int) == 1]
        if len(df_ligada) > 10:
            eb, vb, *_ = compute_energy_and_volume(df_ligada)
            if vb > 0:
                base_ef = float(eb / vb)

    mensagens = []
    nivel = "info"

    if chuva_24h >= 12:
        mensagens.append(
            f"Chuva 24h = {chuva_24h:.1f} mm (crédito hídrico relevante). "
            "Recomendação: avaliar reduzir/adiar bombeamento nas próximas 12–24h."
        )
        nivel = "warning"

    if chuva_24h >= 24:
        mensagens.append(
            f"Chuva 24h alta ({chuva_24h:.1f} mm). "
            "Recomendação: suspender temporariamente o bombeamento e monitorar lâmina para evitar excesso."
        )
        nivel = "warning"

    if risco_frio and fase == "Emborrachamento/Floração":
        alvo_min, alvo_max = 15.0, 20.0
        mensagens.append("Risco de frio marcado: pode-se trabalhar com lâmina maior (~15–20 cm) por janela curta.")
        nivel = "warning"
    else:
        if fase == "Vegetativa":
            alvo_min, alvo_max = 2.5, 7.5
        elif fase == "Reprodutiva":
            alvo_min, alvo_max = 5.0, 10.0
        elif fase == "Emborrachamento/Floração":
            alvo_min, alvo_max = 7.5, 10.0
        else:
            alvo_min, alvo_max = 2.5, 7.5

    if manejo == "Contínuo" and not (risco_frio and fase == "Emborrachamento/Floração"):
        alvo_min, alvo_max = 6.5, 8.5
        mensagens.append("Manejo contínuo: referência operacional ~7,5 cm (faixa 6,5–8,5 cm).")

    if np.isfinite(lamina_atual):
        if lamina_atual > 10.0 and not (risco_frio and fase == "Emborrachamento/Floração"):
            mensagens.append(
                f"Alerta: lâmina alta ({lamina_atual:.1f} cm). Valores >10 cm podem aumentar perdas e risco de acamamento."
            )
            nivel = "warning"

        if lamina_atual < 2.5 and fase != "Maturação":
            mensagens.append(
                f"Atenção: lâmina muito baixa ({lamina_atual:.1f} cm). "
                "Abaixo de ~2,5 cm exige controle operacional mais rigoroso."
            )
            nivel = "warning" if nivel != "error" else nivel

        if lamina_atual < alvo_min:
            mensagens.append(
                f"Ação: lâmina abaixo do alvo da fase {fase} ({lamina_atual:.1f} < {alvo_min:.1f} cm). Priorizar reposição."
            )
            nivel = "error"
        elif lamina_atual > alvo_max:
            mensagens.append(
                f"Observação: lâmina acima do alvo da fase {fase} ({lamina_atual:.1f} > {alvo_max:.1f} cm). "
                "Reduzir bombeamento e acompanhar."
            )
            nivel = "warning" if nivel != "error" else nivel
        else:
            mensagens.append(f"OK: lâmina dentro do alvo da fase {fase} ({alvo_min:.1f}–{alvo_max:.1f} cm).")
            if nivel == "info":
                nivel = "success"
    else:
        mensagens.append("Atenção: lâmina atual indisponível (dado ausente).")
        nivel = "warning"

    if manejo.startswith("Intermitente") and fase != "Maturação" and np.isfinite(lamina_atual):
        on_threshold = max(2.5, alvo_min)
        off_threshold = min(8.0, max(alvo_max, 7.0))

        if lamina_atual <= on_threshold:
            mensagens.append(f"Manejo intermitente: sugerir RETOMAR irrigação (lâmina {lamina_atual:.1f} ≤ {on_threshold:.1f} cm).")
            nivel = "error"
        elif lamina_atual >= off_threshold:
            mensagens.append(f"Manejo intermitente: sugerir PAUSAR bombeamento (lâmina {lamina_atual:.1f} ≥ {off_threshold:.1f} cm).")
            nivel = "warning" if nivel != "error" else nivel
        else:
            mensagens.append(f"Manejo intermitente: faixa operacional {on_threshold:.1f}–{off_threshold:.1f} cm. Manter.")

    if fase == "Maturação":
        if dias_pos_floracao >= 10 and tipo_solo == "Argiloso":
            mensagens.append(
                f"Maturação (solo argiloso): {dias_pos_floracao} dias pós-floração. "
                "Pode-se considerar iniciar supressão da irrigação (monitorar estágio do grão)."
            )
        elif dias_pos_floracao >= 10 and tipo_solo != "Argiloso":
            mensagens.append(
                f"Maturação (solo arenoso/bem drenado): {dias_pos_floracao} dias pós-floração. "
                "Recomendação: cautela e possível postergação da supressão devido à maior drenagem."
            )
        else:
            mensagens.append("Maturação: para supressão, use floração plena e estágio do grão como referência.")

    if eficiencia_6h is not None and base_ef is not None:
        if eficiencia_6h > base_ef * 1.15:
            mensagens.append(
                f"Alerta eficiência energética: 6h = {eficiencia_6h:.3f} kWh/m³ vs baseline = {base_ef:.3f} kWh/m³. "
                "Inspecionar condições hidráulicas e operação da bomba."
            )
            nivel = "warning" if nivel != "error" else nivel

    meta = {
        "bomba_atual": bomba_atual,
        "lamina_atual": lamina_atual,
        "chuva_24h": chuva_24h,
        "bomba_horas_6h": bomba_horas_6h,
        "energia_6h": float(e6),
        "volume_6h": float(v6),
        "eficiencia_6h": eficiencia_6h,
        "bomba_horas_24h": bomba_horas_24h,
        "energia_24h": float(e24),
        "volume_24h": float(v24),
        "eficiencia_24h": eficiencia_24h,
        "lamina_min_24h": lamina_min_24h,
        "lamina_max_24h": lamina_max_24h,
        "baseline_ef": base_ef,
    }

    return nivel, mensagens, meta



# RESULTADOS — baseline vs otimizado + tarifa fixa/variável

def aplicar_otimizacao_regras(
    df,
    lamina_max=9.5,
    chuva_min_mm=10.0,
    evitar_ponta: bool = False,
    ponta_inicio: int = DEFAULT_PONTA_INICIO,
    ponta_fim: int = DEFAULT_PONTA_FIM,
):
    df = ensure_datetime_sorted(df)

    if "chuva_mm" in df.columns:
        s = df.set_index("timestamp")["chuva_mm"].rolling("24h").sum()
        df["chuva_24h"] = s.to_numpy()
    else:
        df["chuva_24h"] = 0.0

    cond_desliga = (pd.to_numeric(df["chuva_24h"], errors="coerce").fillna(0.0) >= float(chuva_min_mm))
    if "lamina_cm" in df.columns:
        cond_desliga = cond_desliga | (pd.to_numeric(df["lamina_cm"], errors="coerce").fillna(0.0) >= float(lamina_max))

    bomba_base = pd.to_numeric(df.get("bomba_ligada", 0), errors="coerce").fillna(0).astype(int).to_numpy()
    bomba_otim = np.where(cond_desliga.to_numpy(), 0, bomba_base)

    if evitar_ponta:
        h = df["timestamp"].dt.hour.to_numpy()
        is_ponta = (h >= int(ponta_inicio)) & (h < int(ponta_fim))
        bomba_otim = np.where(is_ponta, 0, bomba_otim)

    df["bomba_otim"] = bomba_otim

    if "energia_kwh" in df.columns:
        e = pd.to_numeric(df["energia_kwh"], errors="coerce").fillna(0.0).to_numpy()
        df["energia_otim_kwh"] = np.where(bomba_otim == 1, e, 0.0)
    elif "potencia_kw" in df.columns:
        p = pd.to_numeric(df["potencia_kw"], errors="coerce").fillna(0.0).to_numpy()
        df["potencia_otim_kw"] = np.where(bomba_otim == 1, p, 0.0)
    else:
        df["energia_otim_kwh"] = 0.0

    if "vazao_m3h" in df.columns:
        v = pd.to_numeric(df["vazao_m3h"], errors="coerce").fillna(0.0).to_numpy()
        df["vazao_otim_m3h"] = np.where(bomba_otim == 1, v, 0.0)
    else:
        df["vazao_otim_m3h"] = 0.0

    return df


def comparar_cenarios(
    df,
    tarifa_kwh_fixa: float = DEFAULT_TARIFA_FIXA,
    usar_tarifa_variavel: bool = False,
    tarifa_fp: float = DEFAULT_TARIFA_FP,
    tarifa_ponta: float = DEFAULT_TARIFA_PONTA,
    ponta_inicio: int = DEFAULT_PONTA_INICIO,
    ponta_fim: int = DEFAULT_PONTA_FIM,
):
    df = ensure_datetime_sorted(df)

    energia_base, volume_base, _, _, df_dt = compute_energy_and_volume(df)
    horas_bomba_base = int(pd.to_numeric(df_dt.get("bomba_ligada", 0), errors="coerce").fillna(0).sum()) if "bomba_ligada" in df_dt.columns else 0
    ef_base = (energia_base / volume_base) if volume_base > 0 else np.nan

    custo_base, custo_base_ponta, custo_base_fp = calc_custo_energia(
        df_dt,
        usar_variavel=usar_tarifa_variavel,
        tarifa_kwh_fixa=tarifa_kwh_fixa,
        tarifa_fp=tarifa_fp,
        tarifa_ponta=tarifa_ponta,
        ponta_inicio=ponta_inicio,
        ponta_fim=ponta_fim,
        energia_col="energia_kwh" if "energia_kwh" in df_dt.columns else "energia_kwh",
    )

    df_ot = add_dt_hours(df_dt.copy())

    if "energia_otim_kwh" in df_ot.columns:
        energia_otim = float(pd.to_numeric(df_ot["energia_otim_kwh"], errors="coerce").fillna(0.0).sum())
        energia_otim_col = "energia_otim_kwh"
    elif "potencia_otim_kw" in df_ot.columns:
        energia_otim = float(
            (pd.to_numeric(df_ot["potencia_otim_kw"], errors="coerce").fillna(0.0)
             * pd.to_numeric(df_ot["dt_horas"], errors="coerce").fillna(0.0)).sum()
        )
        df_ot["energia_otim_kwh"] = (
            pd.to_numeric(df_ot["potencia_otim_kw"], errors="coerce").fillna(0.0)
            * pd.to_numeric(df_ot["dt_horas"], errors="coerce").fillna(0.0)
        )
        energia_otim_col = "energia_otim_kwh"
    else:
        energia_otim = 0.0
        df_ot["energia_otim_kwh"] = 0.0
        energia_otim_col = "energia_otim_kwh"

    if "vazao_otim_m3h" in df_ot.columns:
        volume_otim = float(
            (pd.to_numeric(df_ot["vazao_otim_m3h"], errors="coerce").fillna(0.0)
             * pd.to_numeric(df_ot["dt_horas"], errors="coerce").fillna(0.0)).sum()
        )
    else:
        volume_otim = 0.0

    horas_bomba_otim = int(pd.to_numeric(df_ot.get("bomba_otim", 0), errors="coerce").fillna(0).sum()) if "bomba_otim" in df_ot.columns else 0
    ef_otim = (energia_otim / volume_otim) if volume_otim > 0 else np.nan

    custo_otim, custo_otim_ponta, custo_otim_fp = calc_custo_energia(
        df_ot,
        usar_variavel=usar_tarifa_variavel,
        tarifa_kwh_fixa=tarifa_kwh_fixa,
        tarifa_fp=tarifa_fp,
        tarifa_ponta=tarifa_ponta,
        ponta_inicio=ponta_inicio,
        ponta_fim=ponta_fim,
        energia_col=energia_otim_col,
    )

    economia_kwh = energia_base - energia_otim
    economia_rs = custo_base - custo_otim
    reducao_volume = volume_base - volume_otim

    return {
        "energia_base": energia_base,
        "energia_otim": energia_otim,
        "economia_kwh": economia_kwh,
        "custo_base": custo_base,
        "custo_otim": custo_otim,
        "economia_rs": economia_rs,
        "volume_base": volume_base,
        "volume_otim": volume_otim,
        "reducao_volume": reducao_volume,
        "horas_bomba_base": horas_bomba_base,
        "horas_bomba_otim": horas_bomba_otim,
        "ef_base": ef_base,
        "ef_otim": ef_otim,
        "custo_base_ponta": custo_base_ponta,
        "custo_base_fp": custo_base_fp,
        "custo_otim_ponta": custo_otim_ponta,
        "custo_otim_fp": custo_otim_fp,
    }


def salvar_resultados_piloto(
    res: dict,
    periodo_label: str,
    lamina_max: float,
    chuva_min_mm: float,
    usar_tarifa_variavel: bool,
    tarifa_kwh_fixa: float,
    tarifa_fp: float,
    tarifa_ponta: float,
    ponta_inicio: int,
    ponta_fim: int,
    evitar_ponta: bool,
    username: str,
    evaluator: str,
    uc_id: str,
):
    _safe_mkdir(LOG_DIR)
    ts = _now_iso()

    row = {
        "timestamp_execucao": ts,
        "uc_id": str(uc_id),
        "periodo": periodo_label,
        "lamina_max_cm": float(lamina_max),
        "chuva_min_24h_mm": float(chuva_min_mm),
        "usar_tarifa_variavel": int(bool(usar_tarifa_variavel)),
        "tarifa_fixa_rs_kwh": float(tarifa_kwh_fixa),
        "tarifa_fp": float(tarifa_fp),
        "tarifa_ponta": float(tarifa_ponta),
        "ponta_inicio": int(ponta_inicio),
        "ponta_fim": int(ponta_fim),
        "evitar_ponta": int(bool(evitar_ponta)),
        "user": username,
        "evaluator": evaluator,
        "app_version": APP_VERSION,
        "run_id": RUN_ID,
        "energia_base_kwh": float(res.get("energia_base", 0.0)),
        "energia_otim_kwh": float(res.get("energia_otim", 0.0)),
        "economia_kwh": float(res.get("economia_kwh", 0.0)),
        "custo_base_rs": float(res.get("custo_base", 0.0)),
        "custo_otim_rs": float(res.get("custo_otim", 0.0)),
        "economia_rs": float(res.get("economia_rs", 0.0)),
        "volume_base_m3": float(res.get("volume_base", 0.0)),
        "volume_otim_m3": float(res.get("volume_otim", 0.0)),
        "reducao_volume_m3": float(res.get("reducao_volume", 0.0)),
        "horas_bomba_base_h": int(res.get("horas_bomba_base", 0)),
        "horas_bomba_otim_h": int(res.get("horas_bomba_otim", 0)),
        "ef_base_kwh_m3": float(res.get("ef_base", np.nan)) if pd.notna(res.get("ef_base", np.nan)) else np.nan,
        "ef_otim_kwh_m3": float(res.get("ef_otim", np.nan)) if pd.notna(res.get("ef_otim", np.nan)) else np.nan,
        "custo_base_ponta_rs": res.get("custo_base_ponta", np.nan),
        "custo_base_fp_rs": res.get("custo_base_fp", np.nan),
        "custo_otim_ponta_rs": res.get("custo_otim_ponta", np.nan),
        "custo_otim_fp_rs": res.get("custo_otim_fp", np.nan),
    }

    df_row = pd.DataFrame([row])
    if not os.path.exists(RES_LOG_PATH):
        df_row.to_csv(RES_LOG_PATH, index=False, encoding="utf-8")
    else:
        df_row.to_csv(RES_LOG_PATH, mode="a", header=False, index=False, encoding="utf-8")

    return RES_LOG_PATH



# MODO AVALIADOR (1 clique) — relatório por UC + PNGs

def relatorio_avaliador_por_uc(
    df_all: pd.DataFrame,
    periodo_label: str,
    lamina_max: float,
    chuva_min_mm: float,
    usar_tarifa_variavel: bool,
    tarifa_kwh_fixa: float,
    tarifa_fp: float,
    tarifa_ponta: float,
    ponta_inicio: int,
    ponta_fim: int,
    evitar_ponta: bool,
    salvar_detalhe_por_uc: bool = False,
):
    _safe_mkdir(LOG_DIR)

    rows = []
    ucs = sorted(df_all["uc_id"].astype(str).unique().tolist())

    for uc in ucs:
        df_uc = df_all[df_all["uc_id"].astype(str) == str(uc)].copy()
        df_uc = ensure_datetime_sorted(df_uc)
        df_uc = ajustar_energia_se_acumulada(df_uc, "energia_kwh")

        max_data = df_uc["timestamp"].max()
        if periodo_label == "Últimas 24h":
            df_f = df_uc[df_uc["timestamp"] >= (max_data - pd.Timedelta(hours=24))]
        elif periodo_label == "Últimos 3 dias":
            df_f = df_uc[df_uc["timestamp"] >= (max_data - pd.Timedelta(days=3))]
        elif periodo_label == "Últimos 7 dias":
            df_f = df_uc[df_uc["timestamp"] >= (max_data - pd.Timedelta(days=7))]
        elif periodo_label == "Últimos 30 dias":
            df_f = df_uc[df_uc["timestamp"] >= (max_data - pd.Timedelta(days=30))]
        else:
            df_f = df_uc.copy()

        if len(df_f) < 10:
            rows.append({
                "uc_id": str(uc),
                "periodo": periodo_label,
                "status": "DADOS_INSUFICIENTES",
                "energia_base_kwh": 0.0,
                "energia_otim_kwh": 0.0,
                "economia_kwh": 0.0,
                "custo_base_rs": 0.0,
                "custo_otim_rs": 0.0,
                "economia_rs": 0.0,
            })
            continue

        df_sim = aplicar_otimizacao_regras(
            df_f,
            lamina_max=lamina_max,
            chuva_min_mm=chuva_min_mm,
            evitar_ponta=evitar_ponta,
            ponta_inicio=int(ponta_inicio),
            ponta_fim=int(ponta_fim),
        )

        res = comparar_cenarios(
            df_sim,
            tarifa_kwh_fixa=tarifa_kwh_fixa,
            usar_tarifa_variavel=usar_tarifa_variavel,
            tarifa_fp=tarifa_fp,
            tarifa_ponta=tarifa_ponta,
            ponta_inicio=int(ponta_inicio),
            ponta_fim=int(ponta_fim),
        )

        rows.append({
            "uc_id": str(uc),
            "periodo": periodo_label,
            "status": "OK",
            "lamina_max_cm": float(lamina_max),
            "chuva_min_24h_mm": float(chuva_min_mm),
            "usar_tarifa_variavel": int(bool(usar_tarifa_variavel)),
            "tarifa_fixa_rs_kwh": float(tarifa_kwh_fixa),
            "tarifa_fp": float(tarifa_fp),
            "tarifa_ponta": float(tarifa_ponta),
            "ponta_inicio": int(ponta_inicio),
            "ponta_fim": int(ponta_fim),
            "evitar_ponta": int(bool(evitar_ponta)),
            "energia_base_kwh": float(res["energia_base"]),
            "energia_otim_kwh": float(res["energia_otim"]),
            "economia_kwh": float(res["economia_kwh"]),
            "custo_base_rs": float(res["custo_base"]),
            "custo_otim_rs": float(res["custo_otim"]),
            "economia_rs": float(res["economia_rs"]),
            "custo_base_ponta_rs": res.get("custo_base_ponta", np.nan),
            "custo_base_fp_rs": res.get("custo_base_fp", np.nan),
            "custo_otim_ponta_rs": res.get("custo_otim_ponta", np.nan),
            "custo_otim_fp_rs": res.get("custo_otim_fp", np.nan),
        })

        if salvar_detalhe_por_uc:
            p_det = os.path.join(LOG_DIR, f"detalhe_uc_{str(uc)}.csv")
            df_sim.to_csv(p_det, index=False, encoding="utf-8")

    df_rep = pd.DataFrame(rows)
    ts = _now_iso().replace(":", "-")
    csv_path = f"{EVAL_UC_REPORT_PREFIX}_{ts}.csv"
    df_rep.to_csv(csv_path, index=False, encoding="utf-8")

    df_ok = df_rep[df_rep["status"] == "OK"].copy()
    png_rs_path = f"{EVAL_GRAF_RS_PREFIX}_{ts}.png"
    png_kwh_path = f"{EVAL_GRAF_KWH_PREFIX}_{ts}.png"

    if len(df_ok) > 0:
        fig1, ax1 = plt.subplots(figsize=(10, 4.8))
        x = np.arange(len(df_ok))
        ax1.bar(x, df_ok["economia_rs"])
        ax1.set_xticks(x)
        ax1.set_xticklabels(df_ok["uc_id"], rotation=0)
        ax1.set_ylabel("R$")
        ax1.set_title(f"Economia estimada (R$) por UC — {periodo_label}")
        fig1.tight_layout()
        fig1.savefig(png_rs_path, dpi=300, bbox_inches="tight")
        plt.close(fig1)

        fig2, ax2 = plt.subplots(figsize=(10, 4.8))
        x = np.arange(len(df_ok))
        ax2.bar(x, df_ok["economia_kwh"])
        ax2.set_xticks(x)
        ax2.set_xticklabels(df_ok["uc_id"], rotation=0)
        ax2.set_ylabel("kWh")
        ax2.set_title(f"Economia estimada (kWh) por UC — {periodo_label}")
        fig2.tight_layout()
        fig2.savefig(png_kwh_path, dpi=300, bbox_inches="tight")
        plt.close(fig2)
    else:
        open(png_rs_path, "wb").close()
        open(png_kwh_path, "wb").close()

    return csv_path, png_rs_path, png_kwh_path



# MODELO ECONÔMICO + UNIT ECONOMICS + BREAK-EVEN

def calc_precificacao_por_valor(economia_rs_mensal: float, pct_captura: float = 0.15, piso: float = 1200.0, teto: float = 3800.0):
    """
    P = min(max(E_mensal * pct_captura, piso), teto)
    """
    if economia_rs_mensal is None or economia_rs_mensal <= 0:
        return 0.0
    preco = economia_rs_mensal * float(pct_captura)
    return float(min(max(preco, piso), teto))


def calc_roi_payback_cliente(preco_mensal: float, economia_rs_mensal: float, investimento_inicial: float):
    """
    ROI (cliente) sobre a mensalidade:
    ROI = (economia - preço) / preço
    Payback (cliente) = investimento_inicial / (economia - preço)
    """
    if economia_rs_mensal <= 0 or preco_mensal <= 0:
        return {"roi_mensal": None, "payback_meses": None, "ganho_liquido": None}

    ganho_liquido = economia_rs_mensal - preco_mensal
    roi_mensal = ganho_liquido / preco_mensal

    payback = None
    if investimento_inicial is not None and investimento_inicial > 0 and ganho_liquido > 0:
        payback = investimento_inicial / ganho_liquido

    return {
        "roi_mensal": float(roi_mensal),
        "payback_meses": float(payback) if payback is not None else None,
        "ganho_liquido": float(ganho_liquido),
    }


def npv_anuidades(cf_anual: float, taxa: float, anos: int) -> float:
    if anos <= 0:
        return 0.0
    return sum(float(cf_anual) / ((1.0 + float(taxa)) ** t) for t in range(1, anos + 1))


def calcular_cenarios_economia(custo_base_periodo: float, alpha: float):
    cenarios = [
        ("Conservador (5%)", 0.05),
        ("Base (10%)", 0.10),
        ("Agressivo (15%)", 0.15),
    ]

    rows = []
    for nome, r in cenarios:
        econ_bruta = float(custo_base_periodo) * float(r)
        receita = econ_bruta * float(alpha)
        econ_liq = econ_bruta - receita
        rows.append({
            "cenário": nome,
            "r": float(r),
            "economia_bruta_rs": econ_bruta,
            "receita_servico_rs": receita,
            "economia_liquida_produtor_rs": econ_liq,
        })

    return pd.DataFrame(rows)


def salvar_cenarios_vpl_fenologia(df_row: pd.DataFrame):
    _safe_mkdir(LOG_DIR)
    if not os.path.exists(ECO_CENARIOS_LOG_PATH):
        df_row.to_csv(ECO_CENARIOS_LOG_PATH, index=False, encoding="utf-8")
    else:
        df_row.to_csv(ECO_CENARIOS_LOG_PATH, mode="a", header=False, index=False, encoding="utf-8")
    return ECO_CENARIOS_LOG_PATH


def unit_economics_break_even(
    preco_mensal: float,
    custo_variavel_por_cliente: float,
    overhead_pct: float,
    opex_mensal: float,
    n_clientes: int,
    capex_inicial: float,

):
    """
    Unit economics (SaaS):
    - Overhead proporcional ao preço (impostos/adm)
    - MC = P - CV - (P*overhead_pct)
    - Break-even (clientes) = OPEX / MC
    - Payback CAPEX (meses) = CAPEX / (MC * N)
    """
    P = float(preco_mensal)
    CV = float(custo_variavel_por_cliente)
    O = float(overhead_pct)
    OPEX = float(opex_mensal)
    N = int(max(0, n_clientes))
    CAPEX = float(capex_inicial)

    overhead_rs = P * O
    mc = P - CV - overhead_rs

    be = None
    if mc > 0:
        be = OPEX / mc

    payback_capex = None
    if mc > 0 and N > 0 and CAPEX > 0:
        payback_capex = CAPEX / (mc * N)

    # lucro/mês do projeto com N clientes
    lucro_mensal = (mc * N) - OPEX

    return {
        "preco_mensal": P,
        "cv_por_cliente": CV,
        "overhead_pct": O,
        "overhead_rs": overhead_rs,
        "margem_contribuicao": mc,
        "opex_mensal": OPEX,
        "break_even_clientes": be,
        "n_clientes": N,
        "lucro_mensal_projeto": lucro_mensal,
        "capex_inicial": CAPEX,
        "payback_capex_meses": payback_capex,
    }


# 
# UI HELPERS
# 
def bloco_contexto_tcc():
    st.markdown(
        f"""
**Contextualização (TCC):** Este dashboard demonstra a aplicação de *Business Intelligence* e *Data Science*
no monitoramento da irrigação do arroz irrigado, com foco em eficiência hídrica e energética.
Dados (sensores/SCADA/clima) são consolidados e transformados em KPIs, alertas e recomendações automatizadas,
apoiando o manejo e a tomada de decisão.

**Versão do protótipo:** {APP_VERSION} | **Execução:** {RUN_ID}
        """.strip()
    )


def fmt_br_money(x: float) -> str:
    try:
        return f"{float(x):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except Exception:
        return "—"


def fmt_br_number(x: float, ndigits: int = 1) -> str:
    try:
        s = f"{float(x):,.{ndigits}f}"
        return s.replace(",", "X").replace(".", ",").replace("X", ".")
    except Exception:
        return "—"


def fig_to_png_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def plot_cenarios_bar(df_cen: pd.DataFrame, title: str):
    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = np.arange(len(df_cen))
    ax.bar(x - 0.25, df_cen["economia_bruta_rs"], width=0.25, label="Economia bruta (R$)")
    ax.bar(x, df_cen["receita_servico_rs"], width=0.25, label="Receita serviço (R$)")
    ax.bar(x + 0.25, df_cen["economia_liquida_produtor_rs"], width=0.25, label="Economia líquida produtor (R$)")
    ax.set_xticks(x)
    ax.set_xticklabels(df_cen["cenário"], rotation=15, ha="right")
    ax.set_ylabel("R$ no período de referência")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_vpl_bar(df_vpl: pd.DataFrame, title: str):
    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = np.arange(len(df_vpl))
    ax.bar(x - 0.2, df_vpl["vpl_projeto_rs"], width=0.4, label="VPL Projeto (receita)")
    ax.bar(x + 0.2, df_vpl["vpl_produtor_rs"], width=0.4, label="VPL Produtor (economia líquida)")
    ax.set_xticks(x)
    ax.set_xticklabels(df_vpl["cenário"], rotation=15, ha="right")
    ax.set_ylabel("R$ (valor presente)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_fenologia_bar(df_fen: pd.DataFrame, title: str):
    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = np.arange(len(df_fen))
    ax.bar(x, df_fen["economia_bruta_rs"])
    ax.set_xticks(x)
    ax.set_xticklabels(df_fen["fase"], rotation=15, ha="right")
    ax.set_ylabel("R$ (economia bruta estimada)")
    ax.set_title(title)
    fig.tight_layout()
    return fig



# UI — TOPO + SIDEBAR (Sair)

evaluator_name = st.session_state.get("evaluator_name", "Avaliador")
login_user = st.session_state.get("login_user", "usuario")

st.title(APP_TITLE)
st.caption(APP_SUBTITLE)
st.markdown(f"**Avaliador:** {evaluator_name}")
bloco_contexto_tcc()

with st.sidebar:
    st.header("Sessão")
    st.caption(f"Usuário autenticado: **{login_user}**")
    st.caption(f"Avaliador: **{evaluator_name}**")
    st.caption(f"Versão: **{APP_VERSION}** | Execução: **{RUN_ID}**")

    user_ok, pass_ok, _ = get_settings()
    if user_ok == "admin" and pass_ok == "admin":
        st.warning("⚠️ Protótipo com credencial padrão (admin/admin). Em produção, use st.secrets/OAuth/SSO.")

    if st.button("Sair"):
        log_access_event("LOGOUT", login_user, evaluator_name)
        st.session_state["authenticated"] = False
        st.session_state["logged_access"] = False
        st.session_state.pop("login_user", None)
        st.session_state.pop("evaluator_name", None)
        st.rerun()




# APP — dados + filtros + interface em 6 abas



# Sidebar limpa: somente filtros principais visíveis
st.sidebar.header("Filtros principais")

usar_simulacao = st.sidebar.toggle("Usar simulação realista (90 dias / 6 UCs)", value=True)
seed = st.sidebar.number_input("Seed simulação", min_value=1, value=42, step=1)
n_dias = st.sidebar.number_input("Dias simulados", min_value=30, max_value=180, value=DEFAULT_N_DIAS, step=5)

with st.sidebar.expander("⚙️ Tarifa de energia", expanded=False):
    usar_tarifa_variavel = st.toggle("Usar tarifa horo-sazonal (ponta/fora-ponta)", value=True)
    tarifa_kwh_fixa = st.number_input("Tarifa fixa (R$/kWh)", min_value=0.0, value=float(DEFAULT_TARIFA_FIXA), step=0.01)
    tarifa_fp = st.number_input("Tarifa fora-ponta (R$/kWh)", min_value=0.0, value=float(DEFAULT_TARIFA_FP), step=0.01)
    tarifa_ponta = st.number_input("Tarifa ponta (R$/kWh)", min_value=0.0, value=float(DEFAULT_TARIFA_PONTA), step=0.01)
    ponta_inicio = st.number_input("Início ponta (hora)", min_value=0, max_value=23, value=int(DEFAULT_PONTA_INICIO), step=1)
    ponta_fim = st.number_input("Fim ponta (hora)", min_value=1, max_value=24, value=int(DEFAULT_PONTA_FIM), step=1)
    evitar_ponta_sim = st.toggle("Simulação baseline evita ponta", value=True)

# Carrega dados após parâmetros básicos e tarifa
df_all = carregar_dados(
    usar_simulacao=bool(usar_simulacao),
    seed=int(seed),
    n_dias=int(n_dias),
    evitar_ponta=bool(evitar_ponta_sim),
    ponta_inicio=int(ponta_inicio),
    ponta_fim=int(ponta_fim),
)

uc_id = st.sidebar.selectbox("UC monitorada", sorted(df_all["uc_id"].astype(str).unique().tolist()))
periodo = st.sidebar.selectbox(
    "Período",
    options=["Últimas 24h", "Últimos 3 dias", "Últimos 7 dias", "Últimos 30 dias", "Tudo"],
    index=3,
)

with st.sidebar.expander(" Parâmetros agronômicos", expanded=False):
    fase = st.selectbox(
        "Fase do cultivo",
        ["Vegetativa", "Reprodutiva", "Emborrachamento/Floração", "Maturação"],
        index=0,
    )
    manejo = st.selectbox(
        "Manejo de irrigação",
        ["Contínuo", "Intermitente (fornecimento intermitente)"],
        index=0,
    )
    tipo_solo = st.selectbox("Tipo de solo", ["Argiloso", "Arenoso/bem drenado"], index=0)
    risco_frio = st.checkbox("Risco de frio (<16°C) no emborrachamento?", value=False)
    dias_pos_floracao = st.number_input("Dias após floração plena", min_value=0, value=0, step=1)

with st.sidebar.expander("Otimização", expanded=False):
    lamina_max = st.slider("Lâmina máxima (cm)", min_value=7.0, max_value=20.0, value=10.0, step=0.5)
    chuva_min_mm = st.slider("Chuva 24h (mm) para reduzir/adiar", min_value=0.0, max_value=60.0, value=12.0, step=1.0)
    evitar_ponta_otim = st.toggle("Otimização: evitar operar na ponta", value=True)

with st.sidebar.expander("Modelo econômico", expanded=False):
    pct_captura = st.slider("Captura de valor (% da economia)", 5, 30, int(DEFAULT_PCT_CAPTURA * 100), 1) / 100.0
    alpha_receita = st.slider("Receita do projeto (% da economia bruta)", 5, 30, int(DEFAULT_ALPHA_RECEITA * 100), 1) / 100.0
    investimento_inicial_cliente = st.number_input("Investimento inicial do cliente (R$)", min_value=0.0, value=12400.0, step=100.0)
    duracao_safra_dias = st.slider("Duração da lâmina contínua (dias)", 80, 120, 90, 1)
    piso_plano = st.number_input("Piso do plano (R$/mês)", min_value=0.0, value=1200.0, step=100.0)
    teto_plano = st.number_input("Teto do plano (R$/mês)", min_value=0.0, value=3800.0, step=100.0)
    taxa_desconto = st.number_input("Taxa de desconto (a.a.)", min_value=0.0, value=float(DEFAULT_TAXA_DESCONTO), step=0.01)
    horizonte_anos = st.number_input("Horizonte (anos)", min_value=1, value=int(DEFAULT_HORIZONTE_ANOS), step=1)

    st.caption("Redução variável por fase fenológica")
    r_veg = st.number_input("Vegetativa r (%)", min_value=0.0, max_value=30.0, value=6.0, step=0.5) / 100.0
    r_rep = st.number_input("Reprodutiva r (%)", min_value=0.0, max_value=30.0, value=10.0, step=0.5) / 100.0
    r_flo = st.number_input("Emborrachamento/Floração r (%)", min_value=0.0, max_value=30.0, value=12.0, step=0.5) / 100.0
    r_mat = st.number_input("Maturação r (%)", min_value=0.0, max_value=30.0, value=4.0, step=0.5) / 100.0

    st.caption("Unit economics do projeto")
    opex_mensal = st.number_input("OPEX mensal (R$)", min_value=0.0, value=float(DEFAULT_OPEX_MENSAL), step=500.0)
    cv_por_cliente = st.number_input("Custo variável por cliente/mês (R$)", min_value=0.0, value=float(DEFAULT_CV_POR_CLIENTE), step=10.0)
    overhead_pct = st.slider("Overhead (% sobre preço)", 0, 30, int(DEFAULT_OVERHEAD_PCT * 100), 1) / 100.0
    n_clientes = st.number_input("Clientes ativos", min_value=0, value=int(DEFAULT_N_CLIENTES), step=1)
    capex_inicial_projeto = st.number_input("CAPEX inicial do projeto (R$)", min_value=0.0, value=35000.0, step=500.0)


# Filtro por UC e período

df_uc = df_all[df_all["uc_id"].astype(str) == str(uc_id)].copy()
df_uc = ensure_datetime_sorted(df_uc)
df_uc = ajustar_energia_se_acumulada(df_uc, "energia_kwh")
max_data = df_uc["timestamp"].max()

if periodo == "Últimas 24h":
    df_f = df_uc[df_uc["timestamp"] >= (max_data - pd.Timedelta(hours=24))]
elif periodo == "Últimos 3 dias":
    df_f = df_uc[df_uc["timestamp"] >= (max_data - pd.Timedelta(days=3))]
elif periodo == "Últimos 7 dias":
    df_f = df_uc[df_uc["timestamp"] >= (max_data - pd.Timedelta(days=7))]
elif periodo == "Últimos 30 dias":
    df_f = df_uc[df_uc["timestamp"] >= (max_data - pd.Timedelta(days=30))]
else:
    df_f = df_uc.copy()

if len(df_f) == 0:
    st.error("Não há dados para o filtro selecionado.")
    st.stop()

# 
# Cálculos globais usados pelas abas
# 
k = kpis_basicos(df_f)
nivel, mensagens, meta = recomendacao_ia(
    df_f,
    fase=fase,
    manejo=manejo,
    tipo_solo=tipo_solo,
    risco_frio=risco_frio,
    dias_pos_floracao=int(dias_pos_floracao),
)

df_sim = aplicar_otimizacao_regras(
    df_f,
    lamina_max=lamina_max,
    chuva_min_mm=chuva_min_mm,
    evitar_ponta=evitar_ponta_otim,
    ponta_inicio=int(ponta_inicio),
    ponta_fim=int(ponta_fim),
)

res = comparar_cenarios(
    df_sim,
    tarifa_kwh_fixa=tarifa_kwh_fixa,
    usar_tarifa_variavel=usar_tarifa_variavel,
    tarifa_fp=tarifa_fp,
    tarifa_ponta=tarifa_ponta,
    ponta_inicio=int(ponta_inicio),
    ponta_fim=int(ponta_fim),
)

dq = data_quality_metrics(df_f)

# Estimativas econômicas derivadas do período selecionado
dias_periodo = max(1.0, (df_f["timestamp"].max() - df_f["timestamp"].min()).total_seconds() / 86400.0)
economia_rs_periodo = float(res.get("economia_rs", 0.0))
economia_kwh_periodo = float(res.get("economia_kwh", 0.0))
economia_rs_mensal = economia_rs_periodo * (30.0 / dias_periodo)
preco_mensal = calc_precificacao_por_valor(economia_rs_mensal, pct_captura=pct_captura, piso=piso_plano, teto=teto_plano)
roi_cliente = calc_roi_payback_cliente(preco_mensal, economia_rs_mensal, investimento_inicial_cliente)
unit = unit_economics_break_even(
    preco_mensal=preco_mensal,
    custo_variavel_por_cliente=cv_por_cliente,
    overhead_pct=overhead_pct,
    opex_mensal=opex_mensal,
    n_clientes=int(n_clientes),
    capex_inicial=capex_inicial_projeto,
)


# Abas principais

tab_visao, tab_monitoramento, tab_ia, tab_economico, tab_analytics, tab_auditoria = st.tabs([
    "🏠 Visão Geral",
    "📊 Monitoramento",
    "🤖 Recomendações IA",
    "⚡ Modelo Econômico",
    "📈 Analytics",
    "⚙️ Auditoria",
])


# ABA 1 — VISÃO GERAL

with tab_visao:
    st.subheader("Visão geral do piloto")
    st.caption("Resumo executivo dos principais indicadores da UC selecionada.")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("UC", str(uc_id))
    c2.metric("Lâmina média", f"{k['lamina_media']:.2f} cm" if np.isfinite(k["lamina_media"]) else "—")
    c3.metric("Energia", f"{fmt_br_number(k['energia_total_kwh'], 1)} kWh")
    c4.metric("Eficiência", f"{k['eficiencia_kwh_m3']:.3f} kWh/m³" if k["eficiencia_kwh_m3"] is not None else "—")
    c5.metric("Chuva 24h", f"{fmt_br_number(k['chuva_24h'], 1)} mm")

    c6, c7, c8, c9 = st.columns(4)
    c6.metric("Volume bombeado", f"{fmt_br_number(k['volume_total_m3'], 1)} m³")
    c7.metric("Horas bomba", f"{int(k['horas_bomba'])} h")
    c8.metric("Economia estimada", f"R$ {fmt_br_money(economia_rs_periodo)}")
    c9.metric("Economia energética", f"{fmt_br_number(economia_kwh_periodo, 1)} kWh")

    st.divider()

    col_a, col_b = st.columns([1.2, 1])
    with col_a:
        st.markdown("### Status de implementação")
        status_df = pd.DataFrame([
            {"Componente": "6 UCs do piloto", "Situação": "Implementado", "Evidência": "DEFAULT_UCS + filtro por UC"},
            {"Componente": "Medição de energia", "Situação": "Implementado", "Evidência": "energia_kwh / potencia_kw"},
            {"Componente": "Sensor de lâmina d’água", "Situação": "Implementado no piloto", "Evidência": "lamina_cm"},
            {"Componente": "SCADA / operação de bombas", "Situação": "Implementado", "Evidência": "bomba_ligada"},
            {"Componente": "Dashboards e KPIs", "Situação": "Implementado", "Evidência": "st.metric + gráficos"},
            {"Componente": "IA explicável por regras", "Situação": "Consolidado no MVP", "Evidência": "recomendacao_ia()"},
            {"Componente": "ML preditivo avançado", "Situação": "Visão futura", "Evidência": "não tratado como produto pronto"},
        ])
        st.dataframe(status_df, use_container_width=True, hide_index=True)

    with col_b:
        st.markdown("### Fluxo de geração de valor")
        st.markdown(
            """
            **Dados de campo**  
            sensores, energia, bomba, chuva  
            ⬇️  
            **Base operacional**  
            limpeza, integração e qualidade  
            ⬇️  
            **BI + Data Science**  
            KPIs, eficiência e cenários  
            ⬇️  
            **Decisão do produtor**  
            recomendações e economia estimada
            """
        )

    if os.path.exists(ARQ_IMG_PATH):
        st.markdown("### Arquitetura da solução")
        st.image(ARQ_IMG_PATH, caption="Arquitetura geral da solução AgroData", use_container_width=True)


# ABA 2 — MONITORAMENTO

with tab_monitoramento:
    st.subheader("Monitoramento operacional")
    st.caption("Visualização limpa das principais variáveis coletadas no piloto.")

    chart_cols = ["timestamp"]
    for col in ["lamina_cm", "energia_kwh", "vazao_m3h", "chuva_mm", "bomba_ligada", "potencia_kw"]:
        if col in df_f.columns:
            chart_cols.append(col)

    col1, col2 = st.columns(2)
    with col1:
        if "lamina_cm" in df_f.columns:
            st.markdown("#### Lâmina d’água (cm)")
            st.line_chart(df_f.set_index("timestamp")[["lamina_cm"]])
        if "vazao_m3h" in df_f.columns:
            st.markdown("#### Vazão (m³/h)")
            st.line_chart(df_f.set_index("timestamp")[["vazao_m3h"]])
    with col2:
        if "energia_kwh" in df_f.columns:
            st.markdown("#### Energia (kWh)")
            st.line_chart(df_f.set_index("timestamp")[["energia_kwh"]])
        if "chuva_mm" in df_f.columns:
            st.markdown("#### Chuva (mm)")
            st.line_chart(df_f.set_index("timestamp")[["chuva_mm"]])

    if "bomba_ligada" in df_f.columns:
        st.markdown("#### Operação da bomba")
        st.area_chart(df_f.set_index("timestamp")[["bomba_ligada"]])

    with st.expander("Ver dados do período", expanded=False):
        st.dataframe(df_f[chart_cols].tail(500), use_container_width=True)


# ABA 3 — RECOMENDAÇÕES IA

with tab_ia:
    st.subheader("Recomendações IA — regras explicáveis")
    st.caption("Módulo de apoio à decisão baseado em regras transparentes e auditáveis.")

    if nivel == "success":
        st.success("Situação operacional adequada para os parâmetros atuais.")
    elif nivel == "warning":
        st.warning("Há pontos de atenção operacional.")
    elif nivel == "error":
        st.error("Há recomendação de intervenção operacional.")
    else:
        st.info("Recomendação informativa.")

    for msg in mensagens:
        st.markdown(f"- {msg}")

    st.divider()
    st.markdown("### Evidências da recomendação")
    meta_df = pd.DataFrame([{
        "Bomba atual": meta.get("bomba_atual"),
        "Lâmina atual (cm)": meta.get("lamina_atual"),
        "Chuva 24h (mm)": meta.get("chuva_24h"),
        "Energia 24h (kWh)": meta.get("energia_24h"),
        "Volume 24h (m³)": meta.get("volume_24h"),
        "Eficiência 24h (kWh/m³)": meta.get("eficiencia_24h"),
        "Baseline eficiência": meta.get("baseline_ef"),
    }])
    st.dataframe(meta_df, use_container_width=True, hide_index=True)

    if st.button("Salvar recomendação em log CSV"):
        path = log_recommendation_event(
            nivel=nivel,
            mensagens=mensagens,
            username=login_user,
            evaluator=evaluator_name,
            periodo_label=periodo,
            fase=fase,
            manejo=manejo,
            tipo_solo=tipo_solo,
            risco_frio=risco_frio,
            dias_pos_floracao=int(dias_pos_floracao),
            meta=meta,
            uc_id=str(uc_id),
        )
        st.success(f"Recomendação salva em: {path}")


# ABA 4 — MODELO ECONÔMICO

with tab_economico:
    st.subheader("Modelo econômico da irrigação")
    st.caption(
        "entradas operacionais → consumo baseline → cenário otimizado → economia → cobrança do serviço."
    )

    st.info(
        "Viabilidade econômica do piloto. "
        "Permite simular a economia por UC/levante, estimar a cobrança da AgroData e calcular o ganho líquido do produtor."
    )

    # ---------------------------
    # 1) Parâmetros gerais do modelo
    # ---------------------------
    st.markdown("### 1. Entradas gerais")
    col_i1, col_i2, col_i3, col_i4 = st.columns(4)
    with col_i1:
        data_inicio_safra = st.date_input("Início da safra", value=df_f["timestamp"].min().date())
    with col_i2:
        data_fim_safra = st.date_input("Fim da safra", value=df_f["timestamp"].max().date())
    with col_i3:
        tarifa_modelo = st.number_input(
            "Tarifa usada no modelo (R$/kWh)",
            min_value=0.0,
            value=float(tarifa_kwh_fixa if not usar_tarifa_variavel else tarifa_fp),
            step=0.01,
            key="modelo_tarifa_kwh",
        )
    with col_i4:
        area_irrigada_ha = st.number_input(
            "Área irrigada considerada (ha)",
            min_value=1.0,
            value=300.0,
            step=10.0,
            key="modelo_area_ha",
        )

    col_i5, col_i6, col_i7, col_i8 = st.columns(4)
    with col_i5:
        dias_modelo = st.number_input(
            "Dias de operação da safra",
            min_value=1,
            value=int(max(1, (pd.Timestamp(data_fim_safra) - pd.Timestamp(data_inicio_safra)).days + 1)),
            step=1,
            key="modelo_dias_operacao",
        )
    with col_i6:
        pct_cobranca_servico = st.slider(
            "Cobrança do serviço (% da economia)",
            min_value=0,
            max_value=50,
            value=int(pct_captura * 100),
            step=1,
            key="modelo_pct_cobranca",
        ) / 100.0
    with col_i7:
        reducao_default = st.slider(
            "Redução otimizada padrão (%)",
            min_value=0,
            max_value=30,
            value=10,
            step=1,
            key="modelo_reducao_default",
        ) / 100.0
    with col_i8:
        eficiencia_default = st.slider(
            "Eficiência média do conjunto (%)",
            min_value=40,
            max_value=100,
            value=85,
            step=1,
            key="modelo_eficiencia_default",
        ) / 100.0

    
    # 2) Tabela editável por UC/levante
    
    st.markdown("### 2. Tabela por UC / levante")
    st.caption(
        "Ajuste potência, horas de operação e redução esperada. "
        
    )

    ucs_modelo = sorted(df_all["uc_id"].astype(str).unique().tolist())
    linhas_default = []
    for uc in ucs_modelo:
        df_tmp = df_all[df_all["uc_id"].astype(str) == uc].copy()
        k_tmp = kpis_basicos(df_tmp) if len(df_tmp) > 0 else {"horas_bomba": 0, "energia_total_kwh": 0}
        horas_por_dia = float(k_tmp.get("horas_bomba", 0)) / max(1, n_dias)
        energia_media_dia = float(k_tmp.get("energia_total_kwh", 0)) / max(1, n_dias)
        # Estimativa simples de potência média em kW e conversão aproximada para cv.
        potencia_kw_est = energia_media_dia / max(1.0, horas_por_dia) if horas_por_dia > 0 else 75.0
        potencia_cv_est = potencia_kw_est / 0.7355
        linhas_default.append({
            "UC": uc,
            "Levante / identificação": f"Levante {uc}",
            "Potência total (cv)": round(max(50.0, potencia_cv_est), 1),
            "Eficiência do conjunto (%)": round(eficiencia_default * 100, 1),
            "Horas/dia baseline": round(max(1.0, horas_por_dia), 1),
            "Dias operação": int(dias_modelo),
            "Redução esperada (%)": round(reducao_default * 100, 1),
        })

    df_inputs_modelo = pd.DataFrame(linhas_default)
    df_editado = st.data_editor(
        df_inputs_modelo,
        use_container_width=True,
        hide_index=True,
        num_rows="dynamic",
        column_config={
            "UC": st.column_config.TextColumn("UC"),
            "Levante / identificação": st.column_config.TextColumn("Levante / identificação"),
            "Potência total (cv)": st.column_config.NumberColumn("Potência total (cv)", min_value=0.0, step=1.0, format="%.1f"),
            "Eficiência do conjunto (%)": st.column_config.NumberColumn("Eficiência do conjunto (%)", min_value=1.0, max_value=100.0, step=1.0, format="%.1f"),
            "Horas/dia baseline": st.column_config.NumberColumn("Horas/dia baseline", min_value=0.0, max_value=24.0, step=0.5, format="%.1f"),
            "Dias operação": st.column_config.NumberColumn("Dias operação", min_value=1, max_value=365, step=1),
            "Redução esperada (%)": st.column_config.NumberColumn("Redução esperada (%)", min_value=0.0, max_value=100.0, step=1.0, format="%.1f"),
        },
        key="modelo_editor_ucs",
    )

    
    # 3) Cálculos do modelo econômico
    
    df_calc = df_editado.copy()
    for col in ["Potência total (cv)", "Eficiência do conjunto (%)", "Horas/dia baseline", "Dias operação", "Redução esperada (%)"]:
        df_calc[col] = pd.to_numeric(df_calc[col], errors="coerce").fillna(0.0)

    df_calc["Potência útil (kW)"] = df_calc["Potência total (cv)"] * 0.7355
    df_calc["Eficiência decimal"] = (df_calc["Eficiência do conjunto (%)"] / 100.0).replace(0, np.nan)
    df_calc["Potência ajustada (kW)"] = df_calc["Potência útil (kW)"] / df_calc["Eficiência decimal"]
    df_calc["kWh baseline"] = df_calc["Potência ajustada (kW)"] * df_calc["Horas/dia baseline"] * df_calc["Dias operação"]
    df_calc["kWh otimizado"] = df_calc["kWh baseline"] * (1.0 - (df_calc["Redução esperada (%)]"] if "Redução esperada (%)]" in df_calc.columns else df_calc["Redução esperada (%)"]) / 100.0)
    df_calc["Economia kWh"] = df_calc["kWh baseline"] - df_calc["kWh otimizado"]
    df_calc["Custo baseline (R$)"] = df_calc["kWh baseline"] * float(tarifa_modelo)
    df_calc["Custo otimizado (R$)"] = df_calc["kWh otimizado"] * float(tarifa_modelo)
    df_calc["Economia bruta (R$)"] = df_calc["Custo baseline (R$)"] - df_calc["Custo otimizado (R$)"]
    df_calc["Cobrança AgroData (R$)"] = df_calc["Economia bruta (R$)"] * float(pct_cobranca_servico)
    df_calc["Economia líquida produtor (R$)"] = df_calc["Economia bruta (R$)"] - df_calc["Cobrança AgroData (R$)"]
    df_calc["Economia por ha (R$/ha)"] = df_calc["Economia líquida produtor (R$)"] / max(1.0, float(area_irrigada_ha))

    # Remove coluna técnica antes de exibir
    df_resultado_modelo = df_calc.drop(columns=["Eficiência decimal"], errors="ignore")

    total_baseline_kwh = float(df_calc["kWh baseline"].sum())
    total_otim_kwh = float(df_calc["kWh otimizado"].sum())
    total_economia_kwh = float(df_calc["Economia kWh"].sum())
    total_custo_base = float(df_calc["Custo baseline (R$)"].sum())
    total_custo_otim = float(df_calc["Custo otimizado (R$)"].sum())
    total_economia_bruta = float(df_calc["Economia bruta (R$)"].sum())
    total_receita_agrodata = float(df_calc["Cobrança AgroData (R$)"].sum())
    total_economia_liquida = float(df_calc["Economia líquida produtor (R$)"].sum())
    economia_por_ha = total_economia_liquida / max(1.0, float(area_irrigada_ha))

    st.markdown("### 3. Resultados consolidados")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Consumo baseline", f"{fmt_br_number(total_baseline_kwh, 1)} kWh")
    m2.metric("Consumo otimizado", f"{fmt_br_number(total_otim_kwh, 1)} kWh")
    m3.metric("Economia energética", f"{fmt_br_number(total_economia_kwh, 1)} kWh")
    m4.metric("Redução média", f"{(total_economia_kwh / total_baseline_kwh * 100):.1f}%" if total_baseline_kwh > 0 else "—")

    m5, m6, m7, m8 = st.columns(4)
    m5.metric("Custo baseline", f"R$ {fmt_br_money(total_custo_base)}")
    m6.metric("Economia bruta", f"R$ {fmt_br_money(total_economia_bruta)}")
    m7.metric("Cobrança AgroData", f"R$ {fmt_br_money(total_receita_agrodata)}")
    m8.metric("Economia líquida produtor", f"R$ {fmt_br_money(total_economia_liquida)}")

    st.caption(f"Economia líquida estimada por hectare: **R$ {fmt_br_money(economia_por_ha)}/ha**")

    st.markdown("### 4. Resultado por UC")
    st.dataframe(
        df_resultado_modelo[[
            "UC", "Levante / identificação", "Potência total (cv)", "Horas/dia baseline", "Dias operação",
            "Redução esperada (%)", "kWh baseline", "kWh otimizado", "Economia kWh",
            "Custo baseline (R$)", "Custo otimizado (R$)", "Economia bruta (R$)",
            "Cobrança AgroData (R$)", "Economia líquida produtor (R$)", "Economia por ha (R$/ha)"
        ]],
        use_container_width=True,
        hide_index=True,
    )

    col_g1, col_g2 = st.columns(2)
    with col_g1:
        st.markdown("#### Economia bruta por UC")
        st.bar_chart(df_calc.set_index("UC")[["Economia bruta (R$)"]])
    with col_g2:
        st.markdown("#### kWh baseline x otimizado")
        st.bar_chart(df_calc.set_index("UC")[["kWh baseline", "kWh otimizado"]])

    
    # 4) Cenários de sensibilidade — 5%, 10%, 15%, 20%
    
    st.markdown("### 5. Sensibilidade por percentual de redução")
    cenarios_pct = [0.05, 0.10, 0.15, 0.20]
    cenarios_rows = []
    for pct in cenarios_pct:
        econ_kwh = total_baseline_kwh * pct
        econ_bruta = econ_kwh * float(tarifa_modelo)
        receita = econ_bruta * float(pct_cobranca_servico)
        liquida = econ_bruta - receita
        cenarios_rows.append({
            "Cenário": f"Redução {pct*100:.0f}%",
            "Economia kWh": econ_kwh,
            "Economia bruta (R$)": econ_bruta,
            "Cobrança AgroData (R$)": receita,
            "Economia líquida produtor (R$)": liquida,
            "Economia por ha (R$/ha)": liquida / max(1.0, float(area_irrigada_ha)),
        })
    df_sens = pd.DataFrame(cenarios_rows)
    st.dataframe(df_sens, use_container_width=True, hide_index=True)
    st.bar_chart(df_sens.set_index("Cenário")[["Economia bruta (R$)", "Economia líquida produtor (R$)"]])

    
    # 5) Comparação com o cálculo baseado nos dados do protótipo
    
    st.markdown("### 6. Comparação com simulação operacional do protótipo")
    st.caption(
        "Este quadro compara o modelo econômico parametrizado acima com o resultado calculado diretamente dos dados da UC selecionada."
    )
    comparativo_modelos = pd.DataFrame([
        {
            "Origem do cálculo": "Modelo econômico por potência/horas",
            "Energia baseline (kWh)": total_baseline_kwh,
            "Energia otimizada (kWh)": total_otim_kwh,
            "Economia kWh": total_economia_kwh,
            "Economia R$": total_economia_bruta,
        },
        {
            "Origem do cálculo": "Dados do protótipo / regras de otimização",
            "Energia baseline (kWh)": res["energia_base"],
            "Energia otimizada (kWh)": res["energia_otim"],
            "Economia kWh": res["economia_kwh"],
            "Economia R$": res["economia_rs"],
        },
    ])
    st.dataframe(comparativo_modelos, use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("### 7. Precificação, ROI, payback e unit economics")
    economia_rs_mensal_modelo = total_economia_bruta * (30.0 / max(1.0, float(dias_modelo)))
    preco_mensal_modelo = calc_precificacao_por_valor(
        economia_rs_mensal_modelo,
        pct_captura=pct_cobranca_servico,
        piso=piso_plano,
        teto=teto_plano,
    )
    roi_cliente_modelo = calc_roi_payback_cliente(preco_mensal_modelo, economia_rs_mensal_modelo, investimento_inicial_cliente)
    unit_modelo = unit_economics_break_even(
        preco_mensal=preco_mensal_modelo,
        custo_variavel_por_cliente=cv_por_cliente,
        overhead_pct=overhead_pct,
        opex_mensal=opex_mensal,
        n_clientes=int(n_clientes),
        capex_inicial=capex_inicial_projeto,
    )

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Economia mensal estimada", f"R$ {fmt_br_money(economia_rs_mensal_modelo)}")
    r2.metric("Preço mensal sugerido", f"R$ {fmt_br_money(preco_mensal_modelo)}")
    r3.metric("ROI mensal cliente", f"{roi_cliente_modelo['roi_mensal'] * 100:.1f}%" if roi_cliente_modelo["roi_mensal"] is not None else "—")
    r4.metric("Payback cliente", f"{roi_cliente_modelo['payback_meses']:.1f} meses" if roi_cliente_modelo["payback_meses"] is not None else "—")

    unit_df = pd.DataFrame([unit_modelo])
    st.dataframe(unit_df, use_container_width=True, hide_index=True)

    # VPL com base nos cenários da tabela de sensibilidade
    df_vpl_modelo = df_sens.copy()
    df_vpl_modelo["VPL AgroData (R$)"] = df_vpl_modelo["Cobrança AgroData (R$)"].apply(lambda x: npv_anuidades(x * (365.0 / max(1.0, float(dias_modelo))), taxa_desconto, int(horizonte_anos)))
    df_vpl_modelo["VPL Produtor (R$)"] = df_vpl_modelo["Economia líquida produtor (R$)"].apply(lambda x: npv_anuidades(x * (365.0 / max(1.0, float(dias_modelo))), taxa_desconto, int(horizonte_anos)))
    st.markdown("#### VPL por cenário de sensibilidade")
    st.dataframe(df_vpl_modelo, use_container_width=True, hide_index=True)

    
    # 6) Downloads e evidências

    st.markdown("### 8. Exportar evidências")
    csv_modelo = df_resultado_modelo.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Baixar resultado do modelo econômico (CSV)",
        data=csv_modelo,
        file_name=f"modelo_economico_{uc_id}_{RUN_ID}.csv",
        mime="text/csv",
    )

    resumo_modelo = pd.DataFrame([{
        "run_id": RUN_ID,
        "uc_selecionada": uc_id,
        "dias_modelo": dias_modelo,
        "area_irrigada_ha": area_irrigada_ha,
        "tarifa_rs_kwh": tarifa_modelo,
        "consumo_baseline_kwh": total_baseline_kwh,
        "consumo_otimizado_kwh": total_otim_kwh,
        "economia_kwh": total_economia_kwh,
        "custo_baseline_rs": total_custo_base,
        "custo_otimizado_rs": total_custo_otim,
        "economia_bruta_rs": total_economia_bruta,
        "cobranca_agrodata_rs": total_receita_agrodata,
        "economia_liquida_produtor_rs": total_economia_liquida,
        "economia_por_ha_rs": economia_por_ha,
        "preco_mensal_sugerido_rs": preco_mensal_modelo,
    }])

    if st.button("Salvar modelo econômico em log CSV"):
        _safe_mkdir(LOG_DIR)
        if not os.path.exists(ECO_LOG_PATH):
            resumo_modelo.to_csv(ECO_LOG_PATH, index=False, encoding="utf-8")
        else:
            resumo_modelo.to_csv(ECO_LOG_PATH, mode="a", header=False, index=False, encoding="utf-8")
        st.success(f"Modelo econômico salvo em: {ECO_LOG_PATH}")


# ABA 5 — ANALYTICS

with tab_analytics:
    st.subheader("Analytics e comparação entre UCs")
    st.caption("Análises exploratórias para apoiar visão de escala e desempenho operacional.")

    rows = []
    for uc in sorted(df_all["uc_id"].astype(str).unique().tolist()):
        df_tmp = df_all[df_all["uc_id"].astype(str) == uc].copy()
        df_tmp = ensure_datetime_sorted(df_tmp)
        max_tmp = df_tmp["timestamp"].max()
        if periodo == "Últimas 24h":
            df_tmp = df_tmp[df_tmp["timestamp"] >= (max_tmp - pd.Timedelta(hours=24))]
        elif periodo == "Últimos 3 dias":
            df_tmp = df_tmp[df_tmp["timestamp"] >= (max_tmp - pd.Timedelta(days=3))]
        elif periodo == "Últimos 7 dias":
            df_tmp = df_tmp[df_tmp["timestamp"] >= (max_tmp - pd.Timedelta(days=7))]
        elif periodo == "Últimos 30 dias":
            df_tmp = df_tmp[df_tmp["timestamp"] >= (max_tmp - pd.Timedelta(days=30))]
        if len(df_tmp) < 2:
            continue
        k_uc = kpis_basicos(df_tmp)
        rows.append({
            "UC": uc,
            "Lâmina média (cm)": k_uc["lamina_media"],
            "Energia (kWh)": k_uc["energia_total_kwh"],
            "Volume (m³)": k_uc["volume_total_m3"],
            "Eficiência (kWh/m³)": k_uc["eficiencia_kwh_m3"],
            "Horas bomba": k_uc["horas_bomba"],
        })

    df_uc_comp = pd.DataFrame(rows)
    st.dataframe(df_uc_comp, use_container_width=True, hide_index=True)

    if len(df_uc_comp) > 0:
        st.markdown("#### Comparativo por UC")
        chart_data = df_uc_comp.set_index("UC")[["Energia (kWh)", "Volume (m³)", "Horas bomba"]]
        st.bar_chart(chart_data)

    st.divider()
    st.markdown("### Correlação entre variáveis operacionais")
    num_cols = [c for c in ["lamina_cm", "energia_kwh", "vazao_m3h", "chuva_mm", "bomba_ligada", "potencia_kw", "et0_mm"] if c in df_f.columns]
    if len(num_cols) >= 2:
        corr = df_f[num_cols].corr(numeric_only=True)
        st.dataframe(corr, use_container_width=True)
    else:
        st.info("Não há variáveis numéricas suficientes para matriz de correlação.")

    st.markdown("### Qualidade dos dados")
    dq_cols = st.columns(4)
    dq_cols[0].metric("Registros", dq["n_registros"])
    dq_cols[1].metric("Missing lâmina", f"{dq['missing_lamina_pct']:.1f}%")
    dq_cols[2].metric("Missing energia", f"{dq['missing_energia_pct']:.1f}%")
    dq_cols[3].metric("dt mediana", f"{dq['dt_mediana_h']:.2f} h" if np.isfinite(dq["dt_mediana_h"]) else "—")


# ABA 6 — AUDITORIA

with tab_auditoria:
    st.subheader("Auditoria, evidências e governança")
    st.caption("Logs, rastreabilidade, qualidade do dado e relatórios do avaliador.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Versão", APP_VERSION)
    c2.metric("RUN_ID", RUN_ID)
    c3.metric("Usuário", login_user)
    c4.metric("Avaliador", evaluator_name)

    st.divider()
    st.markdown("### Modo avaliador — relatório por UC")
    salvar_detalhe_uc = st.checkbox("Salvar detalhe por UC (CSV grande)", value=False)
    if st.button("Gerar relatório consolidado por UC"):
        csv_path, png_rs_path, png_kwh_path = relatorio_avaliador_por_uc(
            df_all=df_all,
            periodo_label=periodo,
            lamina_max=lamina_max,
            chuva_min_mm=chuva_min_mm,
            usar_tarifa_variavel=usar_tarifa_variavel,
            tarifa_kwh_fixa=tarifa_kwh_fixa,
            tarifa_fp=tarifa_fp,
            tarifa_ponta=tarifa_ponta,
            ponta_inicio=int(ponta_inicio),
            ponta_fim=int(ponta_fim),
            evitar_ponta=evitar_ponta_otim,
            salvar_detalhe_por_uc=salvar_detalhe_uc,
        )
        st.success("Relatório gerado com sucesso.")
        with open(csv_path, "rb") as f:
            st.download_button("Baixar relatório por UC (CSV)", data=f, file_name=os.path.basename(csv_path), mime="text/csv")
        if os.path.exists(png_rs_path):
            with open(png_rs_path, "rb") as f:
                st.download_button("Baixar gráfico economia R$ (PNG)", data=f, file_name=os.path.basename(png_rs_path), mime="image/png")
        if os.path.exists(png_kwh_path):
            with open(png_kwh_path, "rb") as f:
                st.download_button("Baixar gráfico economia kWh (PNG)", data=f, file_name=os.path.basename(png_kwh_path), mime="image/png")

    st.markdown("### Qualidade do dado")
    if st.button("Salvar qualidade do dado (CSV)"):
        path = log_data_quality_event(
            username=login_user,
            evaluator=evaluator_name,
            periodo_label=periodo,
            metrics=dq,
            uc_id=str(uc_id),
        )
        st.success(f"Qualidade do dado salva em: {path}")

    st.markdown("### Downloads dos logs")
    for pth, label in [
        (ACCESS_LOG_PATH, "access_log.csv"),
        (REC_LOG_PATH, "recommendations_log.csv"),
        (RES_LOG_PATH, "resultados_piloto.csv"),
        (ECO_LOG_PATH, "modelo_economico.csv"),
        (ECO_CENARIOS_LOG_PATH, "cenarios_vpl_fenologia.csv"),
        (DQ_LOG_PATH, "data_quality_log.csv"),
    ]:
        if os.path.exists(pth):
            with open(pth, "rb") as f:
                st.download_button(f"Baixar {label}", data=f, file_name=os.path.basename(pth), mime="text/csv")

    with st.expander("Visualizar logs recentes", expanded=False):
        for pth, label in [
            (ACCESS_LOG_PATH, "Acessos"),
            (REC_LOG_PATH, "Recomendações"),
            (RES_LOG_PATH, "Resultados"),
            (DQ_LOG_PATH, "Qualidade do dado"),
        ]:
            if os.path.exists(pth):
                st.markdown(f"#### {label}")
                try:
                    st.dataframe(pd.read_csv(pth).tail(20), use_container_width=True)
                except Exception as exc:
                    st.warning(f"Não foi possível ler {pth}: {exc}")

    st.divider()
    if st.button("Limpar logs"):
        removed = clear_logs()
        if removed:
            st.success("Logs removidos: " + ", ".join(removed))
        else:
            st.info("Nenhum log para remover.")


# Rodapé 

st.divider()
st.caption(
    f"Protótipo acadêmico AgroData — BI + Data Science + IA explicável | Versão {APP_VERSION} | RUN_ID {RUN_ID} | "
    
)
