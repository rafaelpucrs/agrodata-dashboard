# =========================================================
# AgroData — Irrigação (BI + Data Science + IA)
# Protótipo acadêmico (TCC) — Versão completa (reescrita e consolidada)
#
# ✅ Login + logs (RUN_ID + APP_VERSION) + auditoria (CSV)
# ✅ Suporte multi-UC (6 UCs do piloto) + "Modo Avaliador" (1 clique)
# ✅ Simulação realista 90 dias:
#    - chuva por eventos
#    - ET0 proxy
#    - lâmina dinâmica (balanço hídrico simplificado)
#    - operação de bomba por horário (evitar ponta opcional)
#    - energia_kwh por intervalo (compatível com acumulada)
# ✅ Intervalo irregular (dt_horas) + integração robusta (energia/volume)
# ✅ KPIs (BI) + recomendações explicáveis (regras)
# ✅ Rolling 24h sem desalinhamento
# ✅ Baseline vs Otimizado
# ✅ Tarifa fixa OU horo-sazonal (ponta/fora-ponta) + evidência
# ✅ Modelo econômico completo (TCC):
#    - Cenários (5% / 10% / 15%)
#    - VPL (horizonte)
#    - Redução variável por fase fenológica
#    - Precificação (captura de valor)
#    - ROI/Payback (cliente)
#    - Unit Economics + Break-even (clientes) + Payback CAPEX do projeto
#    - Gráficos (PNG) + downloads
# ✅ Qualidade de dado (escala) + evidências (CSV)
# ✅ Cache de carregamento
#
# Estrutura recomendada:
# - app.py (este arquivo)
# - data/dados_irrigacao.csv (opcional)
# - data/arquitetura_irrigacao.png (opcional)
# - logs/ (gerado automaticamente)
#
# Observação:
# - Fallback admin/admin é somente para protótipo acadêmico.
# - Para produção: st.secrets, OAuth/SSO, BD, etc.
# =========================================================

import os
import io
import hmac
import uuid
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


# =========================================================
# CONFIG
# =========================================================
st.set_page_config(page_title="AgroData — Irrigação", layout="wide")


# =========================================================
# CONSTANTES / PATHS
# =========================================================
APP_VERSION = "1.0.1"
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


# =========================================================
# SESSÃO — ID de execução (evidência)
# =========================================================
if "run_id" not in st.session_state:
    st.session_state["run_id"] = str(uuid.uuid4())[:8]
RUN_ID = st.session_state["run_id"]


# =========================================================
# UTIL: Credenciais + Nome do Avaliador
# =========================================================
def get_settings():
    """
    Lê configurações via st.secrets (produção) ou variáveis de ambiente (local).
    Observação: fallback admin/admin é apenas para protótipo acadêmico.
    """
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


# =========================================================
# DADOS — helpers para dt e integração (intervalo irregular)
# =========================================================
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


# =========================================================
# SIMULAÇÃO REALISTA (90 dias) + Multi-UC
# =========================================================
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


# =========================================================
# TARIFA HORO-SAZONAL (PONTA/FP) + CUSTOS
# =========================================================
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


# =========================================================
# KPIs (BI)
# =========================================================
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


# =========================================================
# RECOMENDAÇÃO (IA EXPLICÁVEL por regras)
# =========================================================
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


# =========================================================
# RESULTADOS — baseline vs otimizado + tarifa fixa/variável
# =========================================================
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


# =========================================================
# MODO AVALIADOR (1 clique) — relatório por UC + PNGs
# =========================================================
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


# =========================================================
# MODELO ECONÔMICO + UNIT ECONOMICS + BREAK-EVEN
# =========================================================
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


# =========================================================
# UI HELPERS
# =========================================================
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


# =========================================================
# UI — TOPO + SIDEBAR (Sair)
# =========================================================
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


# =========================================================
# APP — dados + filtros
# =========================================================
st.sidebar.header("Fonte de dados")
usar_simulacao = st.sidebar.toggle("Usar simulação realista (90 dias / 6 UCs)", value=True)
seed = st.sidebar.number_input("Seed simulação", min_value=1, value=42, step=1)
n_dias = st.sidebar.number_input("Dias simulados", min_value=30, max_value=180, value=DEFAULT_N_DIAS, step=5)

st.sidebar.subheader("Tarifa (energia)")
usar_tarifa_variavel = st.sidebar.toggle("Usar tarifa horo-sazonal (ponta/fora-ponta)", value=True)
tarifa_kwh_fixa = st.sidebar.number_input("Tarifa fixa (R$/kWh)", min_value=0.0, value=float(DEFAULT_TARIFA_FIXA), step=0.01)

tarifa_fp = st.sidebar.number_input("Tarifa fora-ponta (R$/kWh)", min_value=0.0, value=float(DEFAULT_TARIFA_FP), step=0.01)
tarifa_ponta = st.sidebar.number_input("Tarifa ponta (R$/kWh)", min_value=0.0, value=float(DEFAULT_TARIFA_PONTA), step=0.01)
ponta_inicio = st.sidebar.number_input("Início ponta (hora)", min_value=0, max_value=23, value=int(DEFAULT_PONTA_INICIO), step=1)
ponta_fim = st.sidebar.number_input("Fim ponta (hora)", min_value=1, max_value=24, value=int(DEFAULT_PONTA_FIM), step=1)

evitar_ponta_sim = st.sidebar.toggle("Simulação baseline evita ponta (realista)", value=True)

df_all = carregar_dados(
    usar_simulacao=bool(usar_simulacao),
    seed=int(seed),
    n_dias=int(n_dias),
    evitar_ponta=bool(evitar_ponta_sim),
    ponta_inicio=int(ponta_inicio),
    ponta_fim=int(ponta_fim),
)

st.sidebar.header("Filtros")
uc_id = st.sidebar.selectbox("UC (piloto)", sorted(df_all["uc_id"].astype(str).unique().tolist()))
df_uc = df_all[df_all["uc_id"].astype(str) == str(uc_id)].copy()
df_uc = ensure_datetime_sorted(df_uc)
df_uc = ajustar_energia_se_acumulada(df_uc, "energia_kwh")
max_data = df_uc["timestamp"].max()

periodo = st.sidebar.selectbox(
    "Período",
    options=["Últimas 24h", "Últimos 3 dias", "Últimos 7 dias", "Últimos 30 dias", "Tudo"],
    index=3,
)

st.sidebar.header("Parâmetros agronômicos (arroz)")
fase = st.sidebar.selectbox(
    "Fase do cultivo",
    ["Vegetativa", "Reprodutiva", "Emborrachamento/Floração", "Maturação"],
    index=0,
)
manejo = st.sidebar.selectbox(
    "Manejo de irrigação",
    ["Contínuo", "Intermitente (fornecimento intermitente)"],
    index=0,
)
tipo_solo = st.sidebar.selectbox(
    "Tipo de solo (supressão)",
    ["Argiloso", "Arenoso/bem drenado"],
    index=0,
)
risco_frio = st.sidebar.checkbox("Risco de frio (<16°C) no emborrachamento?", value=False)
dias_pos_floracao = st.sidebar.number_input("Dias após floração plena", min_value=0, value=0, step=1)

st.sidebar.header("Parâmetros (otimização)")
lamina_max = st.sidebar.slider("Lâmina máxima (cm)", min_value=7.0, max_value=20.0, value=10.0, step=0.5)
chuva_min_mm = st.sidebar.slider("Chuva 24h (mm) para reduzir/adiar", min_value=0.0, max_value=60.0, value=12.0, step=1.0)
evitar_ponta_otim = st.sidebar.toggle("Otimização: evitar operar na ponta", value=True)

# Modelo econômico (TCC)
st.sidebar.header("Modelo econômico (TCC)")
pct_captura = st.sidebar.slider("Captura de valor (% da economia)", 5, 30, int(DEFAULT_PCT_CAPTURA * 100), 1) / 100.0
alpha_receita = st.sidebar.slider("Receita do projeto (% da economia bruta)", 5, 30, int(DEFAULT_ALPHA_RECEITA * 100), 1) / 100.0
investimento_inicial_cliente = st.sidebar.number_input("Investimento inicial (cliente) (R$)", min_value=0.0, value=12400.0, step=100.0)

st.sidebar.subheader("Safra (irrigação contínua)")
duracao_safra_dias = st.sidebar.slider("Duração da lâmina contínua (dias)", 80, 120, 90, 1)
piso_plano = st.sidebar.number_input("Piso do plano (R$/mês)", min_value=0.0, value=1200.0, step=100.0)
teto_plano = st.sidebar.number_input("Teto do plano (R$/mês)", min_value=0.0, value=3800.0, step=100.0)

st.sidebar.subheader("VPL")
taxa_desconto = st.sidebar.number_input("Taxa de desconto (a.a.)", min_value=0.0, value=float(DEFAULT_TAXA_DESCONTO), step=0.01)
horizonte_anos = st.sidebar.number_input("Horizonte (anos)", min_value=1, value=int(DEFAULT_HORIZONTE_ANOS), step=1)

st.sidebar.subheader("Fenologia: redução variável r (%) por fase")
r_veg = st.sidebar.number_input("Vegetativa r (%)", min_value=0.0, max_value=30.0, value=6.0, step=0.5) / 100.0
r_rep = st.sidebar.number_input("Reprodutiva r (%)", min_value=0.0, max_value=30.0, value=10.0, step=0.5) / 100.0
r_flo = st.sidebar.number_input("Emborrachamento/Floração r (%)", min_value=0.0, max_value=30.0, value=14.0, step=0.5) / 100.0
r_mat = st.sidebar.number_input("Maturação r (%)", min_value=0.0, max_value=30.0, value=3.0, step=0.5) / 100.0

st.sidebar.subheader("Unit Economics (SaaS do projeto)")
opex_mensal = st.sidebar.number_input("OPEX mensal (custos fixos) (R$)", min_value=0.0, value=float(DEFAULT_OPEX_MENSAL), step=500.0)
cv_por_cliente = st.sidebar.number_input("Custo variável por cliente/mês (R$)", min_value=0.0, value=float(DEFAULT_CV_POR_CLIENTE), step=10.0)
overhead_pct = st.sidebar.slider("Overhead (% sobre preço)", 0, 30, int(DEFAULT_OVERHEAD_PCT * 100), 1) / 100.0
n_clientes = st.sidebar.number_input("Clientes ativos (N)", min_value=0, value=int(DEFAULT_N_CLIENTES), step=1)
capex_inicial_projeto = st.sidebar.number_input("CAPEX inicial do projeto (R$)", min_value=0.0, value=35000.0, step=500.0)

# Evidências / logs
st.sidebar.header("Evidências (logs)")
colL, colR = st.sidebar.columns(2)
with colL:
    if st.button("Limpar logs"):
        removed = clear_logs()
        if removed:
            st.success("Logs removidos: " + ", ".join(removed))
        else:
            st.info("Nenhum log para remover.")

for pth, label in [
    (ACCESS_LOG_PATH, "Baixar access_log.csv"),
    (REC_LOG_PATH, "Baixar recommendations_log.csv"),
    (RES_LOG_PATH, "Baixar resultados_piloto.csv"),
    (ECO_LOG_PATH, "Baixar modelo_economico.csv"),
    (ECO_CENARIOS_LOG_PATH, "Baixar cenarios_vpl_fenologia.csv"),
    (DQ_LOG_PATH, "Baixar data_quality_log.csv"),
]:
    if os.path.exists(pth):
        with open(pth, "rb") as f:
            st.sidebar.download_button(label, data=f, file_name=os.path.basename(pth), mime="text/csv")

st.sidebar.divider()
st.sidebar.header("Modo Avaliador (1 clique)")
salvar_detalhe_uc = st.sidebar.checkbox("Salvar detalhe por UC (CSV grande)", value=False)

if st.sidebar.button("Gerar relatório por UC"):
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
    st.sidebar.success("Relatório gerado!")
    with open(csv_path, "rb") as f:
        st.sidebar.download_button("Baixar relatório (CSV)", data=f, file_name=os.path.basename(csv_path), mime="text/csv")
    if os.path.exists(png_rs_path):
        with open(png_rs_path, "rb") as f:
            st.sidebar.download_button("Baixar gráfico (PNG) — R$", data=f, file_name=os.path.basename(png_rs_path), mime="image/png")
    if os.path.exists(png_kwh_path):
        with open(png_kwh_path, "rb") as f:
            st.sidebar.download_button("Baixar gráfico (PNG) — kWh", data=f, file_name=os.path.basename(png_kwh_path), mime="image/png")


# Filtra df por período (na UC selecionada)
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

tabs = st.tabs([
    "Dashboard",
    "Arquitetura da Solução",
    "Metodologia (simulação)",
    "Resultados (piloto)",
    "Modelo Econômico (TCC)",
    "Modelo Matemático (equações)",
    "Segmentação (TCC)",
    "MVP vs Visão Futura",
    "Escala & Gargalos",
    "Qualidade do Dado (escala)",
])


# =========================================================
# TAB 0 — DASHBOARD
# =========================================================
with tabs[0]:
    k = kpis_basicos(df_f)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Lâmina média (cm)", f"{k['lamina_media']:.2f}" if np.isfinite(k["lamina_media"]) else "—")
    c2.metric("Energia total (kWh)", f"{k['energia_total_kwh']:.1f}")
    c3.metric("Volume total (m³)", f"{k['volume_total_m3']:.1f}")
    c4.metric("Eficiência (kWh/m³)", f"{(k['eficiencia_kwh_m3'] if k['eficiencia_kwh_m3'] is not None else 0):.3f}")
    c5.metric("Horas bomba ligada", f"{int(k['horas_bomba'])} h")

    st.subheader("Recomendação automática (IA explicável) e alertas")
    nivel, mensagens, meta = recomendacao_ia(
        df_f,
        fase=fase,
        manejo=manejo,
        tipo_solo=tipo_solo,
        risco_frio=risco_frio,
        dias_pos_floracao=int(dias_pos_floracao),
    )

    log_key = f"log_rec::{uc_id}::{periodo}::{fase}::{manejo}::{tipo_solo}::{risco_frio}::{int(dias_pos_floracao)}::{df_f['timestamp'].max()}::{RUN_ID}"
    if st.session_state.get(log_key) is not True:
        log_recommendation_event(
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
        st.session_state[log_key] = True

    texto = "\n".join([f"- {m}" for m in mensagens])
    if nivel == "success":
        st.success(texto)
    elif nivel == "info":
        st.info(texto)
    elif nivel == "warning":
        st.warning(texto)
    else:
        st.error(texto)

    st.subheader("Mini-resumo operacional (últimas 6h/24h)")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Bomba agora", "Ligada" if meta["bomba_atual"] == 1 else "Desligada")
    m2.metric("Horas bomba (6h)", f"{meta['bomba_horas_6h']} h")
    m3.metric("Energia (6h)", f"{meta['energia_6h']:.1f} kWh")
    m4.metric("Eficiência (6h)", f"{(meta['eficiencia_6h'] if meta['eficiencia_6h'] is not None else 0):.3f} kWh/m³")

    n1, n2, n3, n4 = st.columns(4)
    n1.metric("Horas bomba (24h)", f"{meta['bomba_horas_24h']} h")
    n2.metric("Energia (24h)", f"{meta['energia_24h']:.1f} kWh")
    n3.metric("Volume (24h)", f"{meta['volume_24h']:.1f} m³")
    n4.metric("Lâmina (24h)", f"{meta['lamina_min_24h']:.1f}–{meta['lamina_max_24h']:.1f} cm")

    st.subheader("Tendências (período selecionado)")
    cc1, cc2 = st.columns(2)

    with cc1:
        st.caption("Lâmina d’água (cm)")
        if "lamina_cm" in df_f.columns:
            st.line_chart(df_f.set_index("timestamp")["lamina_cm"])
        else:
            st.info("Coluna lamina_cm não encontrada.")

    with cc2:
        st.caption("Energia (kWh) e Vazão (m³/h)")
        cols = []
        if "energia_kwh" in df_f.columns:
            cols.append("energia_kwh")
        if "vazao_m3h" in df_f.columns:
            cols.append("vazao_m3h")
        if cols:
            st.line_chart(df_f.set_index("timestamp")[cols])
        else:
            st.info("Colunas de energia/vazão não encontradas.")

    st.subheader("Base de dados (amostra)")
    st.dataframe(df_f.tail(60), use_container_width=True)


# =========================================================
# TAB 1 — ARQUITETURA
# =========================================================
with tabs[1]:
    st.subheader("Arquitetura da Solução Proposta")
    st.write(
        "A solução é estruturada em camadas para viabilizar coleta contínua, armazenamento histórico, "
        "processamento analítico e visualização. Fluxo: "
        "**Sensores → SCADA → Banco de Dados → Processamento (BI/DS/IA) → Dashboards/Alertas**."
    )

    st.markdown(
        """
**Componentes principais:**
- **Sensores / medições:** lâmina d’água, vazão, energia elétrica e variáveis climáticas.
- **SCADA / automação:** consolida leituras e registra eventos operacionais.
- **Banco de dados:** armazena histórico e padroniza dados para análises.
- **Processamento (BI + Data Science + IA):** KPIs, detecção de desvios e recomendações.
- **Dashboards + alertas:** suporte à decisão para o manejo da irrigação.
        """.strip()
    )

    if os.path.exists(ARQ_IMG_PATH):
        st.image(ARQ_IMG_PATH, caption="Arquitetura da solução (imagem)", use_container_width=True)
    else:
        st.caption(f"Imagem de arquitetura não encontrada em: {ARQ_IMG_PATH}")


# =========================================================
# TAB 2 — METODOLOGIA
# =========================================================
with tabs[2]:
    st.subheader("Metodologia (simulação)")
    st.markdown(
        """
**1) Coleta/Geração de dados**
- Leitura de dados reais quando disponíveis; caso contrário, simulação realista multi-UC (90 dias).
- Variáveis: lâmina, vazão, energia, chuva, ET0 proxy e estado da bomba.

**2) Tratamento e organização**
- Padronização, limpeza e ordenação temporal.
- Preparado para *intervalo irregular* via \\(dt\\) (integração robusta de volume/energia).
- Compatibilidade com energia acumulada (conversão para energia por intervalo).

**3) Indicadores (BI)**
- KPIs operacionais e gráficos de tendência.

**4) Suporte à decisão (IA explicável)**
- Regras interpretáveis para alertas e recomendações.
- Otimização por regras (baseline vs otimizado) e opção de evitar ponta.

**5) Evidências (Auditoria)**
- Logs em CSV com versão do app e ID de execução (RUN_ID).
- Relatório por UC (Modo Avaliador) + gráficos (PNG).
        """.strip()
    )


# =========================================================
# TAB 3 — RESULTADOS PILOTO
# =========================================================
with tabs[3]:
    st.subheader("Resultados (piloto) — Baseline vs Otimizado (simulação por regras)")
    st.caption(
        "Comparação do cenário atual (baseline) com um cenário otimizado baseado em regras interpretáveis "
        "(ex.: adiar bombeamento após chuva relevante, evitar excesso de lâmina e (opcional) evitar ponta)."
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

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Energia Base (kWh)", f"{res['energia_base']:.1f}")
    c2.metric("Energia Otim (kWh)", f"{res['energia_otim']:.1f}")
    c3.metric("Economia (kWh)", f"{res['economia_kwh']:.1f}")
    c4.metric("Economia (R$)", fmt_br_money(res["economia_rs"]))

    st.markdown("### Eficiência e operação")
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Eficiência Base (kWh/m³)", f"{(res['ef_base'] if np.isfinite(res['ef_base']) else 0):.3f}")
    d2.metric("Eficiência Otim (kWh/m³)", f"{(res['ef_otim'] if np.isfinite(res['ef_otim']) else 0):.3f}")
    d3.metric("Horas bomba Base", f"{res['horas_bomba_base']} h")
    d4.metric("Horas bomba Otim", f"{res['horas_bomba_otim']} h")

    st.markdown("### Redução de bombeamento (proxy de água)")
    st.info(
        f"Volume bombeado (Base): {res['volume_base']:.1f} m³ | "
        f"(Otimizado): {res['volume_otim']:.1f} m³ | "
        f"Redução estimada: {res['reducao_volume']:.1f} m³"
    )

    if usar_tarifa_variavel:
        st.markdown("### Tarifa horo-sazonal (ponta/fora-ponta) — evidência")
        t1, t2, t3, t4 = st.columns(4)
        t1.metric("Custo base (Ponta)", fmt_br_money(res.get("custo_base_ponta", np.nan)))
        t2.metric("Custo base (Fora-ponta)", fmt_br_money(res.get("custo_base_fp", np.nan)))
        t3.metric("Custo otim (Ponta)", fmt_br_money(res.get("custo_otim_ponta", np.nan)))
        t4.metric("Custo otim (Fora-ponta)", fmt_br_money(res.get("custo_otim_fp", np.nan)))

    st.markdown("### Gráficos comparativos (energia e vazão)")
    comp = pd.DataFrame({"timestamp": df_sim["timestamp"]}).set_index("timestamp")

    if "energia_kwh" in df_sim.columns:
        comp["energia_base_kwh"] = pd.to_numeric(df_sim["energia_kwh"], errors="coerce").fillna(0.0)
    if "energia_otim_kwh" in df_sim.columns:
        comp["energia_otim_kwh"] = pd.to_numeric(df_sim["energia_otim_kwh"], errors="coerce").fillna(0.0)
    if "vazao_m3h" in df_sim.columns:
        comp["vazao_base_m3h"] = pd.to_numeric(df_sim["vazao_m3h"], errors="coerce").fillna(0.0)
    if "vazao_otim_m3h" in df_sim.columns:
        comp["vazao_otim_m3h"] = pd.to_numeric(df_sim["vazao_otim_m3h"], errors="coerce").fillna(0.0)

    cols_e = [c for c in ["energia_base_kwh", "energia_otim_kwh"] if c in comp.columns]
    cols_v = [c for c in ["vazao_base_m3h", "vazao_otim_m3h"] if c in comp.columns]

    if cols_e:
        st.line_chart(comp[cols_e])
    if cols_v:
        st.line_chart(comp[cols_v])

    st.markdown("### Evidência para o TCC (CSV)")
    if st.button("Salvar resultados desta simulação"):
        path = salvar_resultados_piloto(
            res=res,
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
            username=login_user,
            evaluator=evaluator_name,
            uc_id=str(uc_id),
        )
        st.success(f"Resultados salvos em: {path}")

    if os.path.exists(RES_LOG_PATH):
        st.caption("Histórico salvo (últimas 20 linhas):")
        hist = pd.read_csv(RES_LOG_PATH)
        st.dataframe(hist.tail(20), use_container_width=True)


# =========================================================
# TAB 4 — MODELO ECONÔMICO (TCC)
# =========================================================
with tabs[4]:
    st.subheader("Modelo Econômico (TCC) — Economia → Cenários → VPL → Precificação → Unit Economics")

    
    with st.expander("(Unit Economics + Break-even + Captura de Valor)", expanded=False):
        st.markdown(
            r"""
### Análise de Unit Economics e Ponto de Equilíbrio

A sustentabilidade financeira do AgroData foi avaliada por meio da análise de *unit economics*, considerando a relação entre preço, custos variáveis por cliente e custos fixos operacionais.

O modelo adota uma estratégia de **precificação baseada em captura de valor**, na qual o preço mensal do serviço corresponde a um percentual da economia financeira gerada ao produtor rural. Formalmente:

\[
P = \min\left(\max(E_{mensal}\cdot \beta, P_{mín}), P_{máx}\right)
\]

Onde:
- \(P\) = preço mensal do serviço  
- \(E_{mensal}\) = economia mensal estimada ao cliente  
- \(\beta\) = percentual de captura de valor  
- \(P_{mín}\), \(P_{máx}\) = limites estratégicos do plano  

A partir do preço definido, calcula-se a **margem de contribuição por cliente**:

\[
MC = P - CV - O
\]

Onde:
- \(MC\) = margem de contribuição mensal por cliente  
- \(CV\) = custo variável por cliente  
- \(O\) = overhead proporcional (impostos e despesas administrativas)  

O **ponto de equilíbrio (Break-even)** é determinado pela razão entre o custo fixo mensal total (OPEX) e a margem de contribuição:

\[
BE = \frac{OPEX}{MC}
\]

O resultado indica o número mínimo de clientes necessários para que a operação cubra integralmente seus custos fixos.

Adicionalmente, o **payback do investimento inicial (CAPEX)** é calculado considerando a margem total mensal obtida no cenário de clientes ativos:

\[
Payback = \frac{CAPEX}{MC \cdot N}
\]

Essa abordagem permite demonstrar sustentabilidade financeira, escalabilidade e coerência entre valor entregue e rentabilidade do negócio.
            """.strip()
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

    economia_periodo_rs = float(res["economia_rs"])
    economia_periodo_kwh = float(res["economia_kwh"])
    custo_base_periodo_rs = float(res["custo_base"])

    if periodo == "Últimas 24h":
        fator_mes = 30
        dias_periodo = 1
    elif periodo == "Últimos 3 dias":
        fator_mes = 10
        dias_periodo = 3
    elif periodo == "Últimos 7 dias":
        fator_mes = (30 / 7)
        dias_periodo = 7
    elif periodo == "Últimos 30 dias":
        fator_mes = 1
        dias_periodo = 30
    else:
        dias_periodo = max(1, int((df_f["timestamp"].max() - df_f["timestamp"].min()).total_seconds() / 86400))
        fator_mes = 30 / dias_periodo

    economia_mensal_rs = economia_periodo_rs * fator_mes
    economia_mensal_kwh = economia_periodo_kwh * fator_mes

    economia_diaria_rs = economia_periodo_rs / max(1, dias_periodo)
    economia_safra_rs = economia_diaria_rs * duracao_safra_dias

    preco_sugerido = calc_precificacao_por_valor(
        economia_rs_mensal=economia_mensal_rs,
        pct_captura=pct_captura,
        piso=piso_plano,
        teto=teto_plano,
    )

    met_cliente = calc_roi_payback_cliente(
        preco_mensal=preco_sugerido,
        economia_rs_mensal=economia_mensal_rs,
        investimento_inicial=investimento_inicial_cliente,
    )

    st.markdown("### Economia estimada (a partir do período selecionado)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Economia (R$/mês)", fmt_br_money(economia_mensal_rs))
    c2.metric("Economia (kWh/mês)", fmt_br_number(economia_mensal_kwh, 1))
    c3.metric("Economia na safra (R$)", fmt_br_money(economia_safra_rs))
    c4.metric("Tarifa usada", "Horo-sazonal" if usar_tarifa_variavel else f"Fixa ({tarifa_kwh_fixa:.2f})")

    st.markdown("### Precificação + ROI/Payback (cliente)")
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Preço sugerido (R$/mês)", fmt_br_money(preco_sugerido))
    p2.metric("ROI mensal (cliente)", f"{met_cliente['roi_mensal']*100:.1f}%" if met_cliente["roi_mensal"] is not None else "—")
    p3.metric("Ganho líquido (R$/mês)", fmt_br_money(met_cliente["ganho_liquido"]) if met_cliente["ganho_liquido"] is not None else "—")
    p4.metric("Payback (cliente) (meses)", fmt_br_number(met_cliente["payback_meses"], 1) if met_cliente["payback_meses"] is not None else "—")

    st.divider()

    st.markdown("### Unit Economics + Break-even (projeto)")
    ue = unit_economics_break_even(
        preco_mensal=preco_sugerido,
        custo_variavel_por_cliente=cv_por_cliente,
        overhead_pct=overhead_pct,
        opex_mensal=opex_mensal,
        n_clientes=int(n_clientes),
        capex_inicial=capex_inicial_projeto,
    )

    u1, u2, u3, u4 = st.columns(4)
    u1.metric("Margem contribuição / cliente (MC)", fmt_br_money(ue["margem_contribuicao"]))
    u2.metric("Overhead (R$/cliente)", fmt_br_money(ue["overhead_rs"]))
    u3.metric("Break-even (clientes)", fmt_br_number(ue["break_even_clientes"], 1) if ue["break_even_clientes"] is not None else "—")
    u4.metric("Lucro mensal (N clientes)", fmt_br_money(ue["lucro_mensal_projeto"]))

    u5, u6, u7, u8 = st.columns(4)
    u5.metric("OPEX mensal", fmt_br_money(ue["opex_mensal"]))
    u6.metric("Clientes ativos (N)", f"{ue['n_clientes']}")
    u7.metric("CAPEX inicial (projeto)", fmt_br_money(ue["capex_inicial"]))
    u8.metric("Payback CAPEX (meses)", fmt_br_number(ue["payback_capex_meses"], 1) if ue["payback_capex_meses"] is not None else "—")

    st.caption("Interpretação: break-even indica quantos clientes são necessários para cobrir OPEX; payback CAPEX usa a margem total MC·N.")

    st.divider()

    st.markdown("### Cenários de economia (5% / 10% / 15%) sobre o custo base do período")
    df_cen = calcular_cenarios_economia(custo_base_periodo=custo_base_periodo_rs, alpha=alpha_receita)

    df_cen_show = df_cen.copy()
    df_cen_show["r (%)"] = (df_cen_show["r"] * 100).round(1)
    df_cen_show["Economia bruta (R$)"] = df_cen_show["economia_bruta_rs"].round(2)
    df_cen_show["Receita serviço (R$)"] = df_cen_show["receita_servico_rs"].round(2)
    df_cen_show["Economia líquida produtor (R$)"] = df_cen_show["economia_liquida_produtor_rs"].round(2)
    st.dataframe(df_cen_show[["cenário", "r (%)", "Economia bruta (R$)", "Receita serviço (R$)", "Economia líquida produtor (R$)"]], use_container_width=True)

    fig_c = plot_cenarios_bar(df_cen, title=f"Cenários — base do período ({periodo})")
    png_c = fig_to_png_bytes(fig_c)
    st.image(png_c, caption="Gráfico — Cenários (economia vs receita)", use_container_width=True)
    st.download_button("Baixar gráfico (PNG) — Cenários", data=png_c, file_name="grafico_cenarios.png", mime="image/png")

    st.divider()

    st.markdown("### VPL (Valor Presente Líquido) — horizonte e taxa definidos")
    custo_base_mensal = custo_base_periodo_rs * fator_mes
    custo_base_anual = custo_base_mensal * 12

    vpl_rows = []
    for _, row in df_cen.iterrows():
        nome = row["cenário"]
        r = float(row["r"])
        econ_bruta_anual = custo_base_anual * r
        receita_anual = econ_bruta_anual * alpha_receita
        econ_liq_anual = econ_bruta_anual - receita_anual

        vpl_proj = npv_anuidades(receita_anual, taxa_desconto, int(horizonte_anos))
        vpl_prod = npv_anuidades(econ_liq_anual, taxa_desconto, int(horizonte_anos))

        vpl_rows.append({
            "cenário": nome,
            "r": r,
            "cf_receita_anual_rs": receita_anual,
            "cf_econ_liq_anual_rs": econ_liq_anual,
            "vpl_projeto_rs": vpl_proj,
            "vpl_produtor_rs": vpl_prod,
        })

    df_vpl = pd.DataFrame(vpl_rows)
    df_vpl_show = df_vpl.copy()
    df_vpl_show["r (%)"] = (df_vpl_show["r"] * 100).round(1)
    df_vpl_show["Receita anual (R$)"] = df_vpl_show["cf_receita_anual_rs"].round(2)
    df_vpl_show["Economia líquida anual (R$)"] = df_vpl_show["cf_econ_liq_anual_rs"].round(2)
    df_vpl_show["VPL projeto (R$)"] = df_vpl_show["vpl_projeto_rs"].round(2)
    df_vpl_show["VPL produtor (R$)"] = df_vpl_show["vpl_produtor_rs"].round(2)

    st.dataframe(
        df_vpl_show[["cenário", "r (%)", "Receita anual (R$)", "Economia líquida anual (R$)", "VPL projeto (R$)", "VPL produtor (R$)"]],
        use_container_width=True
    )

    fig_v = plot_vpl_bar(df_vpl, title=f"VPL em {int(horizonte_anos)} anos — taxa {taxa_desconto*100:.0f}% a.a.")
    png_v = fig_to_png_bytes(fig_v)
    st.image(png_v, caption="Gráfico — VPL (Projeto vs Produtor)", use_container_width=True)
    st.download_button("Baixar gráfico (PNG) — VPL", data=png_v, file_name="grafico_vpl.png", mime="image/png")

    st.divider()

    st.markdown("### Redução variável por fase fenológica (r por fase) — estimativa de economia bruta")
    r_por_fase = {"Vegetativa": r_veg, "Reprodutiva": r_rep, "Emborrachamento/Floração": r_flo, "Maturação": r_mat}

    df_fen = pd.DataFrame([
        {"fase": "Vegetativa", "r": r_veg},
        {"fase": "Reprodutiva", "r": r_rep},
        {"fase": "Emborrachamento/Floração", "r": r_flo},
        {"fase": "Maturação", "r": r_mat},
    ])
    df_fen["economia_bruta_rs"] = custo_base_periodo_rs * df_fen["r"]
    df_fen["r_pct"] = (df_fen["r"] * 100).round(1)

    st.info(
        f"Fase selecionada no painel: **{fase}** → r = **{r_por_fase.get(fase, 0.0)*100:.1f}%**. "
        "Use isso como conexão simples entre fenologia e o modelo econômico."
    )

    df_fen_show = df_fen.copy()
    df_fen_show["Economia bruta (R$)"] = df_fen_show["economia_bruta_rs"].round(2)
    st.dataframe(df_fen_show[["fase", "r_pct", "Economia bruta (R$)"]].rename(columns={"r_pct": "r (%)"}), use_container_width=True)

    fig_f = plot_fenologia_bar(df_fen, title=f"Economia bruta estimada por fase — base do período ({periodo})")
    png_f = fig_to_png_bytes(fig_f)
    st.image(png_f, caption="Gráfico — Economia por fase fenológica", use_container_width=True)
    st.download_button("Baixar gráfico (PNG) — Fenologia", data=png_f, file_name="grafico_fenologia.png", mime="image/png")

    st.divider()

    st.markdown("### Evidência exportável (para anexar no TCC)")
    evidencia = {
        "timestamp_execucao": _now_iso(),
        "uc_id": str(uc_id),
        "periodo": periodo,
        "usar_tarifa_variavel": bool(usar_tarifa_variavel),
        "tarifa_fixa_rs_kwh": float(tarifa_kwh_fixa),
        "tarifa_fp": float(tarifa_fp),
        "tarifa_ponta": float(tarifa_ponta),
        "ponta_inicio": int(ponta_inicio),
        "ponta_fim": int(ponta_fim),
        "lamina_max_cm": float(lamina_max),
        "chuva_min_24h_mm": float(chuva_min_mm),
        "custo_base_periodo_rs": float(custo_base_periodo_rs),
        "economia_periodo_rs": float(economia_periodo_rs),
        "economia_mensal_rs_estim": float(economia_mensal_rs),
        "economia_safra_dias": int(duracao_safra_dias),
        "economia_safra_rs_estim": float(economia_safra_rs),
        "pct_captura_value_based": float(pct_captura),
        "alpha_receita_projeto": float(alpha_receita),
        "preco_sugerido_rs_mensal": float(preco_sugerido),
        "investimento_inicial_cliente_rs": float(investimento_inicial_cliente),
        "roi_mensal_cliente": met_cliente["roi_mensal"],
        "payback_cliente_meses": met_cliente["payback_meses"],
        "taxa_desconto_aa": float(taxa_desconto),
        "horizonte_anos": int(horizonte_anos),
        "r_fase_vegetativa": float(r_veg),
        "r_fase_reprodutiva": float(r_rep),
        "r_fase_emborrachamento_floracao": float(r_flo),
        "r_fase_maturacao": float(r_mat),
        # Unit economics
        "opex_mensal_rs": float(opex_mensal),
        "cv_por_cliente_rs": float(cv_por_cliente),
        "overhead_pct": float(overhead_pct),
        "n_clientes": int(n_clientes),
        "capex_inicial_projeto_rs": float(capex_inicial_projeto),
        "mc_por_cliente_rs": float(ue["margem_contribuicao"]),
        "break_even_clientes": ue["break_even_clientes"],
        "lucro_mensal_projeto_rs": float(ue["lucro_mensal_projeto"]),
        "payback_capex_meses": ue["payback_capex_meses"],
        "app_version": APP_VERSION,
        "run_id": RUN_ID,
        "user": login_user,
        "evaluator": evaluator_name,
    }
    st.json(evidencia)

    if st.button("Salvar evidência econômica (CSV)"):
        _safe_mkdir(LOG_DIR)
        df_row = pd.DataFrame([evidencia])
        if not os.path.exists(ECO_LOG_PATH):
            df_row.to_csv(ECO_LOG_PATH, index=False, encoding="utf-8")
        else:
            df_row.to_csv(ECO_LOG_PATH, mode="a", header=False, index=False, encoding="utf-8")
        st.success(f"Modelo econômico salvo em: {ECO_LOG_PATH}")

    if st.button("Salvar tabela de cenários/VPL/fenologia (CSV)"):
        ts = _now_iso()
        df_out = df_vpl[["cenário", "r", "vpl_projeto_rs", "vpl_produtor_rs"]].copy()
        df_out["timestamp_execucao"] = ts
        df_out["user"] = login_user
        df_out["evaluator"] = evaluator_name
        df_out["uc_id"] = str(uc_id)
        df_out["periodo"] = periodo
        df_out["usar_tarifa_variavel"] = int(bool(usar_tarifa_variavel))
        df_out["tarifa_fixa_rs_kwh"] = float(tarifa_kwh_fixa)
        df_out["tarifa_fp"] = float(tarifa_fp)
        df_out["tarifa_ponta"] = float(tarifa_ponta)
        df_out["ponta_inicio"] = int(ponta_inicio)
        df_out["ponta_fim"] = int(ponta_fim)
        df_out["custo_base_periodo_rs"] = float(custo_base_periodo_rs)
        df_out["alpha_receita"] = float(alpha_receita)
        df_out["taxa_desconto"] = float(taxa_desconto)
        df_out["horizonte_anos"] = int(horizonte_anos)
        df_out["r_fase_veg"] = float(r_veg)
        df_out["r_fase_rep"] = float(r_rep)
        df_out["r_fase_flo"] = float(r_flo)
        df_out["r_fase_mat"] = float(r_mat)
        df_out["app_version"] = APP_VERSION
        df_out["run_id"] = RUN_ID

        path = salvar_cenarios_vpl_fenologia(df_out)
        st.success(f"Tabela salva em: {path}")

    if os.path.exists(ECO_LOG_PATH):
        st.caption("Histórico (modelo_economico.csv) — últimas 20 linhas:")
        hist_eco = pd.read_csv(ECO_LOG_PATH)
        st.dataframe(hist_eco.tail(20), use_container_width=True)


# =========================================================
# TAB 5 — MODELO MATEMÁTICO (equações)
# =========================================================
with tabs[5]:
    st.subheader("Modelo Matemático Formal (equações) — versão para o TCC")
    st.markdown(
        r"""
### Variáveis
- \(E\): energia consumida (kWh)  
- \(T\): tarifa (R$/kWh)  
- \(C\): custo (R$)  
- \(r\): redução percentual de energia (0–1)  
- \(\alpha\): participação/receita do projeto sobre a economia bruta (0–1)  
- \(S\): economia bruta (R$)  
- \(R\): receita do projeto (R$)  
- \(S_{liq}\): economia líquida do produtor (R$)  
- \(i\): taxa de desconto anual  
- \(n\): horizonte (anos)

### Economia e precificação
1. Custo do período:
\[
C = E \cdot T
\]

2. Economia bruta:
\[
S = C \cdot r
\]

3. Receita do projeto:
\[
R = \alpha \cdot S
\]

4. Economia líquida do produtor:
\[
S_{liq} = S - R = S(1-\alpha)
\]

### VPL (anuidade constante)
Para um fluxo anual constante \(CF\):
\[
VPL = \sum_{t=1}^{n}\frac{CF}{(1+i)^t}
\]

No protótipo:
- \(CF_{proj} = R_{anual}\)
- \(CF_{prod} = S_{liq,anual}\)

### Unit economics e break-even
Precificação por valor:
\[
P = \min(\max(E_{mensal}\cdot \beta, P_{mín}), P_{máx})
\]

Margem de contribuição:
\[
MC = P - CV - O
\]

Ponto de equilíbrio:
\[
BE = \frac{OPEX}{MC}
\]

Payback CAPEX:
\[
Payback = \frac{CAPEX}{MC \cdot N}
\]
        """.strip()
    )


# =========================================================
# TAB 6 — SEGMENTAÇÃO
# =========================================================
with tabs[6]:
    st.subheader("Segmentação (TCC) — Priorização do Cliente Inicial")
    st.markdown(
        """
**Segmento prioritário (beachhead):**
- Produtores de arroz irrigado no RS;
- Médio/grande porte;
- Bombeamento elétrico + medições (energia/nível/vazão);
- Preferencialmente com SCADA/sensores já instalados;
- Alto custo energético.
        """.strip()
    )

    st.markdown("### Checklist de aderência do cliente (simulação)")
    area_ha = st.number_input("Área irrigada (ha)", min_value=0, value=300, step=50)
    tem_scada = st.checkbox("Possui SCADA / automação?", value=True)
    tem_medicao_energia = st.checkbox("Possui medição de energia (kWh)?", value=True)
    custo_mensal = st.number_input("Custo mensal de energia (R$)", min_value=0.0, value=50000.0, step=5000.0)

    score = 0
    score += 1 if area_ha >= 300 else 0
    score += 1 if tem_scada else 0
    score += 1 if tem_medicao_energia else 0
    score += 1 if custo_mensal >= 50000 else 0

    st.metric("Aderência ao segmento inicial (0–4)", score)

    if score >= 3:
        st.success("Cliente bem alinhado ao segmento inicial — maior chance de ROI rápido e implantação simples.")
    else:
        st.warning("Cliente fora do foco inicial — pode exigir maior customização e reduzir velocidade de escala.")


# =========================================================
# TAB 7 — MVP vs VISÃO FUTURA
# =========================================================
with tabs[7]:
    st.subheader("MVP vs Visão Futura — formato estratégico")
    st.markdown(
        """
### Estratégia (visão em camadas)
**Nível 1 — MVP (piloto, curto prazo):** provar valor (economia + controle), com transparência e auditabilidade.  
**Nível 2 — Produto inicial (médio prazo):** padronizar integração e onboarding; ampliar portfólio de recomendações.  
**Nível 3 — Escala (longo prazo):** IA preditiva/prescritiva + automação e benchmarking regional.

| Dimensão | MVP (piloto) | Próximo estágio (produto) | Visão futura (escala) |
|---|---|---|---|
| Objetivo | Evidenciar economia e controle operacional | Replicação e implantação rápida | Otimização e previsões com IA |
| Tipo de análise | Descritiva/diagnóstica (BI) | Diagnóstico + detecção inteligente | Preditiva/prescritiva (ML) |
| Recomendações | Regras interpretáveis + logs | Regras + priorização + templates | Modelos + otimização multiobjetivo |
| Integração | 1–6 UCs (piloto) | Conectores por fornecedor | APIs clima + mobile + benchmark |
| Evidência | CSVs + RUN_ID | Trilhas de auditoria por cliente | Governança de dados e SLAs |
        """.strip()
    )


# =========================================================
# TAB 8 — ESCALA & GARGALOS
# =========================================================
with tabs[8]:
    st.subheader("Escala & Gargalos Operacionais (TCC)")
    st.markdown(
        """
**Gargalos para escalar:**
- Integração com SCADA heterogêneo (protocolos/dados);
- Padronização de sensores e qualidade do dado;
- Conectividade rural e disponibilidade de rede;
- Treinamento/adoção (mudança cultural);
- Suporte técnico especializado e SLAs.

**Mitigações planejadas:**
- Templates de integração por fornecedor;
- Validação automática de qualidade do dado;
- Interface simples + tutoriais;
- Documentação e onboarding padronizado;
- Priorização de nicho inicial com infraestrutura já instalada.

**Estratégia de execução (sequência sugerida):**
1) Beachhead com produtores com SCADA já instalado (rápido ROI)  
2) Padronizar conectores + checklist de qualidade do dado  
3) Expandir recomendações e relatórios comparativos (benchmark)  
4) Evoluir para modelos preditivos (ET0, demanda hídrica, detecção anomalias)
        """.strip()
    )


# =========================================================
# TAB 9 — QUALIDADE DO DADO (ESCALA)
# =========================================================
with tabs[9]:
    st.subheader("Qualidade do Dado (escala) — validação automática")
    st.caption("Métricas simples e auditáveis para apoiar a escalabilidade (sensores, SCADA e integridade temporal).")

    dq = data_quality_metrics(df_f)
    st.json(dq)

    st.markdown("### Interpretação rápida")
    colA, colB, colC, colD = st.columns(4)
    colA.metric("Registros", dq["n_registros"])
    colB.metric("Missing lâmina (%)", f"{dq['missing_lamina_pct']:.1f}%")
    colC.metric("Missing energia (%)", f"{dq['missing_energia_pct']:.1f}%")
    colD.metric("dt mediana (h)", f"{dq['dt_mediana_h']:.2f}" if np.isfinite(dq["dt_mediana_h"]) else "—")

    st.markdown("### Evidência (CSV)")
    if st.button("Salvar qualidade do dado (CSV)"):
        path = log_data_quality_event(
            username=login_user,
            evaluator=evaluator_name,
            periodo_label=periodo,
            metrics=dq,
            uc_id=str(uc_id),
        )
        st.success(f"Qualidade do dado salva em: {path}")

    if os.path.exists(DQ_LOG_PATH):
        st.caption("Histórico (data_quality_log.csv) — últimas 20 linhas:")
        hist_dq = pd.read_csv(DQ_LOG_PATH)
        st.dataframe(hist_dq.tail(20), use_container_width=True)


# =========================================================
# RODAPÉ TÉCNICO / INSTITUCIONAL (TCC)
# =========================================================
st.divider()

st.markdown(
    f"""
---
### 📌 Encerramento do Protótipo Acadêmico

Este sistema demonstra a aplicação integrada de:
- **Business Intelligence (BI)** → KPIs e visualização operacional  
- **Data Science** → Tratamento, integração temporal e métricas de eficiência  
- **Inteligência Artificial Explicável** → Regras interpretáveis para suporte à decisão  
- **Modelo Econômico Quantitativo** → Cenários, VPL, precificação por valor, unit economics e break-even  
- **Governança e Auditoria** → Logs com versão e ID de execução  

---

### 🔎 Rastreabilidade da Execução
- **Versão do aplicativo:** `{APP_VERSION}`  
- **ID da execução (RUN_ID):** `{RUN_ID}`  
- **Usuário autenticado:** `{login_user}`  
- **Avaliador:** `{evaluator_name}`  
- **Timestamp da sessão:** `{datetime.now().isoformat(timespec="seconds")}`  

---

### 🧠 Contribuição Acadêmica (Resumo)
O protótipo valida a viabilidade técnica e econômica de um modelo de
monitoramento inteligente da irrigação, demonstrando:
1. Redução simulada de consumo energético  
2. Apoio ao manejo hídrico baseado em fase fenológica  
3. Integração entre eficiência operacional e geração de valor  
4. Estrutura preparada para escalabilidade e modelos preditivos futuros  

---

### 🚀 Próximos Passos (Visão Futura)
- Integração com APIs meteorológicas em tempo real  
- Modelos preditivos de demanda hídrica (Machine Learning)  
- Benchmark entre propriedades (comparação de eficiência)  
- Aplicação mobile para alertas operacionais  
- Engine de recomendação adaptativa  

---

🔐 **Observação:** Este sistema é um protótipo acadêmico desenvolvido para fins de TCC.
Em ambiente produtivo, recomenda-se:
- Autenticação OAuth/SSO  
- Banco de dados dedicado (PostgreSQL/Cloud)  
- Monitoramento de disponibilidade (SLA)  
- Pipeline ETL estruturado  
---
"""
)

st.success("Protótipo executado com sucesso — evidências geradas para fins acadêmicos.")