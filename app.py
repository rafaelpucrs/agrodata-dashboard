# =========================================================
# AgroData — Irrigação (BI + Data Science + IA)
# Protótipo acadêmico (TCC) com:
# ✅ Login + logs (com ID de execução + versão)
# ✅ KPIs + recomendações (regras explicáveis)
# ✅ Baseline vs Otimizado (simulação)
# ✅ Modelo Econômico completo:
#    - Cenários (5% / 10% / 15%)
#    - VPL (5 anos)
#    - Redução variável por fase fenológica
#    - Gráficos (PNG) + download
#    - Evidências (CSV) + download
#
# Melhorias aplicadas (recomendações):
# ✅ Rolling 24h sem desalinhamento
# ✅ Preparado para dados reais com intervalo irregular (dt_horas)
# ✅ Cache de carregamento de dados
# ✅ Funções renomeadas para coerência (cenários ≠ VPL)
# ✅ “Validação automática do dado” (qualidade) + evidências
# ✅ APP_VERSION + RUN_ID (execução) nos logs
# ✅ freq="h" (evita warnings)
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
APP_VERSION = "0.4.0"
APP_TITLE = "AgroData — Irrigação (BI + Data Science + IA)"
APP_SUBTITLE = (
    "Protótipo acadêmico para o TCC: monitoramento operacional da irrigação com indicadores (KPIs), "
    "alertas e recomendações automatizadas de suporte à decisão + modelo econômico."
)

DATA_CSV_PATH = os.path.join("data", "dados_irrigacao.csv")
ARQ_IMG_PATH = os.path.join("data", "arquitetura_irrigacao.png")

LOG_DIR = "logs"
REC_LOG_PATH = os.path.join(LOG_DIR, "recommendations_log.csv")
RES_LOG_PATH = os.path.join(LOG_DIR, "resultados_piloto.csv")
ECO_LOG_PATH = os.path.join(LOG_DIR, "modelo_economico.csv")
ECO_CENARIOS_LOG_PATH = os.path.join(LOG_DIR, "cenarios_vpl_fenologia.csv")
DQ_LOG_PATH = os.path.join(LOG_DIR, "data_quality_log.csv")

# Defaults (seu cenário)
DEFAULT_TARIFA = 0.82
DEFAULT_HORAS_DIA = 20
DEFAULT_TAXA_DESCONTO = 0.12
DEFAULT_HORIZONTE_ANOS = 5
DEFAULT_ALPHA_RECEITA = 0.20  # % receita do projeto em cima da economia bruta


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


def log_access_event(event: str, username: str, evaluator: str):
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, "access_log.csv")

    ts = datetime.now().isoformat(timespec="seconds")
    row = f"{ts},{event},{username},{evaluator},{APP_VERSION},{RUN_ID}\n"

    if not os.path.exists(log_path):
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("timestamp,event,user,evaluator,app_version,run_id\n")

    with open(log_path, "a", encoding="utf-8") as f:
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
):
    os.makedirs(LOG_DIR, exist_ok=True)

    ts = datetime.now().isoformat(timespec="seconds")
    mensagens_txt = " | ".join([str(m).replace("\n", " ").strip() for m in mensagens])

    row = {
        "timestamp": ts,
        "nivel": nivel,
        "mensagens": mensagens_txt,
        "user": username,
        "evaluator": evaluator,
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


def log_data_quality_event(username: str, evaluator: str, periodo_label: str, metrics: dict):
    os.makedirs(LOG_DIR, exist_ok=True)
    ts = datetime.now().isoformat(timespec="seconds")

    row = {
        "timestamp": ts,
        "user": username,
        "evaluator": evaluator,
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
    os.makedirs(LOG_DIR, exist_ok=True)

    removed = []
    for p in [REC_LOG_PATH, RES_LOG_PATH, ECO_LOG_PATH, ECO_CENARIOS_LOG_PATH, DQ_LOG_PATH]:
        if os.path.exists(p):
            os.remove(p)
            removed.append(os.path.basename(p))

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


def compute_energy_and_volume(df: pd.DataFrame):
    """
    Retorna energia_total_kwh, volume_total_m3 usando:
    - energia_kwh: assume energia por amostra (já integrada)
    - se não existir energia_kwh, mas existir potencia_kw: integra potencia_kw * dt_horas
    - volume: integra vazao_m3h * dt_horas (robusto)
    """
    df = add_dt_hours(df)

    # energia
    if "energia_kwh" in df.columns:
        energia_total = float(df["energia_kwh"].sum())
        energia_series = df["energia_kwh"].astype(float)
    elif "potencia_kw" in df.columns:
        energia_series = df["potencia_kw"].astype(float) * df["dt_horas"].astype(float)
        energia_total = float(energia_series.sum())
    else:
        energia_series = pd.Series([0.0] * len(df))
        energia_total = 0.0

    # volume (sempre integra)
    if "vazao_m3h" in df.columns:
        volume_series = df["vazao_m3h"].astype(float) * df["dt_horas"].astype(float)
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
    n = len(df)

    def pct_missing(col):
        if col not in df.columns:
            return 100.0
        return float(df[col].isna().mean() * 100.0)

    metrics = {
        "n_registros": int(n),
        "missing_timestamp_pct": pct_missing("timestamp"),
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
    else:
        metrics["dt_mediana_h"] = np.nan
        metrics["dt_max_h"] = np.nan

    # outliers simples (ex.: lâmina fora da faixa esperada do protótipo)
    if "lamina_cm" in df.columns:
        lam = df["lamina_cm"].astype(float)
        metrics["lamina_outlier_pct"] = float(((lam < 0) | (lam > 25)).mean() * 100.0)
    else:
        metrics["lamina_outlier_pct"] = 0.0

    return metrics


# =========================================================
# FUNÇÕES — geração e carga (com cache)
# =========================================================
def gerar_dados_exemplo(n_horas=240, seed=42):
    rng = np.random.default_rng(seed)
    now = datetime.now().replace(minute=0, second=0, microsecond=0)
    ts = pd.date_range(end=now, periods=n_horas, freq="h")  # freq="h" (recomendado)

    chuva = np.zeros(n_horas)
    for _ in range(10):
        idx = int(rng.integers(0, n_horas))
        dur = int(rng.integers(2, 8))
        chuva[idx: idx + dur] += float(rng.uniform(1, 6))

    bomba_ligada = (rng.random(n_horas) > 0.35).astype(int)

    vazao = np.where(bomba_ligada == 1, rng.normal(75, 12, n_horas), rng.normal(5, 2, n_horas))
    vazao = np.clip(vazao, 0, None)

    energia = np.where(bomba_ligada == 1, rng.normal(55, 10, n_horas), rng.normal(2, 1, n_horas))
    energia = np.clip(energia, 0, None)

    lamina = np.zeros(n_horas)
    lamina[0] = 7.5
    for i in range(1, n_horas):
        ganho_irrig = (vazao[i] / 1200)
        ganho_chuva = (chuva[i] / 20)
        perda = rng.normal(0.02, 0.03)
        lamina[i] = lamina[i - 1] + ganho_irrig + ganho_chuva - perda

    lamina = np.clip(lamina, 4.5, 12.0)

    return pd.DataFrame(
        {
            "timestamp": ts,
            "lamina_cm": lamina,
            "vazao_m3h": vazao,
            "energia_kwh": energia,
            "chuva_mm": chuva,
            "bomba_ligada": bomba_ligada,
        }
    )


@st.cache_data(show_spinner=False)
def carregar_dados_cached(csv_path: str, file_sig: str):
    """
    Cache com assinatura (file_sig) para recarregar quando arquivo muda.
    """
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)

        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        else:
            for c in df.columns:
                if "data" in c.lower() or "hora" in c.lower():
                    df["timestamp"] = pd.to_datetime(df[c], errors="coerce")
                    break

        df = ensure_datetime_sorted(df)
        return df

    return ensure_datetime_sorted(gerar_dados_exemplo())


def carregar_dados():
    file_sig = "no_file"
    if os.path.exists(DATA_CSV_PATH):
        file_sig = str(os.path.getmtime(DATA_CSV_PATH))
    return carregar_dados_cached(DATA_CSV_PATH, file_sig)


# =========================================================
# KPIs e recomendação (IA explicável)
# =========================================================
def kpis_basicos(df: pd.DataFrame):
    df = ensure_datetime_sorted(df)
    energia_total, volume_total, _, _, df_dt = compute_energy_and_volume(df)

    lamina_media = float(df_dt["lamina_cm"].mean()) if "lamina_cm" in df_dt.columns else np.nan
    horas_bomba = int(df_dt["bomba_ligada"].sum()) if "bomba_ligada" in df_dt.columns else 0

    ult_24h = df_dt[df_dt["timestamp"] >= (df_dt["timestamp"].max() - pd.Timedelta(hours=24))]
    chuva_24h = float(ult_24h["chuva_mm"].sum()) if "chuva_mm" in df_dt.columns else 0.0

    eficiencia = (energia_total / volume_total) if volume_total > 0 else None

    return {
        "lamina_media": lamina_media,
        "total_energia": energia_total,
        "total_volume": volume_total,
        "eficiencia": eficiencia,
        "horas_bomba": horas_bomba,
        "chuva_24h": chuva_24h,
    }


def recomendacao_ia(df, fase: str, manejo: str, tipo_solo: str, risco_frio: bool, dias_pos_floracao: int):
    df = ensure_datetime_sorted(df)
    agora = df["timestamp"].max()

    ult_6h = df[df["timestamp"] >= (agora - pd.Timedelta(hours=6))]
    ult_24h = df[df["timestamp"] >= (agora - pd.Timedelta(hours=24))]

    chuva_24h = float(ult_24h["chuva_mm"].sum()) if "chuva_mm" in df.columns else 0.0
    lamina_atual = float(df.iloc[-1]["lamina_cm"]) if "lamina_cm" in df.columns else np.nan
    bomba_atual = int(df.iloc[-1]["bomba_ligada"]) if "bomba_ligada" in df.columns else 0

    # 6h (robusto com dt)
    e6, v6, _, _, ult_6h_dt = compute_energy_and_volume(ult_6h) if len(ult_6h) else (0.0, 0.0, None, None, ult_6h)
    eficiencia_6h = (e6 / v6) if v6 > 0 else None
    bomba_horas_6h = int(ult_6h_dt["bomba_ligada"].sum()) if "bomba_ligada" in ult_6h_dt.columns else 0

    # 24h
    e24, v24, _, _, ult_24h_dt = compute_energy_and_volume(ult_24h) if len(ult_24h) else (0.0, 0.0, None, None, ult_24h)
    eficiencia_24h = (e24 / v24) if v24 > 0 else None
    bomba_horas_24h = int(ult_24h_dt["bomba_ligada"].sum()) if "bomba_ligada" in ult_24h_dt.columns else 0

    lamina_min_24h = float(ult_24h["lamina_cm"].min()) if len(ult_24h) and "lamina_cm" in ult_24h.columns else np.nan
    lamina_max_24h = float(ult_24h["lamina_cm"].max()) if len(ult_24h) and "lamina_cm" in ult_24h.columns else np.nan

    # baseline eficiência (histórico com bomba ligada)
    base_ef = None
    if "bomba_ligada" in df.columns:
        df_ligada = df[df["bomba_ligada"] == 1]
        if len(df_ligada) > 10:
            eb, vb, *_ = compute_energy_and_volume(df_ligada)
            if vb > 0:
                base_ef = float(eb / vb)

    mensagens = []
    nivel = "info"

    # 1) chuva como crédito hídrico
    if chuva_24h >= 12:
        mensagens.append(
            f"Chuva 24h = {chuva_24h:.1f} mm (≈ demanda diária média). "
            "Recomendação: avaliar reduzir/adiar bombeamento nas próximas 12–24h."
        )
        nivel = "warning"

    if chuva_24h >= 24:
        mensagens.append(
            f"Chuva 24h alta ({chuva_24h:.1f} mm). "
            "Recomendação: suspender temporariamente o bombeamento e monitorar lâmina para evitar excesso."
        )
        nivel = "warning"

    # 2) alvos por fase / risco de frio
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

    # alertas por lâmina
    if np.isfinite(lamina_atual):
        if lamina_atual > 10.0 and not (risco_frio and fase == "Emborrachamento/Floração"):
            mensagens.append(
                f"Alerta: lâmina alta ({lamina_atual:.1f} cm). "
                "Valores >10 cm podem aumentar perdas e risco de acamamento."
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
                f"Ação: lâmina abaixo do alvo da fase {fase} ({lamina_atual:.1f} < {alvo_min:.1f} cm). "
                "Priorizar reposição."
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

    # 3) manejo intermitente (ON/OFF)
    if manejo.startswith("Intermitente") and fase != "Maturação" and np.isfinite(lamina_atual):
        on_threshold = max(2.5, alvo_min)
        off_threshold = min(8.0, max(alvo_max, 7.0))

        if lamina_atual <= on_threshold:
            mensagens.append(
                f"Manejo intermitente: sugerir RETOMAR irrigação (lâmina {lamina_atual:.1f} ≤ {on_threshold:.1f} cm)."
            )
            nivel = "error"
        elif lamina_atual >= off_threshold:
            mensagens.append(
                f"Manejo intermitente: sugerir PAUSAR bombeamento (lâmina {lamina_atual:.1f} ≥ {off_threshold:.1f} cm)."
            )
            nivel = "warning" if nivel != "error" else nivel
        else:
            mensagens.append(f"Manejo intermitente: faixa operacional {on_threshold:.1f}–{off_threshold:.1f} cm. Manter.")

    # 4) supressão na maturação
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
            mensagens.append(
                "Maturação: para supressão, use floração plena e estágio do grão (ex.: pastoso predominante) como referência."
            )

    # 5) eficiência energética (desvio do baseline)
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
# RESULTADOS — baseline vs otimizado
# =========================================================
def aplicar_otimizacao_regras(df, lamina_max=9.5, chuva_min_mm=10.0):
    """
    Simula cenário otimizado simples:
    - Desliga a bomba quando chuva 24h >= chuva_min_mm OU lâmina >= lamina_max.
    Rolling 24h corrigido (sem desalinhamento).
    """
    df = ensure_datetime_sorted(df)

    if "chuva_mm" in df.columns:
        s = df.set_index("timestamp")["chuva_mm"].rolling("24h").sum()
        df["chuva_24h"] = s.to_numpy()  # <-- sem reset_index(drop=True)
    else:
        df["chuva_24h"] = 0.0

    cond_desliga = (df["chuva_24h"] >= float(chuva_min_mm))
    if "lamina_cm" in df.columns:
        cond_desliga = cond_desliga | (df["lamina_cm"] >= float(lamina_max))

    df["bomba_otim"] = np.where(cond_desliga, 0, df.get("bomba_ligada", 0))

    # energia e vazão otimizadas (se bomba desligada => 0)
    if "energia_kwh" in df.columns:
        df["energia_otim_kwh"] = np.where(df["bomba_otim"] == 1, df["energia_kwh"].astype(float), 0.0)
    elif "potencia_kw" in df.columns:
        # energia derivada da potência e dt (depois integraremos)
        df["potencia_otim_kw"] = np.where(df["bomba_otim"] == 1, df["potencia_kw"].astype(float), 0.0)
    else:
        df["energia_otim_kwh"] = 0.0

    if "vazao_m3h" in df.columns:
        df["vazao_otim_m3h"] = np.where(df["bomba_otim"] == 1, df["vazao_m3h"].astype(float), 0.0)
    else:
        df["vazao_otim_m3h"] = 0.0

    return df


def comparar_cenarios(df, tarifa_kwh=DEFAULT_TARIFA):
    """
    Cálculo robusto:
    - volume_base = integral(vazao_m3h * dt_horas)
    - energia_base = soma(energia_kwh) OU integral(potencia_kw * dt_horas)
    Idem para o cenário otimizado.
    """
    df = ensure_datetime_sorted(df)

    # base
    energia_base, volume_base, _, _, df_dt = compute_energy_and_volume(df)
    horas_bomba_base = int(df_dt["bomba_ligada"].sum()) if "bomba_ligada" in df_dt.columns else 0
    ef_base = (energia_base / volume_base) if volume_base > 0 else np.nan
    custo_base = energia_base * float(tarifa_kwh)

    # otimizado (energia/vazão otimizadas)
    df_ot = df_dt.copy()
    df_ot = add_dt_hours(df_ot)

    # energia otimizada
    if "energia_otim_kwh" in df_ot.columns:
        energia_otim = float(df_ot["energia_otim_kwh"].sum())
    elif "potencia_otim_kw" in df_ot.columns:
        energia_otim = float((df_ot["potencia_otim_kw"].astype(float) * df_ot["dt_horas"].astype(float)).sum())
    else:
        energia_otim = 0.0

    # volume otimizado
    if "vazao_otim_m3h" in df_ot.columns:
        volume_otim = float((df_ot["vazao_otim_m3h"].astype(float) * df_ot["dt_horas"].astype(float)).sum())
    else:
        volume_otim = 0.0

    horas_bomba_otim = int(df_ot["bomba_otim"].sum()) if "bomba_otim" in df_ot.columns else 0
    ef_otim = (energia_otim / volume_otim) if volume_otim > 0 else np.nan
    custo_otim = energia_otim * float(tarifa_kwh)

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
    }


def salvar_resultados_piloto(res: dict, periodo_label: str, lamina_max: float, chuva_min_mm: float, tarifa_kwh: float, username: str, evaluator: str):
    os.makedirs(LOG_DIR, exist_ok=True)
    ts = datetime.now().isoformat(timespec="seconds")

    row = {
        "timestamp_execucao": ts,
        "periodo": periodo_label,
        "lamina_max_cm": float(lamina_max),
        "chuva_min_24h_mm": float(chuva_min_mm),
        "tarifa_rs_kwh": float(tarifa_kwh),
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
    }

    df_row = pd.DataFrame([row])
    if not os.path.exists(RES_LOG_PATH):
        df_row.to_csv(RES_LOG_PATH, index=False, encoding="utf-8")
    else:
        df_row.to_csv(RES_LOG_PATH, mode="a", header=False, index=False, encoding="utf-8")

    return RES_LOG_PATH


# =========================================================
# MODELO ECONÔMICO
# =========================================================
def calc_precificacao_por_valor(economia_rs_mensal: float, pct_captura: float = 0.15, piso: float = 1200.0, teto: float = 3800.0):
    if economia_rs_mensal is None or economia_rs_mensal <= 0:
        return 0.0
    preco = economia_rs_mensal * float(pct_captura)
    return float(min(max(preco, piso), teto))


def calc_roi_payback(preco_mensal: float, economia_rs_mensal: float, investimento_inicial: float):
    """
    ROI do CLIENTE sobre a mensalidade do serviço:
    ROI = (economia - preço) / preço
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
    """
    Apenas cenários de economia/receita (não calcula VPL aqui).
    """
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
    os.makedirs(LOG_DIR, exist_ok=True)
    if not os.path.exists(ECO_CENARIOS_LOG_PATH):
        df_row.to_csv(ECO_CENARIOS_LOG_PATH, index=False, encoding="utf-8")
    else:
        df_row.to_csv(ECO_CENARIOS_LOG_PATH, mode="a", header=False, index=False, encoding="utf-8")
    return ECO_CENARIOS_LOG_PATH


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

    # lembrete de segurança (protótipo)
    user_ok, pass_ok, _ = get_settings()
    if user_ok == "admin" and pass_ok == "admin":
        st.warning("⚠️ Protótipo com credencial padrão (admin/admin). Em produção, use st.secrets/OAuth.")

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
df = carregar_dados()

st.sidebar.header("Filtros")
max_data = df["timestamp"].max()

periodo = st.sidebar.selectbox(
    "Período",
    options=["Últimas 24h", "Últimos 3 dias", "Últimos 7 dias", "Tudo"],
    index=2,
)

# parâmetros agronômicos
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

# parâmetros da simulação
st.sidebar.header("Parâmetros (simulação)")
tarifa_kwh = st.sidebar.number_input("Tarifa de energia (R$/kWh)", min_value=0.0, value=float(DEFAULT_TARIFA), step=0.01)
lamina_max = st.sidebar.slider("Lâmina máxima (cm)", min_value=7.0, max_value=20.0, value=10.0, step=0.5)
chuva_min_mm = st.sidebar.slider("Chuva 24h (mm) para reduzir/adiar", min_value=0.0, max_value=50.0, value=12.0, step=1.0)

# modelo econômico
st.sidebar.header("Modelo econômico (TCC)")
pct_captura = st.sidebar.slider("Captura de valor (% da economia)", 5, 30, 20, 1) / 100.0
alpha_receita = st.sidebar.slider("Receita do projeto (% da economia bruta)", 5, 30, int(DEFAULT_ALPHA_RECEITA * 100), 1) / 100.0

investimento_inicial = st.sidebar.number_input("Investimento inicial estimado (R$)", min_value=0.0, value=12400.0, step=100.0)
horas_dia_ref = st.sidebar.number_input("Horas/dia referência (econômico)", min_value=0, max_value=24, value=DEFAULT_HORAS_DIA, step=1)

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

# evidências / logs
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
    (REC_LOG_PATH, "Baixar recommendations_log.csv"),
    (RES_LOG_PATH, "Baixar resultados_piloto.csv"),
    (ECO_LOG_PATH, "Baixar modelo_economico.csv"),
    (ECO_CENARIOS_LOG_PATH, "Baixar cenarios_vpl_fenologia.csv"),
    (DQ_LOG_PATH, "Baixar data_quality_log.csv"),
]:
    if os.path.exists(pth):
        with open(pth, "rb") as f:
            st.sidebar.download_button(label, data=f, file_name=os.path.basename(pth), mime="text/csv")

# filtra df por período
if periodo == "Últimas 24h":
    df_f = df[df["timestamp"] >= (max_data - pd.Timedelta(hours=24))]
elif periodo == "Últimos 3 dias":
    df_f = df[df["timestamp"] >= (max_data - pd.Timedelta(days=3))]
elif periodo == "Últimos 7 dias":
    df_f = df[df["timestamp"] >= (max_data - pd.Timedelta(days=7))]
else:
    df_f = df.copy()

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
    c2.metric("Energia total (kWh)", f"{k['total_energia']:.1f}")
    c3.metric("Volume total (m³)", f"{k['total_volume']:.1f}")
    c4.metric("Eficiência (kWh/m³)", f"{(k['eficiencia'] if k['eficiencia'] is not None else 0):.3f}")
    c5.metric("Horas bomba ligada", f"{int(k['horas_bomba'])} h")

    st.subheader("Recomendação automática (IA) e alertas")
    nivel, mensagens, meta = recomendacao_ia(
        df_f,
        fase=fase,
        manejo=manejo,
        tipo_solo=tipo_solo,
        risco_frio=risco_frio,
        dias_pos_floracao=int(dias_pos_floracao),
    )

    # registra 1x por combinação + timestamp final do período
    log_key = f"log_rec::{periodo}::{fase}::{manejo}::{tipo_solo}::{risco_frio}::{int(dias_pos_floracao)}::{df_f['timestamp'].max()}::{RUN_ID}"
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
    st.dataframe(df_f.tail(50), use_container_width=True)


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
- Leitura de dados reais quando disponíveis; caso contrário, dados sintéticos para validação.
- Variáveis: lâmina, vazão, energia, chuva e estado da bomba.

**2) Tratamento e organização**
- Padronização, limpeza e ordenação temporal.
- Preparado para *intervalo irregular* via \(dt\\) (integração de volume/energia).

**3) Indicadores (BI)**
- KPIs operacionais e gráficos de tendência.

**4) Suporte à decisão (IA explicável)**
- Regras interpretáveis para alertas e recomendações.
- Estrutura preparada para evolução com modelos preditivos.

**5) Evidências**
- Logs em CSV com versão do app e ID de execução (auditoria).
        """.strip()
    )


# =========================================================
# TAB 3 — RESULTADOS PILOTO
# =========================================================
with tabs[3]:
    st.subheader("Resultados (piloto) — Baseline vs Otimizado (simulação por regras)")
    st.caption(
        "Comparação do cenário atual (baseline) com um cenário otimizado baseado em regras interpretáveis "
        "(ex.: adiar bombeamento após chuva relevante e evitar excesso de lâmina)."
    )

    df_sim = aplicar_otimizacao_regras(df_f, lamina_max=lamina_max, chuva_min_mm=chuva_min_mm)
    res = comparar_cenarios(df_sim, tarifa_kwh=tarifa_kwh)

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

    st.markdown("### Gráficos comparativos (energia e vazão)")
    comp = pd.DataFrame({"timestamp": df_sim["timestamp"]}).set_index("timestamp")

    if "energia_kwh" in df_sim.columns:
        comp["energia_base_kwh"] = df_sim["energia_kwh"]
    if "energia_otim_kwh" in df_sim.columns:
        comp["energia_otim_kwh"] = df_sim["energia_otim_kwh"]
    if "vazao_m3h" in df_sim.columns:
        comp["vazao_base_m3h"] = df_sim["vazao_m3h"]
    if "vazao_otim_m3h" in df_sim.columns:
        comp["vazao_otim_m3h"] = df_sim["vazao_otim_m3h"]

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
            tarifa_kwh=tarifa_kwh,
            username=login_user,
            evaluator=evaluator_name,
        )
        st.success(f"Resultados salvos em: {path}")

    if os.path.exists(RES_LOG_PATH):
        st.caption("Histórico salvo (últimas 20 linhas):")
        hist = pd.read_csv(RES_LOG_PATH)
        st.dataframe(hist.tail(20), use_container_width=True)


# =========================================================
# TAB 4 — MODELO ECONÔMICO
# =========================================================
with tabs[4]:
    st.subheader("Modelo Econômico (TCC) — Economia → Cenários → VPL → Precificação")

    df_sim = aplicar_otimizacao_regras(df_f, lamina_max=lamina_max, chuva_min_mm=chuva_min_mm)
    res = comparar_cenarios(df_sim, tarifa_kwh=tarifa_kwh)

    economia_periodo_rs = float(res["economia_rs"])
    economia_periodo_kwh = float(res["economia_kwh"])
    custo_base_periodo_rs = float(res["custo_base"])

    # fator para “mensalizar” conforme período escolhido
    if periodo == "Últimas 24h":
        fator_mes = 30
        dias_periodo = 1
    elif periodo == "Últimos 3 dias":
        fator_mes = 10
        dias_periodo = 3
    elif periodo == "Últimos 7 dias":
        fator_mes = (30 / 7)
        dias_periodo = 7
    else:
        dias_periodo = max(1, int((df_f["timestamp"].max() - df_f["timestamp"].min()).total_seconds() / 86400))
        fator_mes = 30 / dias_periodo

    economia_mensal_rs = economia_periodo_rs * fator_mes
    economia_mensal_kwh = economia_periodo_kwh * fator_mes

    economia_diaria_rs = economia_periodo_rs / dias_periodo
    economia_safra_rs = economia_diaria_rs * duracao_safra_dias

    preco_sugerido = calc_precificacao_por_valor(
        economia_rs_mensal=economia_mensal_rs,
        pct_captura=pct_captura,
        piso=piso_plano,
        teto=teto_plano,
    )

    met = calc_roi_payback(
        preco_mensal=preco_sugerido,
        economia_rs_mensal=economia_mensal_rs,
        investimento_inicial=investimento_inicial,
    )

    st.markdown("### Economia estimada (a partir do período selecionado)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Economia (R$/mês)", fmt_br_money(economia_mensal_rs))
    c2.metric("Economia (kWh/mês)", f"{economia_mensal_kwh:,.1f}".replace(",", "X").replace(".", ",").replace("X", "."))
    c3.metric("Economia na safra (R$)", fmt_br_money(economia_safra_rs))
    c4.metric("Tarifa usada (R$/kWh)", f"{tarifa_kwh:.2f}".replace(".", ","))

    st.markdown("### Precificação + ROI/Payback (ROI do cliente sobre a mensalidade)")
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Preço sugerido (R$/mês)", fmt_br_money(preco_sugerido))
    p2.metric("ROI mensal (líquido)", f"{met['roi_mensal']*100:.1f}%" if met["roi_mensal"] is not None else "—")
    p3.metric("Ganho líquido (R$/mês)", fmt_br_money(met["ganho_liquido"]) if met["ganho_liquido"] is not None else "—")
    p4.metric("Payback (meses)", f"{met['payback_meses']:.1f}".replace(".", ",") if met["payback_meses"] is not None else "—")

    st.divider()

    # Cenários 5/10/15%
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

    # VPL
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

    # Fenologia: r variável
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
        "Use isso como forma simples de conectar manejo fenológico ao modelo econômico."
    )

    df_fen_show = df_fen.copy()
    df_fen_show["Economia bruta (R$)"] = df_fen_show["economia_bruta_rs"].round(2)
    st.dataframe(df_fen_show[["fase", "r_pct", "Economia bruta (R$)"]].rename(columns={"r_pct": "r (%)"}), use_container_width=True)

    fig_f = plot_fenologia_bar(df_fen, title=f"Economia bruta estimada por fase — base do período ({periodo})")
    png_f = fig_to_png_bytes(fig_f)
    st.image(png_f, caption="Gráfico — Economia por fase fenológica", use_container_width=True)
    st.download_button("Baixar gráfico (PNG) — Fenologia", data=png_f, file_name="grafico_fenologia.png", mime="image/png")

    st.divider()

    # Evidências exportáveis
    st.markdown("### Evidência exportável (para anexar no TCC)")
    evidencia = {
        "periodo": periodo,
        "tarifa_rs_kwh": float(tarifa_kwh),
        "horas_dia_ref": int(horas_dia_ref),
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
        "investimento_inicial_rs": float(investimento_inicial),
        "roi_mensal": met["roi_mensal"],
        "payback_meses": met["payback_meses"],
        "taxa_desconto_aa": float(taxa_desconto),
        "horizonte_anos": int(horizonte_anos),
        "r_fase_vegetativa": float(r_veg),
        "r_fase_reprodutiva": float(r_rep),
        "r_fase_emborrachamento_floracao": float(r_flo),
        "r_fase_maturacao": float(r_mat),
        "app_version": APP_VERSION,
        "run_id": RUN_ID,
    }
    st.json(evidencia)

    if st.button("Salvar evidência econômica (CSV)"):
        os.makedirs(LOG_DIR, exist_ok=True)
        df_row = pd.DataFrame([{
            "timestamp_execucao": datetime.now().isoformat(timespec="seconds"),
            "user": login_user,
            "evaluator": evaluator_name,
            **evidencia
        }])
        if not os.path.exists(ECO_LOG_PATH):
            df_row.to_csv(ECO_LOG_PATH, index=False, encoding="utf-8")
        else:
            df_row.to_csv(ECO_LOG_PATH, mode="a", header=False, index=False, encoding="utf-8")
        st.success(f"Modelo econômico salvo em: {ECO_LOG_PATH}")

    if st.button("Salvar tabela de cenários/VPL/fenologia (CSV)"):
        ts = datetime.now().isoformat(timespec="seconds")
        df_out = df_vpl[["cenário", "r", "vpl_projeto_rs", "vpl_produtor_rs"]].copy()
        df_out["timestamp_execucao"] = ts
        df_out["user"] = login_user
        df_out["evaluator"] = evaluator_name
        df_out["periodo"] = periodo
        df_out["tarifa_rs_kwh"] = float(tarifa_kwh)
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
        """.strip()
    )
    st.info(
        "Seção indicada para o TCC: evidencia mensuração/auditoria do protótipo (mesmo sendo MVP por regras)."
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
    st.subheader("MVP vs Visão Futura")
    st.markdown(
        """
| Dimensão | MVP (piloto) | Visão Futura (escala) |
|---|---|---|
| Objetivo | Evidenciar economia e controle operacional | Otimização e previsões com IA |
| Tipo de análise | Descritiva/diagnóstica (BI) | Preditiva/prescritiva (ML) |
| Alertas | Regras interpretáveis | Alertas inteligentes + priorização |
| Integração | Sensores/SCADA + histórico | APIs clima + mobile + benchmark |
| Implantação | Customizada no piloto | Padronizada e replicável |
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
        )
        st.success(f"Qualidade do dado salva em: {path}")

    if os.path.exists(DQ_LOG_PATH):
        st.caption("Histórico (data_quality_log.csv) — últimas 20 linhas:")
        hist_dq = pd.read_csv(DQ_LOG_PATH)
        st.dataframe(hist_dq.tail(20), use_container_width=True)
