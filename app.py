import os
import hmac
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st


# =========================================================
# UTIL: Credenciais + Nome do Avaliador
# =========================================================
def get_settings():
    """
    Lê configurações via st.secrets (produção) ou variáveis de ambiente (local).
    """
    app_user = str(st.secrets.get("APP_USER", os.getenv("APP_USER", "admin")))
    app_pass = str(st.secrets.get("APP_PASSWORD", os.getenv("APP_PASSWORD", "admin")))
    evaluator = str(st.secrets.get("APP_EVALUATOR_NAME", os.getenv("APP_EVALUATOR_NAME", "Avaliador")))
    return app_user, app_pass, evaluator


def log_access_event(event: str, username: str, evaluator: str):
    """
    Registra eventos simples de acesso em CSV (timestamp local).
    Observação: em Streamlit Cloud, o filesystem pode ser efêmero; para TCC normalmente é suficiente.
    """
    os.makedirs("logs", exist_ok=True)
    log_path = os.path.join("logs", "access_log.csv")

    ts = datetime.now().isoformat(timespec="seconds")
    row = f"{ts},{event},{username},{evaluator}\n"

    if not os.path.exists(log_path):
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("timestamp,event,user,evaluator\n")

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
    """
    Registra recomendações/alertas gerados pelo motor de regras em CSV.
    Inclui mini-resumo operacional (6h/24h) e estado da bomba.
    """
    os.makedirs("logs", exist_ok=True)
    log_path = os.path.join("logs", "recommendations_log.csv")

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

    if not os.path.exists(log_path):
        df_row.to_csv(log_path, index=False, encoding="utf-8")
    else:
        df_row.to_csv(log_path, mode="a", header=False, index=False, encoding="utf-8")

    return log_path


def clear_logs():
    """
    Remove logs CSV e limpa flags de sessão associadas.
    """
    os.makedirs("logs", exist_ok=True)
    rec_log_path = os.path.join("logs", "recommendations_log.csv")
    res_log_path = os.path.join("logs", "resultados_piloto.csv")
    eco_log_path = os.path.join("logs", "modelo_economico.csv")

    removed = []
    for p in [rec_log_path, res_log_path, eco_log_path]:
        if os.path.exists(p):
            os.remove(p)
            removed.append(os.path.basename(p))

    # limpa flags de sessão para permitir novo registro limpo
    for k in list(st.session_state.keys()):
        if str(k).startswith("log_rec::"):
            st.session_state.pop(k, None)

    return removed


def check_login():
    """
    Tela de login com sessão. Sem autenticação, não renderiza absolutamente nada do dashboard.
    """
    if st.session_state.get("authenticated", False):
        return True

    user_ok, pass_ok, evaluator = get_settings()

    st.set_page_config(page_title="AgroData — Login", layout="wide")

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


st.set_page_config(
    page_title="AgroData — Irrigação",
    layout="wide",
)

APP_TITLE = "AgroData — Irrigação (BI + Data Science + IA)"
APP_SUBTITLE = (
    "Protótipo acadêmico para o TCC: monitoramento operacional da irrigação com indicadores (KPIs), "
    "alertas e recomendações automatizadas de suporte à decisão."
)

DATA_CSV_PATH = os.path.join("data", "dados_irrigacao.csv")
ARQ_IMG_PATH = os.path.join("data", "arquitetura_irrigacao.png")


# =========================================================
# FUNÇÕES (DADOS, KPIs e SUPORTE À DECISÃO)
# =========================================================
def gerar_dados_exemplo(n_horas=240, seed=42):
    """
    Gera uma base sintética (frequência horária) para validação do pipeline do protótipo,
    com variáveis típicas do domínio: lâmina d’água, vazão, energia, chuva e estado da bomba.
    """
    rng = np.random.default_rng(seed)
    now = datetime.now().replace(minute=0, second=0, microsecond=0)
    ts = pd.date_range(end=now, periods=n_horas, freq="H")

    chuva = np.zeros(n_horas)
    for _ in range(10):
        idx = rng.integers(0, n_horas)
        dur = rng.integers(2, 8)
        chuva[idx: idx + dur] += rng.uniform(1, 6)

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


def carregar_dados():
    """
    Carrega dados reais (CSV) quando disponíveis. Caso não exista arquivo,
    utiliza dados sintéticos para demonstrar as funcionalidades do protótipo.
    """
    if os.path.exists(DATA_CSV_PATH):
        df = pd.read_csv(DATA_CSV_PATH)

        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        else:
            for c in df.columns:
                if "data" in c.lower() or "hora" in c.lower():
                    df["timestamp"] = pd.to_datetime(df[c], errors="coerce")
                    break

        df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
        return df

    return gerar_dados_exemplo()


def kpis_basicos(df):
    total_energia = float(df["energia_kwh"].sum())
    total_volume = float(df["vazao_m3h"].sum())
    lamina_media = float(df["lamina_cm"].mean())

    eficiencia = (total_energia / total_volume) if total_volume > 0 else None
    horas_bomba = int(df["bomba_ligada"].sum())
    chuva_24h = float(df[df["timestamp"] >= (df["timestamp"].max() - pd.Timedelta(hours=24))]["chuva_mm"].sum())

    return {
        "lamina_media": lamina_media,
        "total_energia": total_energia,
        "total_volume": total_volume,
        "eficiencia": eficiencia,
        "horas_bomba": horas_bomba,
        "chuva_24h": chuva_24h,
    }


def recomendacao_ia(df, fase: str, manejo: str, tipo_solo: str, risco_frio: bool, dias_pos_floracao: int):
    """
    Motor de regras (IA explicável) para suporte à decisão no manejo da irrigação do arroz.
    - Usa faixas técnicas de lâmina, efeito de chuva (crédito hídrico) e opções por fase.
    - Inclui mini-resumo operacional (6h/24h) e estado atual da bomba para logging.

    Ajuste TCC: reforço de manejo contínuo com referência ~7,5 cm (faixa operacional 6,5–8,5 cm),
    prática comum após 1–2 folhas e sustentada por ~80–100 dias (configurado na seção econômica).
    """
    df = df.copy().sort_values("timestamp")
    agora = df["timestamp"].max()

    ult_6h = df[df["timestamp"] >= (agora - pd.Timedelta(hours=6))]
    ult_24h = df[df["timestamp"] >= (agora - pd.Timedelta(hours=24))]

    chuva_24h = float(ult_24h["chuva_mm"].sum())
    lamina_atual = float(df.iloc[-1]["lamina_cm"])
    bomba_atual = int(df.iloc[-1]["bomba_ligada"])

    # 6h
    bomba_horas_6h = int(ult_6h["bomba_ligada"].sum())
    energia_6h = float(ult_6h["energia_kwh"].sum())
    volume_6h = float(ult_6h["vazao_m3h"].sum())
    eficiencia_6h = (energia_6h / volume_6h) if volume_6h > 0 else None

    # 24h
    bomba_horas_24h = int(ult_24h["bomba_ligada"].sum())
    energia_24h = float(ult_24h["energia_kwh"].sum())
    volume_24h = float(ult_24h["vazao_m3h"].sum())
    eficiencia_24h = (energia_24h / volume_24h) if volume_24h > 0 else None

    lamina_min_24h = float(ult_24h["lamina_cm"].min()) if len(ult_24h) else np.nan
    lamina_max_24h = float(ult_24h["lamina_cm"].max()) if len(ult_24h) else np.nan

    # baseline eficiência (histórico com bomba ligada)
    df_ligada = df[df["bomba_ligada"] == 1]
    base_ef = None
    if len(df_ligada) > 10 and float(df_ligada["vazao_m3h"].sum()) > 0:
        base_ef = float(df_ligada["energia_kwh"].sum() / df_ligada["vazao_m3h"].sum())

    mensagens = []
    nivel = "info"

    # 1) chuva como crédito hídrico (heurística: demanda ~12 mm/dia)
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
        else:  # Maturação
            alvo_min, alvo_max = 2.5, 7.5

    # reforço para manejo contínuo (inundação contínua ~7,5 cm)
    if manejo == "Contínuo" and not (risco_frio and fase == "Emborrachamento/Floração"):
        alvo_min, alvo_max = 6.5, 8.5
        mensagens.append("Manejo contínuo: referência operacional ~7,5 cm (faixa 6,5–8,5 cm), usual após 1–2 folhas.")
        if nivel == "success":
            nivel = "info"

    # alertas por lâmina
    if lamina_atual > 10.0 and not (risco_frio and fase == "Emborrachamento/Floração"):
        mensagens.append(
            f"Alerta: lâmina alta ({lamina_atual:.1f} cm). "
            "Valores >10 cm podem aumentar perdas e risco de acamamento. Reduzir bombeamento e monitorar."
        )
        nivel = "warning"

    if lamina_atual < 2.5 and fase != "Maturação":
        mensagens.append(
            f"Atenção: lâmina muito baixa ({lamina_atual:.1f} cm). "
            "Abaixo de ~2,5 cm exige controle operacional mais rigoroso. Avaliar reposição."
        )
        nivel = "warning" if nivel != "error" else nivel

    # faixa-alvo por fase
    if lamina_atual < alvo_min:
        mensagens.append(
            f"Ação: lâmina abaixo do alvo da fase {fase} ({lamina_atual:.1f} < {alvo_min:.1f} cm). "
            "Priorizar reposição e verificar perdas/entrada de água."
        )
        nivel = "error"
    elif lamina_atual > alvo_max:
        mensagens.append(
            f"Observação: lâmina acima do alvo da fase {fase} ({lamina_atual:.1f} > {alvo_max:.1f} cm). "
            "Reduzir bombeamento e acompanhar tendência."
        )
        nivel = "warning" if nivel != "error" else nivel
    else:
        mensagens.append(f"OK: lâmina dentro do alvo da fase {fase} ({alvo_min:.1f}–{alvo_max:.1f} cm).")
        if nivel == "info":
            nivel = "success"

    # 3) manejo intermitente (controle simples ON/OFF)
    if manejo.startswith("Intermitente") and fase != "Maturação":
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

    # 4) supressão na maturação (regra simples por solo / dias pós-floração)
    if fase == "Maturação":
        if dias_pos_floracao >= 10 and tipo_solo == "Argiloso":
            mensagens.append(
                f"Maturação (solo argiloso): {dias_pos_floracao} dias pós-floração. "
                "Pode-se considerar iniciar supressão da irrigação, monitorando estágio do grão."
            )
            if nivel == "success":
                nivel = "info"
        elif dias_pos_floracao >= 10 and tipo_solo != "Argiloso":
            mensagens.append(
                f"Maturação (solo arenoso/bem drenado): {dias_pos_floracao} dias pós-floração. "
                "Recomendação: cautela e possível postergação da supressão devido à maior drenagem."
            )
            if nivel == "success":
                nivel = "info"
        else:
            mensagens.append(
                "Maturação: para supressão, use floração plena e estágio do grão (ex.: pastoso predominante) como referência."
            )
            if nivel == "success":
                nivel = "info"

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
        "energia_6h": energia_6h,
        "volume_6h": volume_6h,
        "eficiencia_6h": eficiencia_6h,
        "bomba_horas_24h": bomba_horas_24h,
        "energia_24h": energia_24h,
        "volume_24h": volume_24h,
        "eficiencia_24h": eficiencia_24h,
        "lamina_min_24h": lamina_min_24h,
        "lamina_max_24h": lamina_max_24h,
        "baseline_ef": base_ef,
    }

    return nivel, mensagens, meta


# =========================================================
# FUNÇÕES — RESULTADOS (baseline vs otimizado) + LOG CSV
# =========================================================
def aplicar_otimizacao_regras(df, lamina_max=9.5, chuva_min_mm=10.0):
    """
    Simula um cenário otimizado simples:
    - Desliga a bomba quando chuva 24h >= chuva_min_mm OU lâmina >= lamina_max.
    - Caso contrário, mantém o estado original.
    """
    df = df.copy().sort_values("timestamp")

    df["chuva_24h"] = (
        df.set_index("timestamp")["chuva_mm"]
        .rolling("24h")
        .sum()
        .reset_index(drop=True)
    )

    cond_desliga = (df["chuva_24h"] >= chuva_min_mm) | (df["lamina_cm"] >= lamina_max)

    df["bomba_otim"] = np.where(cond_desliga, 0, df["bomba_ligada"])
    df["energia_otim_kwh"] = np.where(df["bomba_otim"] == 1, df["energia_kwh"], 0.0)
    df["vazao_otim_m3h"] = np.where(df["bomba_otim"] == 1, df["vazao_m3h"], 0.0)

    return df


def comparar_cenarios(df, tarifa_kwh=0.95):
    energia_base = float(df["energia_kwh"].sum())
    volume_base = float(df["vazao_m3h"].sum())
    horas_bomba_base = int(df["bomba_ligada"].sum())
    ef_base = (energia_base / volume_base) if volume_base > 0 else np.nan
    custo_base = energia_base * float(tarifa_kwh)

    energia_otim = float(df["energia_otim_kwh"].sum())
    volume_otim = float(df["vazao_otim_m3h"].sum())
    horas_bomba_otim = int(df["bomba_otim"].sum())
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


def salvar_resultados_piloto(res: dict, periodo_label: str, lamina_max: float, chuva_min_mm: float, tarifa_kwh: float):
    """
    Salva uma linha de resultados baseline vs otimizado em CSV (evidência para o TCC).
    """
    os.makedirs("logs", exist_ok=True)
    out_path = os.path.join("logs", "resultados_piloto.csv")

    ts = datetime.now().isoformat(timespec="seconds")

    row = {
        "timestamp_execucao": ts,
        "periodo": periodo_label,
        "lamina_max_cm": float(lamina_max),
        "chuva_min_24h_mm": float(chuva_min_mm),
        "tarifa_rs_kwh": float(tarifa_kwh),

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

    if not os.path.exists(out_path):
        df_row.to_csv(out_path, index=False, encoding="utf-8")
    else:
        df_row.to_csv(out_path, mode="a", header=False, index=False, encoding="utf-8")

    return out_path


# =========================================================
# FUNÇÕES — MODELO ECONÔMICO (value-based pricing) + LOG CSV
# =========================================================
def calc_precificacao_por_valor(
    economia_rs_mensal: float,
    pct_captura: float = 0.15,
    piso: float = 1200.0,
    teto: float = 3800.0
):
    """
    Sugere preço mensal baseado em 'value-based pricing':
    preço = economia_mensal * pct_captura, limitado por piso/teto.
    """
    if economia_rs_mensal is None or economia_rs_mensal <= 0:
        return 0.0
    preco = economia_rs_mensal * float(pct_captura)
    return float(min(max(preco, piso), teto))


def calc_roi_payback(preco_mensal: float, economia_rs_mensal: float, investimento_inicial: float):
    """
    ROI mensal simples e payback em meses.
    ROI = (economia - preço) / preço
    Payback = investimento / (economia - preço)
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


def salvar_modelo_economico(evidencia: dict, username: str, evaluator: str):
    """
    Salva uma linha do Modelo Econômico (evidência para o TCC) em CSV.
    Inclui: economia mensal/safra, % captura, preço sugerido, ROI e payback.
    """
    os.makedirs("logs", exist_ok=True)
    out_path = os.path.join("logs", "modelo_economico.csv")

    ts = datetime.now().isoformat(timespec="seconds")

    row = {
        "timestamp_execucao": ts,
        "user": username,
        "evaluator": evaluator,

        "periodo": str(evidencia.get("periodo")),
        "tarifa_rs_kwh": float(evidencia.get("tarifa_rs_kwh", 0.0)),
        "lamina_max_cm": float(evidencia.get("lamina_max_cm", np.nan)),
        "chuva_min_24h_mm": float(evidencia.get("chuva_min_24h_mm", np.nan)),

        "economia_periodo_rs": float(evidencia.get("economia_periodo_rs", 0.0)),
        "economia_mensal_rs_estim": float(evidencia.get("economia_mensal_rs_estim", 0.0)),
        "economia_safra_dias": int(evidencia.get("economia_safra_dias", 0)),
        "economia_safra_rs_estim": float(evidencia.get("economia_safra_rs_estim", 0.0)),

        "pct_captura": float(evidencia.get("pct_captura", 0.0)),
        "preco_sugerido_rs_mensal": float(evidencia.get("preco_sugerido_rs_mensal", 0.0)),
        "investimento_inicial_rs": float(evidencia.get("investimento_inicial_rs", 0.0)),

        "roi_mensal": float(evidencia.get("roi_mensal")) if evidencia.get("roi_mensal") is not None else np.nan,
        "payback_meses": float(evidencia.get("payback_meses")) if evidencia.get("payback_meses") is not None else np.nan,
    }

    df_row = pd.DataFrame([row])

    if not os.path.exists(out_path):
        df_row.to_csv(out_path, index=False, encoding="utf-8")
    else:
        df_row.to_csv(out_path, mode="a", header=False, index=False, encoding="utf-8")

    return out_path


def bloco_contexto_tcc():
    st.markdown(
        """
        **Contextualização (TCC):** Este dashboard demonstra a aplicação de *Business Intelligence* e *Data Science*
        no monitoramento da irrigação do arroz irrigado, com foco em eficiência hídrica e energética.
        Dados de diferentes fontes (sensores/SCADA/clima) são consolidados e transformados em indicadores (KPIs),
        alertas e recomendações automatizadas, apoiando a tomada de decisão do manejo.
        """
    )


def fmt_br_money(x: float) -> str:
    try:
        return f"{float(x):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except Exception:
        return "—"


# =========================================================
# UI — TOPO (com nome do avaliador) + SIDEBAR (Sair)
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

    if st.button("Sair"):
        log_access_event("LOGOUT", login_user, evaluator_name)
        st.session_state["authenticated"] = False
        st.session_state["logged_access"] = False
        st.session_state.pop("login_user", None)
        st.session_state.pop("evaluator_name", None)
        st.rerun()


# =========================================================
# APP
# =========================================================
df = carregar_dados()

st.sidebar.header("Filtros")
max_data = df["timestamp"].max()

periodo = st.sidebar.selectbox(
    "Período",
    options=["Últimas 24h", "Últimos 3 dias", "Últimos 7 dias", "Tudo"],
    index=2,
)

# parâmetros agronômicos (motor de regras)
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

# parâmetros da simulação (aba Resultados)
st.sidebar.header("Parâmetros (simulação)")
tarifa_kwh = st.sidebar.number_input("Tarifa de energia (R$/kWh)", min_value=0.0, value=0.95, step=0.05)
lamina_max = st.sidebar.slider("Lâmina máxima (cm)", min_value=7.0, max_value=20.0, value=10.0, step=0.5)
chuva_min_mm = st.sidebar.slider("Chuva 24h (mm) para reduzir/adiar", min_value=0.0, max_value=50.0, value=12.0, step=1.0)

# modelo econômico (pedido do orientador)
st.sidebar.header("Modelo econômico (TCC)")
pct_captura = st.sidebar.slider("Captura de valor (% da economia)", 5, 30, 15, 1) / 100.0
investimento_inicial = st.sidebar.number_input("Investimento inicial estimado (R$)", min_value=0.0, value=12400.0, step=100.0)

st.sidebar.subheader("Safra (irrigação contínua)")
duracao_safra_dias = st.sidebar.slider("Duração da lâmina contínua (dias)", 80, 100, 90, 1)

piso_plano = st.sidebar.number_input("Piso do plano (R$/mês)", min_value=0.0, value=1200.0, step=100.0)
teto_plano = st.sidebar.number_input("Teto do plano (R$/mês)", min_value=0.0, value=3800.0, step=100.0)

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

rec_log_path = os.path.join("logs", "recommendations_log.csv")
res_log_path = os.path.join("logs", "resultados_piloto.csv")
eco_log_path = os.path.join("logs", "modelo_economico.csv")

if os.path.exists(rec_log_path):
    with open(rec_log_path, "rb") as f:
        st.sidebar.download_button("Baixar recommendations_log.csv", data=f, file_name="recommendations_log.csv", mime="text/csv")

if os.path.exists(res_log_path):
    with open(res_log_path, "rb") as f:
        st.sidebar.download_button("Baixar resultados_piloto.csv", data=f, file_name="resultados_piloto.csv", mime="text/csv")

if os.path.exists(eco_log_path):
    with open(eco_log_path, "rb") as f:
        st.sidebar.download_button("Baixar modelo_economico.csv", data=f, file_name="modelo_economico.csv", mime="text/csv")

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
    "Segmentação (TCC)",
    "MVP vs Visão Futura",
    "Escala & Gargalos",
])

# =========================================================
# TAB 0 — DASHBOARD
# =========================================================
with tabs[0]:
    k = kpis_basicos(df_f)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Lâmina média (cm)", f"{k['lamina_media']:.2f}")
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
    log_key = f"log_rec::{periodo}::{fase}::{manejo}::{tipo_solo}::{risco_frio}::{int(dias_pos_floracao)}::{df_f['timestamp'].max()}"
    if st.session_state.get(log_key) != True:
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
        st.line_chart(df_f.set_index("timestamp")["lamina_cm"])

    with cc2:
        st.caption("Energia (kWh/h) e Vazão (m³/h)")
        st.line_chart(df_f.set_index("timestamp")[["energia_kwh", "vazao_m3h"]])

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
        """
    )

    if os.path.exists(ARQ_IMG_PATH):
        st.image(ARQ_IMG_PATH, caption="Arquitetura da solução (imagem)", use_container_width=True)
    else:
        st.caption("Imagem de arquitetura não encontrada em: data/arquitetura_irrigacao.png")


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
        - Consolidação histórica para análise.

        **3) Indicadores (BI)**
        - KPIs operacionais e gráficos de tendência.

        **4) Suporte à decisão (IA explicável)**
        - Regras interpretáveis para alertas e recomendações.
        - Estrutura preparada para evolução com modelos preditivos.
        """
    )


# =========================================================
# TAB 3 — RESULTADOS PILOTO
# =========================================================
with tabs[3]:
    st.subheader("Resultados (piloto) — Baseline vs Otimizado (simulação por regras)")
    st.caption(
        "Comparação do cenário atual (baseline) com um cenário otimizado baseado em regras interpretáveis "
        "(ex.: adiar bombeamento após chuva relevante e evitar excesso de lâmina). "
        "Quando houver dados reais no CSV, a comparação se aplica ao período selecionado."
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

    st.markdown("### Gráfico comparativo (energia e vazão)")
    comp = pd.DataFrame(
        {
            "timestamp": df_sim["timestamp"],
            "energia_base_kwh": df_sim["energia_kwh"],
            "energia_otim_kwh": df_sim["energia_otim_kwh"],
            "vazao_base_m3h": df_sim["vazao_m3h"],
            "vazao_otim_m3h": df_sim["vazao_otim_m3h"],
        }
    ).set_index("timestamp")

    st.line_chart(comp[["energia_base_kwh", "energia_otim_kwh"]])
    st.line_chart(comp[["vazao_base_m3h", "vazao_otim_m3h"]])

    st.markdown("### Evidência para o TCC (CSV)")
    if st.button("Salvar resultados desta simulação"):
        path = salvar_resultados_piloto(
            res=res,
            periodo_label=periodo,
            lamina_max=lamina_max,
            chuva_min_mm=chuva_min_mm,
            tarifa_kwh=tarifa_kwh,
        )
        st.success(f"Resultados salvos em: {path}")

    if os.path.exists(res_log_path):
        st.caption("Histórico salvo (últimas 20 linhas):")
        hist = pd.read_csv(res_log_path)
        st.dataframe(hist.tail(20), use_container_width=True)

        with open(res_log_path, "rb") as f:
            st.download_button(
                label="Baixar CSV de evidências",
                data=f,
                file_name="resultados_piloto.csv",
                mime="text/csv",
            )

    with st.expander("Ver base simulada/real com colunas de otimização (amostra)"):
        st.dataframe(df_sim.tail(80), use_container_width=True)


# =========================================================
# TAB 4 — MODELO ECONÔMICO
# =========================================================
with tabs[4]:
    st.subheader("Modelo Econômico (TCC) — Economia → Precificação → ROI")

    df_sim = aplicar_otimizacao_regras(df_f, lamina_max=lamina_max, chuva_min_mm=chuva_min_mm)
    res = comparar_cenarios(df_sim, tarifa_kwh=tarifa_kwh)

    economia_periodo_rs = float(res["economia_rs"])
    energia_periodo_kwh = float(res["economia_kwh"])

    # fator para "mensalizar" a economia conforme período escolhido
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
        # "Tudo": calcula duração aproximada do período disponível
        dias_periodo = max(1, int((df_f["timestamp"].max() - df_f["timestamp"].min()).total_seconds() / 86400))
        fator_mes = 30 / dias_periodo

    economia_mensal_rs = economia_periodo_rs * fator_mes
    economia_mensal_kwh = energia_periodo_kwh * fator_mes

    # economia na safra (80–100 dias de inundação contínua)
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

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Economia estimada (R$/mês)", fmt_br_money(economia_mensal_rs))
    c2.metric("Economia estimada na safra (R$)", fmt_br_money(economia_safra_rs))
    c3.metric("Preço sugerido (R$/mês)", fmt_br_money(preco_sugerido))
    c4.metric("ROI mensal (líquido)", f"{met['roi_mensal']*100:.1f}%" if met["roi_mensal"] is not None else "—")

    st.markdown("### Payback e interpretação")
    if met["payback_meses"] is not None:
        st.success(f"Payback estimado: **{met['payback_meses']:.1f} meses** (ganho líquido = economia − preço).")
    else:
        st.info("Payback não calculado (economia insuficiente ou preço zerado).")

    st.markdown(
        f"""
**Ligação direta preço ↔ economia (pedido do orientador):**
- O protótipo estima economia financeira (R$) pela diferença entre cenário base e otimizado.
- A precificação é sugerida por captura de valor (**{pct_captura*100:.0f}%** da economia mensal), limitada por piso/teto.
- Isso torna explícita a justificativa econômica da assinatura SaaS (value-based pricing).
        """
    )

    st.markdown("### Evidência exportável (para anexar no TCC)")
    evidencia = {
        "periodo": periodo,
        "tarifa_rs_kwh": tarifa_kwh,
        "lamina_max_cm": lamina_max,
        "chuva_min_24h_mm": chuva_min_mm,

        "economia_periodo_rs": economia_periodo_rs,
        "economia_mensal_rs_estim": economia_mensal_rs,
        "economia_safra_dias": duracao_safra_dias,
        "economia_safra_rs_estim": economia_safra_rs,

        "pct_captura": pct_captura,
        "preco_sugerido_rs_mensal": preco_sugerido,
        "investimento_inicial_rs": investimento_inicial,

        "roi_mensal": met["roi_mensal"],
        "payback_meses": met["payback_meses"],
    }
    st.json(evidencia)

    st.markdown("### Salvar evidência econômica (CSV)")
    colA, colB = st.columns([1, 2])

    with colA:
        if st.button("Salvar modelo econômico"):
            path = salvar_modelo_economico(
                evidencia=evidencia,
                username=login_user,
                evaluator=evaluator_name,
            )
            st.success(f"Modelo econômico salvo em: {path}")

    with colB:
        if os.path.exists(eco_log_path):
            st.caption("Histórico salvo (últimas 20 linhas):")
            hist_eco = pd.read_csv(eco_log_path)
            st.dataframe(hist_eco.tail(20), use_container_width=True)


# =========================================================
# TAB 5 — SEGMENTAÇÃO (TCC)
# =========================================================
with tabs[5]:
    st.subheader("Segmentação (TCC) — Priorização do Cliente Inicial")

    st.markdown(
        """
**Segmento prioritário (beachhead):**
- Produtores de arroz irrigado no RS;
- Médio/grande porte (ex.: área irrigada ≥ 300 ha);
- Bombeamento elétrico + medições (energia/nível/vazão);
- Preferencialmente com SCADA/sensores já instalados;
- Alto custo energético (ex.: > R$ 50 mil/mês).
        """
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
# TAB 6 — MVP vs VISÃO FUTURA
# =========================================================
with tabs[6]:
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
        """
    )


# =========================================================
# TAB 7 — ESCALA & GARGALOS
# =========================================================
with tabs[7]:
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
        """
    )
