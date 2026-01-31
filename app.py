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

    # Cria cabeçalho se arquivo não existe
    if not os.path.exists(log_path):
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("timestamp,event,user,evaluator\n")

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(row)



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

            # logar apenas 1x por sessão de login
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
        chuva[idx : idx + dur] += rng.uniform(1, 6)

    bomba_ligada = (rng.random(n_horas) > 0.35).astype(int)

    vazao = np.where(bomba_ligada == 1, rng.normal(75, 12, n_horas), rng.normal(5, 2, n_horas))
    vazao = np.clip(vazao, 0, None)

    energia = np.where(bomba_ligada == 1, rng.normal(55, 10, n_horas), rng.normal(2, 1, n_horas))
    energia = np.clip(energia, 0, None)

    lamina = np.zeros(n_horas)
    lamina[0] = 7.5
    for i in range(1, n_horas):
        ganho_irrig = (vazao[i] / 1200)  # efeito aproximado da irrigação no protótipo
        ganho_chuva = (chuva[i] / 20)    # conversão aproximada mm → cm para simulação
        perda = rng.normal(0.02, 0.03)   # perdas/evaporação aproximadas
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
    total_energia = df["energia_kwh"].sum()
    total_volume = df["vazao_m3h"].sum()
    lamina_media = df["lamina_cm"].mean()

    eficiencia = (total_energia / total_volume) if total_volume > 0 else None
    horas_bomba = int(df["bomba_ligada"].sum())
    chuva_24h = df[df["timestamp"] >= (df["timestamp"].max() - pd.Timedelta(hours=24))]["chuva_mm"].sum()

    return {
        "lamina_media": lamina_media,
        "total_energia": total_energia,
        "total_volume": total_volume,
        "eficiencia": eficiencia,
        "horas_bomba": horas_bomba,
        "chuva_24h": chuva_24h,
    }


def recomendacao_ia(df):
    """
    Suporte à decisão baseado em regras (IA explicável).
    Gera recomendações e alertas interpretáveis para o manejo, a partir de dados operacionais.
    """
    df = df.copy()
    agora = df["timestamp"].max()

    ult_6h = df[df["timestamp"] >= (agora - pd.Timedelta(hours=6))]
    ult_24h = df[df["timestamp"] >= (agora - pd.Timedelta(hours=24))]

    chuva_24h = ult_24h["chuva_mm"].sum()
    lamina_atual = float(df.iloc[-1]["lamina_cm"])

    energia_6h = ult_6h["energia_kwh"].sum()
    volume_6h = ult_6h["vazao_m3h"].sum()
    eficiencia_6h = (energia_6h / volume_6h) if volume_6h > 0 else None

    df_ligada = df[df["bomba_ligada"] == 1]
    base_ef = None
    if len(df_ligada) > 10 and df_ligada["vazao_m3h"].sum() > 0:
        base_ef = df_ligada["energia_kwh"].sum() / df_ligada["vazao_m3h"].sum()

    mensagens = []
    nivel = "info"

    if chuva_24h >= 10:
        mensagens.append(
            f"Recomendação: considerar reduzir/adiar o bombeamento nas próximas 6h "
            f"(chuva acumulada nas últimas 24h: {chuva_24h:.1f} mm)."
        )
        nivel = "warning"

    if lamina_atual >= 9.5:
        mensagens.append(
            f"Alerta: lâmina d’água elevada ({lamina_atual:.1f} cm). "
            "Avaliar redução do tempo de bombeamento para evitar excesso."
        )
        nivel = "warning"

    if lamina_atual <= 6.0:
        mensagens.append(
            f"Ação prioritária: lâmina d’água baixa ({lamina_atual:.1f} cm). "
            "Priorizar reposição e verificar perdas/condições de entrada de água."
        )
        nivel = "error"

    if eficiencia_6h is not None and base_ef is not None:
        if eficiencia_6h > base_ef * 1.15:
            mensagens.append(
                f"Alerta de eficiência energética: últimas 6h = {eficiencia_6h:.3f} kWh/m³ "
                f"vs baseline = {base_ef:.3f} kWh/m³. "
                "Recomenda-se inspeção de condições hidráulicas e operacionais."
            )
            nivel = "warning"

    if not mensagens:
        mensagens.append(
            "Condição operacional dentro do padrão observado para o período. "
            "Manter monitoramento e reavaliar em intervalo regular."
        )
        nivel = "success"

    return nivel, mensagens


def bloco_contexto_tcc():
    st.markdown(
        """
        **Contextualização (TCC):** Este dashboard demonstra a aplicação de *Business Intelligence* e *Data Science* 
        no monitoramento da irrigação do arroz irrigado, com foco em eficiência hídrica e energética. 
        Dados de diferentes fontes (sensores/SCADA/clima) são consolidados e transformados em indicadores (KPIs), 
        alertas e recomendações automatizadas, apoiando a tomada de decisão do manejo.
        """
    )


# =========================================================
# UI — TOPO (com nome do avaliador) + SIDEBAR (Sair)
# =========================================================
evaluator_name = st.session_state.get("evaluator_name", "Avaliador")
login_user = st.session_state.get("login_user", "usuario")

st.title(APP_TITLE)
st.caption(APP_SUBTITLE)

# ✅ Mostra no topo o avaliador
st.markdown(f"**Avaliador:** {evaluator_name}")

bloco_contexto_tcc()

with st.sidebar:
    st.header("Sessão")
    st.caption(f"Usuário autenticado: **{login_user}**")
    st.caption(f"Avaliador: **{evaluator_name}**")

    if st.button("Sair"):
        # log de logout
        log_access_event("LOGOUT", login_user, evaluator_name)

        # limpa sessão
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

if periodo == "Últimas 24h":
    df_f = df[df["timestamp"] >= (max_data - pd.Timedelta(hours=24))]
elif periodo == "Últimos 3 dias":
    df_f = df[df["timestamp"] >= (max_data - pd.Timedelta(days=3))]
elif periodo == "Últimos 7 dias":
    df_f = df[df["timestamp"] >= (max_data - pd.Timedelta(days=7))]
else:
    df_f = df.copy()

tabs = st.tabs(["Dashboard", "Arquitetura da Solução", "Metodologia (simulação)"])


with tabs[0]:
    k = kpis_basicos(df_f)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Lâmina média (cm)", f"{k['lamina_media']:.2f}")
    c2.metric("Energia total (kWh)", f"{k['total_energia']:.1f}")
    c3.metric("Volume total (m³)", f"{k['total_volume']:.1f}")
    c4.metric("Eficiência (kWh/m³)", f"{(k['eficiencia'] if k['eficiencia'] is not None else 0):.3f}")
    c5.metric("Horas bomba ligada", f"{int(k['horas_bomba'])} h")

    st.subheader("Recomendação automática (IA) e alertas")
    nivel, mensagens = recomendacao_ia(df_f)

    texto = "\n".join([f"- {m}" for m in mensagens])
    if nivel == "success":
        st.success(texto)
    elif nivel == "info":
        st.info(texto)
    elif nivel == "warning":
        st.warning(texto)
    else:
        st.error(texto)

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