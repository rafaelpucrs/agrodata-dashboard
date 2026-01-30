import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st

# -----------------------------
# Config
# -----------------------------
st.set_page_config(
    page_title="AgroData — Irrigação ",
    layout="wide",
)

APP_TITLE = "AgroData — Irrigação  (BI + Data Science + IA)"
APP_SUBTITLE = (
    "Protótipo demonstrativo para o TCC: monitoramento da lâmina d’água, energia e operação das bombas "
    "com indicadores (KPIs), alertas e recomendações automáticas."
)

DATA_CSV_PATH = os.path.join("data", "dados_irrigacao.csv")  # opcional
ARQ_IMG_PATH = os.path.join("data", "arquitetura_irrigacao.png")  # coloque sua imagem aqui

# -----------------------------
# Helpers
# -----------------------------
def gerar_dados_exemplo(n_horas=240, seed=42):
    """Gera uma base sintética simples (10 dias hora a hora) para o dashboard funcionar."""
    rng = np.random.default_rng(seed)
    now = datetime.now().replace(minute=0, second=0, microsecond=0)
    ts = pd.date_range(end=now, periods=n_horas, freq="H")

    # Simula chuva (mm/h) com eventos
    chuva = np.zeros(n_horas)
    for _ in range(10):
        idx = rng.integers(0, n_horas)
        dur = rng.integers(2, 8)
        chuva[idx:idx+dur] += rng.uniform(1, 6)  # evento leve/moderado

    # Vazão (m³/h) e energia (kWh/h) relacionadas à bomba
    bomba_ligada = (rng.random(n_horas) > 0.35).astype(int)
    vazao = np.where(bomba_ligada == 1, rng.normal(75, 12, n_horas), rng.normal(5, 2, n_horas))
    vazao = np.clip(vazao, 0, None)

    energia = np.where(bomba_ligada == 1, rng.normal(55, 10, n_horas), rng.normal(2, 1, n_horas))
    energia = np.clip(energia, 0, None)

    # Lâmina d'água (cm) respondendo à irrigação e chuva
    lamina = np.zeros(n_horas)
    lamina[0] = 7.5
    for i in range(1, n_horas):
        # aumenta com vazao/irrigação e chuva, reduz evap/perdas
        ganho_irrig = (vazao[i] / 1200)  # efeito pequeno
        ganho_chuva = (chuva[i] / 20)    # mm -> cm aproximado (bem simplificado)
        perda = rng.normal(0.02, 0.03)
        lamina[i] = lamina[i-1] + ganho_irrig + ganho_chuva - perda
    lamina = np.clip(lamina, 4.5, 12.0)

    df = pd.DataFrame({
        "timestamp": ts,
        "lamina_cm": lamina,
        "vazao_m3h": vazao,
        "energia_kwh": energia,
        "chuva_mm": chuva,
        "bomba_ligada": bomba_ligada,
    })

    return df


def carregar_dados():
    """Carrega CSV real se existir; senão gera exemplo."""
    if os.path.exists(DATA_CSV_PATH):
        df = pd.read_csv(DATA_CSV_PATH)
        # tenta padronizar
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        else:
            # tenta achar alguma coluna de data/hora
            for c in df.columns:
                if "data" in c.lower() or "hora" in c.lower():
                    df["timestamp"] = pd.to_datetime(df[c], errors="coerce")
                    break
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
        return df
    return gerar_dados_exemplo()


def kpis_basicos(df):
    # períodos e agregações
    total_energia = df["energia_kwh"].sum()
    total_volume = df["vazao_m3h"].sum()  # m³ (porque é por hora e freq=H, simplificação)
    lamina_media = df["lamina_cm"].mean()

    eficiencia = None
    if total_volume > 0:
        eficiencia = total_energia / total_volume  # kWh/m³

    horas_bomba = df["bomba_ligada"].sum()

    chuva_24h = df[df["timestamp"] >= (df["timestamp"].max() - pd.Timedelta(hours=24))]["chuva_mm"].sum()

    return {
        "lamina_media": lamina_media,
        "total_energia": total_energia,
        "total_volume": total_volume,
        "eficiencia": eficiencia,
        "horas_bomba": horas_bomba,
        "chuva_24h": chuva_24h
    }


def recomendacao_ia(df):
    """
    Recomendações automáticas simples (regras + baseline).
    Isso não substitui o ML do TCC, mas serve como 'IA inicial' (sistema especialista),
    e você pode evoluir para modelo preditivo depois.
    """
    df = df.copy()
    agora = df["timestamp"].max()

    # Recortes
    ult_6h = df[df["timestamp"] >= (agora - pd.Timedelta(hours=6))]
    ult_24h = df[df["timestamp"] >= (agora - pd.Timedelta(hours=24))]

    chuva_24h = ult_24h["chuva_mm"].sum()
    lamina_atual = df.iloc[-1]["lamina_cm"]

    # eficiência recente (últimas 6h)
    energia_6h = ult_6h["energia_kwh"].sum()
    volume_6h = ult_6h["vazao_m3h"].sum()
    eficiencia_6h = (energia_6h / volume_6h) if volume_6h > 0 else None

    # baseline simples: média histórica de eficiência quando bomba ligada
    df_ligada = df[df["bomba_ligada"] == 1]
    base_ef = None
    if len(df_ligada) > 10:
        base_ef = df_ligada["energia_kwh"].sum() / df_ligada["vazao_m3h"].sum()

    mensagens = []
    nivel = "info"

    # Regras de decisão (ajuste no TCC conforme recomendações técnicas/IRGA/Embrapa)
    # 1) Chuva relevante -> segurar bombeamento
    if chuva_24h >= 10:
        mensagens.append(f"Sugestão: reduzir/adiar o bombeamento nas próximas 6h (chuva nas últimas 24h: {chuva_24h:.1f} mm).")
        nivel = "warning"

    # 2) Lâmina alta -> reduzir irrigação
    if lamina_atual >= 9.5:
        mensagens.append(f"Atenção: lâmina atual elevada ({lamina_atual:.1f} cm). Avaliar reduzir tempo de bomba para evitar excesso.")
        nivel = "warning"

    # 3) Lâmina baixa -> priorizar reposição
    if lamina_atual <= 6.0:
        mensagens.append(f"Ação recomendada: lâmina baixa ({lamina_atual:.1f} cm). Priorizar reposição e verificar perdas/entrada de água.")
        nivel = "error"

    # 4) Eficiência energética pior que baseline -> investigar
    if eficiencia_6h is not None and base_ef is not None:
        if eficiencia_6h > base_ef * 1.15:
            mensagens.append(
                f"Alerta: eficiência energética abaixo do esperado. "
                f"Últimas 6h = {eficiencia_6h:.3f} kWh/m³ vs baseline = {base_ef:.3f} kWh/m³. "
                "Verificar obstruções, sucção, recalque, válvulas ou condições de operação."
            )
            nivel = "warning"

    if not mensagens:
        mensagens.append("Operação dentro do padrão observado. Manter monitoramento e revisar novamente em algumas horas.")
        nivel = "success"

    return nivel, mensagens


def bloco_contexto_tcc():
    st.markdown(
        """
        **Contexto técnico (TCC):** este dashboard demonstra a aplicação de *Business Intelligence* e *Data Science* 
        no monitoramento da irrigação do arroz irrigado, com foco na eficiência hídrica e energética. 
        Os dados (sensores/SCADA/clima) são consolidados, tratados e transformados em indicadores (KPIs), alertas 
        e recomendações automáticas para apoiar a tomada de decisão do produtor.
        """
    )


# -----------------------------
# UI
# -----------------------------
st.title(APP_TITLE)
st.caption(APP_SUBTITLE)
bloco_contexto_tcc()

df = carregar_dados()

# Sidebar filtros
st.sidebar.header("Filtros")
max_data = df["timestamp"].max()
min_data = df["timestamp"].min()

periodo = st.sidebar.selectbox(
    "Período",
    options=["Últimas 24h", "Últimos 3 dias", "Últimos 7 dias", "Tudo"],
    index=2
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

# -----------------------------
# Tab 1 — Dashboard
# -----------------------------
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

    if nivel == "success":
        st.success("\n".join([f"- {m}" for m in mensagens]))
    elif nivel == "info":
        st.info("\n".join([f"- {m}" for m in mensagens]))
    elif nivel == "warning":
        st.warning("\n".join([f"- {m}" for m in mensagens]))
    else:
        st.error("\n".join([f"- {m}" for m in mensagens]))

    st.subheader("Tendências (período selecionado)")
    cc1, cc2 = st.columns(2)

    with cc1:
        st.caption("Lâmina d'água (cm)")
        st.line_chart(df_f.set_index("timestamp")["lamina_cm"])

    with cc2:
        st.caption("Energia (kWh/h) e Vazão (m³/h)")
        st.line_chart(df_f.set_index("timestamp")[["energia_kwh", "vazao_m3h"]])

    st.subheader("Dados (amostra)")
    st.dataframe(df_f.tail(50), use_container_width=True)

# -----------------------------
# Tab 2 — Arquitetura
# -----------------------------
with tabs[1]:
    st.subheader("Arquitetura da Solução Proposta")

    st.write(
        "A solução é organizada em camadas para permitir coleta contínua, armazenamento, processamento analítico "
        "e visualização. O fluxo principal segue: **Sensores → SCADA → Banco de Dados → Processamento (BI/DS/IA) → Dashboards/Alertas**."
    )

    st.markdown(
        """
        **Componentes principais:**
        - **Sensores / medições:** nível da lâmina d’água, vazão, energia elétrica, clima (chuva/temperatura).
        - **SCADA / automação:** consolida leituras, registra estados e eventos (ligado/desligado/falha).
        - **Banco de dados:** armazena histórico e padroniza as informações.
        - **Processamento (BI + Data Science + IA):** KPIs, detecção de padrões, regras e modelos futuros.
        - **Dashboards + alertas:** visualização e recomendações para apoiar o manejo.
        """
    )



# -----------------------------
# Tab 3 — Metodologia
# -----------------------------
with tabs[2]:
    st.subheader("Metodologia (simulação)")

    st.markdown(
        """
        Esta aba descreve, de forma simplificada, o método aplicado no protótipo (base para o TCC):

        **1) Geração/Coleta de dados**
        - Leitura de dados reais (CSV/SCADA/sensores) **ou** geração de dados sintéticos para teste.
        - Variáveis básicas: *lâmina (cm), vazão (m³/h), energia (kWh/h), chuva (mm), estado da bomba*.

        **2) Tratamento e organização**
        - Padronização de unidades e timestamps.
        - Limpeza (valores ausentes) e ordenação temporal.
        - Consolidação em base histórica para análises.

        **3) Análise e indicadores (BI)**
        - KPIs: lâmina média, energia total, volume total, eficiência (kWh/m³), horas de bomba ligada.
        - Gráficos de tendência para acompanhamento operacional.

        **4) Regras / IA inicial (suporte à decisão)**
        - Regras simples para recomendação (chuva recente, lâmina alta/baixa e eficiência fora do padrão).
        - Base para evolução futura para modelos preditivos (ex.: regressão, séries temporais, LSTM).
        """
    )

    st.write("**Observação:** o protótipo foi estruturado para evoluir do modo regra (IA inicial) para modo preditivo (IA avançada), mantendo rastreabilidade e justificativa das recomendações.")

