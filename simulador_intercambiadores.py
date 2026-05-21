git add simulador_intercambiadores.pyimport streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ─── Configuración ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Simulador — Intercambiadores de Calor",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── CSS ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600;700&display=swap');

html, body, [class*="css"] { background-color: #0e1117; color: #e0e0e0; }
h1,h2,h3,h4,h5,h6 { font-family:'IBM Plex Mono',monospace!important; color:#f5f5f5; }

.stTabs [data-baseweb="tab-list"] {
    gap:6px; background-color:#1a1d27; border-radius:8px; padding:4px;
}
.stTabs [data-baseweb="tab"] {
    font-family:'IBM Plex Mono',monospace; background-color:transparent;
    color:#aaa; border-radius:6px; padding:8px 14px; font-size:0.85rem;
}
.stTabs [aria-selected="true"] {
    background-color:#2d3147!important; color:#ffffff!important;
}

.metric-card {
    background-color:#1e2130; border:1px solid #2d3147;
    border-radius:10px; padding:14px 12px; text-align:center; margin:4px 0;
}
.metric-value {
    font-family:'IBM Plex Mono',monospace; font-size:1.45rem;
    font-weight:700; color:#e8e8e8;
}
.metric-unit { font-size:0.82rem; color:#888; }
.metric-label {
    font-family:'IBM Plex Mono',monospace; font-size:0.70rem;
    color:#777; margin-top:4px;
}

.sel-card {
    background-color:#1e2130; border:2px solid #2d3147;
    border-radius:14px; padding:18px 16px; text-align:center; margin:6px 0;
}
.sel-title {
    font-family:'IBM Plex Mono',monospace; font-size:1.2rem;
    font-weight:700; margin-bottom:6px;
}
.sel-desc {
    font-family:'IBM Plex Mono',monospace; font-size:0.73rem;
    color:#888; margin-bottom:10px;
}

[data-testid="stSidebar"] { background-color:#141720; }
.stAlert { border-radius:8px; }
div.stButton>button { font-family:'IBM Plex Mono',monospace; border-radius:8px; }
</style>
""", unsafe_allow_html=True)

# ─── Constantes ───────────────────────────────────────────────────────────
C_HOT    = "#e63946"
C_COLD_P = "#457b9d"
C_COLD_C = "#f4a261"
C_LIME   = "#b5e550"
BG_FIG   = "#0e1117"
BG_AX    = "#1e2130"
GRID_C   = "#2d3147"
TEXT_C   = "#e0e0e0"

FLUIDOS = {
    "Agua":           4186.0,
    "Aceite térmico": 2000.0,
    "Aire":           1005.0,
}

# ─── Estado de sesión ─────────────────────────────────────────────────────
if "paso" not in st.session_state:
    st.session_state.paso = "seleccion"
if "comparativa" not in st.session_state:
    st.session_state.comparativa = False

# ─── Helpers de cálculo ───────────────────────────────────────────────────
def _lmtd(dt1: float, dt2: float) -> float:
    if dt1 <= 0 or dt2 <= 0:
        return float("nan")
    if abs(dt1 - dt2) < 1e-8:
        return dt1
    return (dt1 - dt2) / np.log(dt1 / dt2)


def calcular(Thi, Tci, mh, mc, Cph, Cpc, U, As):
    Ch   = mh * Cph
    Cc   = mc * Cpc
    Cmin = min(Ch, Cc)
    Cmax = max(Ch, Cc)
    c    = Cmin / Cmax
    NTU  = U * As / Cmin
    Qmax = Cmin * (Thi - Tci)

    eps_p = (1.0 - np.exp(-NTU * (1.0 + c))) / (1.0 + c)
    if abs(c - 1.0) < 1e-6:
        eps_cc = NTU / (1.0 + NTU)
    else:
        num    = 1.0 - np.exp(-NTU * (1.0 - c))
        den    = 1.0 - c * np.exp(-NTU * (1.0 - c))
        eps_cc = num / den

    Qp  = eps_p  * Qmax
    Qcc = eps_cc * Qmax
    Tho_p  = Thi - Qp  / Ch;  Tco_p  = Tci + Qp  / Cc
    Tho_cc = Thi - Qcc / Ch;  Tco_cc = Tci + Qcc / Cc

    dT1_p  = Thi - Tci;       dT2_p  = Tho_p  - Tco_p
    dT1_cc = Thi - Tco_cc;    dT2_cc = Tho_cc - Tci

    return dict(
        Ch=Ch, Cc=Cc, Cmin=Cmin, Cmax=Cmax, c=c, NTU=NTU, Qmax=Qmax,
        eps_p=eps_p, eps_cc=eps_cc, Qp=Qp, Qcc=Qcc,
        Tho_p=Tho_p,   Tco_p=Tco_p,
        Tho_cc=Tho_cc, Tco_cc=Tco_cc,
        dT1_p=dT1_p,   dT2_p=dT2_p,   LMTDp=_lmtd(dT1_p,  dT2_p),
        dT1_cc=dT1_cc, dT2_cc=dT2_cc, LMTDcc=_lmtd(dT1_cc, dT2_cc),
    )


def _sync_cp(key_select, key_cp):
    st.session_state[key_cp] = FLUIDOS[st.session_state[key_select]]


def ax_estilo(ax, titulo):
    ax.set_facecolor(BG_AX)
    ax.set_title(titulo, color=TEXT_C, fontsize=11, pad=10, fontfamily="monospace")
    ax.tick_params(colors=TEXT_C, labelsize=9)
    for sp in ax.spines.values():
        sp.set_color(GRID_C)
    ax.grid(color=GRID_C, linestyle="--", linewidth=0.5, alpha=0.7)
    ax.set_xlabel("Posición adimensional  (x / L)", color=TEXT_C, fontsize=9)
    ax.set_ylabel("Temperatura  (°C)", color=TEXT_C, fontsize=9)


# ─── Sidebar compartido ───────────────────────────────────────────────────
def _sidebar(color: str = "#e0e0e0", nombre: str = "Sistema") -> dict:
    with st.sidebar:
        st.markdown(
            f"<h2 style='font-family:IBM Plex Mono;font-size:0.95rem;"
            f"color:{color};'>⚙️ Parámetros — {nombre}</h2>",
            unsafe_allow_html=True,
        )
        st.markdown("---")

        st.markdown("**Temperaturas de entrada (°C)**")
        Thi = st.slider("T entrada fluido caliente  Th,i", 60.0, 200.0, 120.0, 1.0)
        Tci = st.slider("T entrada fluido frío  Tc,i",     10.0,  80.0,  30.0, 1.0)

        st.markdown("**Gastos másicos (kg/s)**")
        mh = st.slider("Gasto másico caliente  ṁh", 0.1, 5.0, 1.0, 0.1)
        mc = st.slider("Gasto másico frío  ṁc",     0.1, 5.0, 1.0, 0.1)

        st.markdown("**Calores específicos (J/kg·K)**")
        st.selectbox(
            "Fluido caliente", list(FLUIDOS.keys()),
            key="fluido_h",
            on_change=_sync_cp, args=("fluido_h", "Cph"),
        )
        Cph = st.slider("Cp fluido caliente", 500.0, 5000.0,
                        FLUIDOS["Agua"], 50.0, key="Cph")
        st.selectbox(
            "Fluido frío", list(FLUIDOS.keys()),
            key="fluido_c",
            on_change=_sync_cp, args=("fluido_c", "Cpc"),
        )
        Cpc = st.slider("Cp fluido frío", 500.0, 5000.0,
                        FLUIDOS["Agua"], 50.0, key="Cpc")

        st.markdown("**Parámetros de transferencia**")
        U  = st.slider("Coeficiente global  U  (W/m²·K)", 50.0, 3000.0, 500.0, 25.0)
        As = st.slider("Área de transferencia  As  (m²)",  0.5,  30.0,   5.0,  0.5)

        st.markdown("---")
        st.markdown(
            "<div style='font-family:IBM Plex Mono;font-size:0.68rem;color:#444;'>"
            "Çengel &amp; Ghajar (2011) — Cap. 13</div>",
            unsafe_allow_html=True,
        )
    return dict(Thi=Thi, Tci=Tci, mh=mh, mc=mc, Cph=Cph, Cpc=Cpc, U=U, As=As)


# ─── Diagrama esquemático para la pantalla de selección ───────────────────
def _fig_seleccion(modo: str):
    fig, ax = plt.subplots(figsize=(5.5, 2.4))
    fig.patch.set_facecolor(BG_AX)
    ax.set_facecolor(BG_AX)
    ax.set_xlim(0, 10); ax.set_ylim(0, 4); ax.axis("off")

    rect = plt.Rectangle((2, 1.0), 6, 2.0, lw=1.5,
                          edgecolor=GRID_C, facecolor="#161929", zorder=2)
    ax.add_patch(rect)
    ax.text(5, 2.0, "Intercambiador", color="#444", fontsize=7.5,
            ha="center", va="center", fontfamily="monospace", zorder=3)

    arw = dict(arrowstyle="-|>", lw=2.2, mutation_scale=14)

    # Fluido caliente — siempre izquierda → derecha (arriba)
    ax.annotate("", xy=(2.15, 2.6), xytext=(0.4, 2.6),
                arrowprops={**arw, "color": C_HOT}, zorder=4)
    ax.annotate("", xy=(9.6, 2.6), xytext=(7.85, 2.6),
                arrowprops={**arw, "color": C_HOT}, zorder=4)
    ax.text(0.25, 2.6, "Th,i", color=C_HOT, fontsize=7.5,
            va="center", ha="right", fontfamily="monospace")
    ax.text(9.75, 2.6, "Th,o", color=C_HOT, fontsize=7.5,
            va="center", ha="left", fontfamily="monospace")

    if modo == "paralelo":
        c_frio = C_COLD_P
        ax.annotate("", xy=(2.15, 1.4), xytext=(0.4, 1.4),
                    arrowprops={**arw, "color": c_frio}, zorder=4)
        ax.annotate("", xy=(9.6, 1.4), xytext=(7.85, 1.4),
                    arrowprops={**arw, "color": c_frio}, zorder=4)
        ax.text(0.25, 1.4, "Tc,i", color=c_frio, fontsize=7.5,
                va="center", ha="right", fontfamily="monospace")
        ax.text(9.75, 1.4, "Tc,o", color=c_frio, fontsize=7.5,
                va="center", ha="left", fontfamily="monospace")
        ax.text(5, 0.45, "↔ mismo sentido", color="#666", fontsize=7,
                ha="center", fontfamily="monospace")
    else:
        c_frio = C_COLD_C
        ax.annotate("", xy=(0.4, 1.4), xytext=(2.15, 1.4),
                    arrowprops={**arw, "color": c_frio}, zorder=4)
        ax.annotate("", xy=(7.85, 1.4), xytext=(9.6, 1.4),
                    arrowprops={**arw, "color": c_frio}, zorder=4)
        ax.text(9.75, 1.4, "Tc,i", color=c_frio, fontsize=7.5,
                va="center", ha="left", fontfamily="monospace")
        ax.text(0.25, 1.4, "Tc,o", color=c_frio, fontsize=7.5,
                va="center", ha="right", fontfamily="monospace")
        ax.text(5, 0.45, "↔ sentidos opuestos", color="#666", fontsize=7,
                ha="center", fontfamily="monospace")

    plt.tight_layout(pad=0.3)
    return fig


# ─── Diagrama del intercambiador con temperaturas reales ──────────────────
def _fig_intercambiador(modo: str, Thi, Tci, Tho, Tco):
    fig, ax = plt.subplots(figsize=(11, 2.8))
    fig.patch.set_facecolor(BG_FIG)
    ax.set_facecolor(BG_FIG)
    ax.set_xlim(0, 14); ax.set_ylim(0, 4); ax.axis("off")

    rect = plt.Rectangle((3.2, 0.6), 7.6, 2.8, lw=1.8,
                          edgecolor="#3a3f5c", facecolor="#1a1d2e", zorder=2)
    ax.add_patch(rect)
    ax.plot([3.2, 10.8], [2.0, 2.0], color="#2d3147", lw=1.2, ls="--", zorder=3)

    arw = dict(arrowstyle="-|>", lw=2.6, mutation_scale=17)
    c_frio = C_COLD_P if modo == "paralelo" else C_COLD_C

    # Fluido caliente
    ax.annotate("", xy=(3.3, 2.8), xytext=(0.6, 2.8),
                arrowprops={**arw, "color": C_HOT}, zorder=4)
    ax.annotate("", xy=(13.4, 2.8), xytext=(10.7, 2.8),
                arrowprops={**arw, "color": C_HOT}, zorder=4)
    ax.text(0.5, 2.8,
            f"Th,ent\n{Thi:.1f} °C",
            color=C_HOT, fontsize=9, va="center", ha="right",
            fontfamily="monospace", fontweight="bold")
    ax.text(13.5, 2.8,
            f"Th,sal\n{Tho:.1f} °C",
            color=C_HOT, fontsize=9, va="center", ha="left",
            fontfamily="monospace", fontweight="bold")
    ax.text(7.0, 3.1, "── Fluido caliente ──", color=C_HOT, fontsize=7.5,
            ha="center", va="bottom", fontfamily="monospace", alpha=0.65)

    if modo == "paralelo":
        ax.annotate("", xy=(3.3, 1.2), xytext=(0.6, 1.2),
                    arrowprops={**arw, "color": c_frio}, zorder=4)
        ax.annotate("", xy=(13.4, 1.2), xytext=(10.7, 1.2),
                    arrowprops={**arw, "color": c_frio}, zorder=4)
        ax.text(0.5, 1.2,
                f"Tc,ent\n{Tci:.1f} °C",
                color=c_frio, fontsize=9, va="center", ha="right",
                fontfamily="monospace", fontweight="bold")
        ax.text(13.5, 1.2,
                f"Tc,sal\n{Tco:.1f} °C",
                color=c_frio, fontsize=9, va="center", ha="left",
                fontfamily="monospace", fontweight="bold")
        ax.text(7.0, 0.9, "── Fluido frío ──", color=c_frio, fontsize=7.5,
                ha="center", va="top", fontfamily="monospace", alpha=0.65)
    else:
        ax.annotate("", xy=(0.6, 1.2), xytext=(3.3, 1.2),
                    arrowprops={**arw, "color": c_frio}, zorder=4)
        ax.annotate("", xy=(10.7, 1.2), xytext=(13.4, 1.2),
                    arrowprops={**arw, "color": c_frio}, zorder=4)
        ax.text(13.5, 1.2,
                f"Tc,ent\n{Tci:.1f} °C",
                color=c_frio, fontsize=9, va="center", ha="left",
                fontfamily="monospace", fontweight="bold")
        ax.text(0.5, 1.2,
                f"Tc,sal\n{Tco:.1f} °C",
                color=c_frio, fontsize=9, va="center", ha="right",
                fontfamily="monospace", fontweight="bold")
        ax.text(7.0, 0.9, "── Fluido frío ──", color=c_frio, fontsize=7.5,
                ha="center", va="top", fontfamily="monospace", alpha=0.65)

    nombre = "Flujo Paralelo" if modo == "paralelo" else "Flujo a Contracorriente"
    ax.text(7.0, 3.95, nombre, color="#c0c0c0", fontsize=10,
            ha="center", va="top", fontfamily="monospace", fontweight="bold")

    plt.tight_layout(pad=0.2)
    return fig


# ══════════════════════════════════════════════════════════════════════════
# PASO 1 — Pantalla de selección de arreglo
# ══════════════════════════════════════════════════════════════════════════
def pantalla_seleccion():
    st.markdown(
        "<h1 style='font-family:IBM Plex Mono;text-align:center;"
        "background:linear-gradient(90deg,#e63946 30%,#f4a261 100%);"
        "-webkit-background-clip:text;-webkit-text-fill-color:transparent;"
        "margin-bottom:2px;'>Simulador de Intercambiadores de Calor</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align:center;font-family:IBM Plex Mono;color:#555;"
        "font-size:0.78rem;margin-top:0;'>"
        "Çengel &amp; Ghajar (2011), Cap. 13 &nbsp;|&nbsp; Métodos LMTD y NTU-ε</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.markdown(
        "<h3 style='font-family:IBM Plex Mono;text-align:center;"
        "color:#aaa;font-size:1.05rem;margin-bottom:20px;'>"
        "Selecciona el tipo de arreglo a simular</h3>",
        unsafe_allow_html=True,
    )

    col1, gap, col2 = st.columns([10, 1, 10])

    with col1:
        st.markdown(
            "<div class='sel-card'>"
            "<div class='sel-title' style='color:#457b9d;'>⟶ Flujo Paralelo</div>"
            "<div class='sel-desc'>"
            "Ambos fluidos fluyen en el mismo sentido.<br>"
            "Mayor diferencia de temperatura en la entrada."
            "</div></div>",
            unsafe_allow_html=True,
        )
        fig = _fig_seleccion("paralelo")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        if st.button("Simular Flujo Paralelo", use_container_width=True, type="primary"):
            st.session_state.paso = "paralelo"
            st.session_state.comparativa = False
            st.rerun()

    with col2:
        st.markdown(
            "<div class='sel-card'>"
            "<div class='sel-title' style='color:#f4a261;'>⟵⟶ Flujo a Contracorriente</div>"
            "<div class='sel-desc'>"
            "Los fluidos fluyen en sentidos opuestos.<br>"
            "Mayor efectividad — arreglo preferido en la práctica."
            "</div></div>",
            unsafe_allow_html=True,
        )
        fig = _fig_seleccion("contracorriente")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        if st.button("Simular Contracorriente", use_container_width=True, type="primary"):
            st.session_state.paso = "contracorriente"
            st.session_state.comparativa = False
            st.rerun()

    st.markdown("---")
    _, btn_col, _ = st.columns([3, 4, 3])
    with btn_col:
        if st.button("📊 Comparar ambos arreglos directamente",
                     use_container_width=True):
            st.session_state.paso = "paralelo"
            st.session_state.comparativa = True
            st.rerun()

    st.markdown(
        "<p style='text-align:center;font-family:IBM Plex Mono;"
        "font-size:0.70rem;color:#333;margin-top:6px;'>"
        "También puedes explorar la vista comparativa con ambos arreglos lado a lado.</p>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════
# PASO 2 — Simulador del arreglo seleccionado
# ══════════════════════════════════════════════════════════════════════════
def pantalla_simulador(modo: str):
    color_modo = C_COLD_P if modo == "paralelo" else C_COLD_C
    nombre_modo = "Flujo Paralelo" if modo == "paralelo" else "Flujo a Contracorriente"

    p = _sidebar(color=color_modo, nombre=nombre_modo)
    Thi, Tci = p["Thi"], p["Tci"]
    mh,  mc  = p["mh"],  p["mc"]
    Cph, Cpc = p["Cph"], p["Cpc"]
    U,   As  = p["U"],   p["As"]

    # Encabezado + navegación
    hc1, hc2, hc3 = st.columns([5, 2, 1])
    with hc1:
        st.markdown(
            f"<h1 style='font-family:IBM Plex Mono;font-size:1.45rem;"
            f"color:{color_modo};margin-bottom:0;'>{nombre_modo}</h1>",
            unsafe_allow_html=True,
        )
    with hc2:
        if st.button("📊 Comparar arreglos", use_container_width=True):
            st.session_state.comparativa = True
            st.rerun()
    with hc3:
        if st.button("← Volver", use_container_width=True):
            st.session_state.paso = "seleccion"
            st.rerun()

    st.markdown("---")

    if Thi <= Tci:
        st.error(
            f"⚠️  Th,i = {Thi} °C debe ser mayor que Tc,i = {Tci} °C. "
            "Ajusta los sliders del panel lateral."
        )
        st.stop()

    res = calcular(Thi, Tci, mh, mc, Cph, Cpc, U, As)
    Q    = res["Qp"]    if modo == "paralelo" else res["Qcc"]
    eps  = res["eps_p"] if modo == "paralelo" else res["eps_cc"]
    LMTD = res["LMTDp"] if modo == "paralelo" else res["LMTDcc"]
    Tho  = res["Tho_p"] if modo == "paralelo" else res["Tho_cc"]
    Tco  = res["Tco_p"] if modo == "paralelo" else res["Tco_cc"]
    dT1  = res["dT1_p"] if modo == "paralelo" else res["dT1_cc"]
    dT2  = res["dT2_p"] if modo == "paralelo" else res["dT2_cc"]

    # ── Esquema visual del intercambiador ─────────────────────────────────
    fig_ic = _fig_intercambiador(modo, Thi, Tci, Tho, Tco)
    st.pyplot(fig_ic, use_container_width=True)
    plt.close(fig_ic)

    st.markdown("")

    # ── Métricas ──────────────────────────────────────────────────────────
    cols = st.columns(6)
    tarjetas = [
        ("NTU",           f"{res['NTU']:.3f}", ""),
        ("c = Cmin/Cmax", f"{res['c']:.3f}",   ""),
        ("Efectividad ε", f"{eps*100:.1f}",     "%"),
        ("Q transferido", f"{Q/1000:.2f}",      "kW"),
        ("ΔTlm",          f"{LMTD:.2f}",        "°C"),
        ("Qmax posible",  f"{res['Qmax']/1000:.2f}", "kW"),
    ]
    for col, (lbl, val, unit) in zip(cols, tarjetas):
        with col:
            st.markdown(
                f"<div class='metric-card'>"
                f"<div class='metric-value'>{val}"
                f"<span class='metric-unit'> {unit}</span></div>"
                f"<div class='metric-label'>{lbl}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    st.markdown("")

    # ── Perfil de temperatura ──────────────────────────────────────────────
    st.markdown("#### Perfil de temperatura a lo largo del intercambiador")
    x = np.linspace(0.0, 1.0, 300)

    if modo == "paralelo":
        Th_x   = Thi + (Tho - Thi) * x
        Tc_x   = Tci + (Tco - Tci) * x
        c_frio = C_COLD_P
    else:
        Th_x   = Thi + (Tho - Thi) * x
        Tc_x   = Tco + (Tci - Tco) * x   # frío entra por x=1
        c_frio = C_COLD_C

    fig, ax = plt.subplots(figsize=(12, 4.8))
    fig.patch.set_facecolor(BG_FIG)
    ax_estilo(ax, f"Perfil de temperatura — {nombre_modo}")

    ax.plot(x, Th_x, color=C_HOT,   lw=2.5,
            label=f"Fluido caliente   Th,o = {Tho:.1f} °C")
    ax.plot(x, Tc_x, color=c_frio,  lw=2.5,
            label=f"Fluido frío          Tc,o = {Tco:.1f} °C")
    ax.fill_between(x,
                    np.minimum(Tc_x, Th_x), np.maximum(Tc_x, Th_x),
                    alpha=0.07, color=C_HOT)

    for xp, lbl, th_v, tc_v in [
        (0.02, f"ΔT₁ = {dT1:.1f} °C", Th_x[0],  Tc_x[0]),
        (0.98, f"ΔT₂ = {dT2:.1f} °C", Th_x[-1], Tc_x[-1]),
    ]:
        ha = "left" if xp < 0.5 else "right"
        ax.annotate("",
                    xy=(xp, min(tc_v, th_v)), xytext=(xp, max(tc_v, th_v)),
                    arrowprops=dict(arrowstyle="<->", color="#ffdd57", lw=1.5))
        ax.text(xp + (0.02 if xp < 0.5 else -0.02), (th_v + tc_v) / 2,
                lbl, color="#ffdd57", fontsize=8.5, va="center", ha=ha)

    ax.text(0.5, 0.06,
            f"ΔTlm = {LMTD:.2f} °C     Q = {Q/1000:.2f} kW",
            transform=ax.transAxes, color="white", fontsize=10, ha="center",
            bbox=dict(facecolor="#2d3147", edgecolor=c_frio,
                      boxstyle="round,pad=0.45", alpha=0.9))
    ax.legend(fontsize=9, facecolor=BG_AX, edgecolor=GRID_C, labelcolor=TEXT_C)

    plt.tight_layout(pad=1.2)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # ── Tabla detallada ────────────────────────────────────────────────────
    with st.expander("📋 Ver tabla detallada de resultados"):
        data = {
            "Parámetro": [
                "ṁh (kg/s)", "ṁc (kg/s)",
                "Ch = ṁh·Cph (W/K)", "Cc = ṁc·Cpc (W/K)",
                "Cmin (W/K)", "Cmax (W/K)", "c = Cmin/Cmax",
                "NTU = U·As / Cmin",
                "Qmax (kW)", f"ε {nombre_modo} (%)",
                f"Q {nombre_modo} (kW)",
                "ΔTlm (°C)", "Th,o (°C)", "Tc,o (°C)",
            ],
            "Valor": [
                f"{mh:.3f}", f"{mc:.3f}",
                f"{res['Ch']:.2f}", f"{res['Cc']:.2f}",
                f"{res['Cmin']:.2f}", f"{res['Cmax']:.2f}",
                f"{res['c']:.5f}", f"{res['NTU']:.5f}",
                f"{res['Qmax']/1000:.4f}", f"{eps*100:.3f}",
                f"{Q/1000:.4f}",
                f"{LMTD:.3f}", f"{Tho:.3f}", f"{Tco:.3f}",
            ],
        }
        st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════
# PASO 3 — Comparativa de ambos arreglos
# ══════════════════════════════════════════════════════════════════════════
def pantalla_comparativa():
    p = _sidebar(color=C_LIME, nombre="Comparativa")
    Thi, Tci = p["Thi"], p["Tci"]
    mh,  mc  = p["mh"],  p["mc"]
    Cph, Cpc = p["Cph"], p["Cpc"]
    U,   As  = p["U"],   p["As"]

    hc1, hc2 = st.columns([6, 1])
    with hc1:
        st.markdown(
            f"<h1 style='font-family:IBM Plex Mono;font-size:1.45rem;"
            f"color:{C_LIME};margin-bottom:0;'>Comparativa de Arreglos</h1>",
            unsafe_allow_html=True,
        )
    with hc2:
        if st.button("← Volver", use_container_width=True):
            st.session_state.comparativa = False
            st.rerun()

    st.markdown("---")

    if Thi <= Tci:
        st.error(f"⚠️  Th,i = {Thi} °C debe ser mayor que Tc,i = {Tci} °C.")
        st.stop()

    res = calcular(Thi, Tci, mh, mc, Cph, Cpc, U, As)

    # ── Métricas comparativas ──────────────────────────────────────────────
    mc1, mc2, mc3, mc4, mc5, mc6 = st.columns(6)
    tarjetas = [
        ("NTU",             f"{res['NTU']:.3f}",        ""),
        ("c = Cmin/Cmax",   f"{res['c']:.3f}",          ""),
        ("ε  Paralelo",     f"{res['eps_p']*100:.1f}",  "%"),
        ("ε  Contracorriente", f"{res['eps_cc']*100:.1f}", "%"),
        ("Q  Paralelo",     f"{res['Qp']/1000:.2f}",    "kW"),
        ("Q  Contracorriente", f"{res['Qcc']/1000:.2f}", "kW"),
    ]
    col_colors = [TEXT_C, TEXT_C, C_COLD_P, C_COLD_C, C_COLD_P, C_COLD_C]
    for col, (lbl, val, unit), cc in zip(
            [mc1,mc2,mc3,mc4,mc5,mc6], tarjetas, col_colors):
        with col:
            st.markdown(
                f"<div class='metric-card'>"
                f"<div class='metric-value' style='color:{cc}'>{val}"
                f"<span class='metric-unit'> {unit}</span></div>"
                f"<div class='metric-label'>{lbl}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    dq = (res["Qcc"] - res["Qp"]) / 1000
    de = (res["eps_cc"] - res["eps_p"]) * 100
    st.markdown(
        f"<p style='text-align:center;font-family:IBM Plex Mono;"
        f"font-size:0.82rem;color:{C_LIME};margin:10px 0 0 0;'>"
        f"Contracorriente transfiere <strong>{dq:+.3f} kW</strong> adicionales "
        f"({de:+.2f} pp de efectividad)</p>",
        unsafe_allow_html=True,
    )

    st.markdown("")

    tab1, tab2, tab3 = st.tabs([
        "📈 Perfiles de temperatura",
        "📊 Curvas NTU-ε",
        "📋 Tabla resumen",
    ])

    # ── Tab 1: Perfiles ───────────────────────────────────────────────────
    with tab1:
        x = np.linspace(0.0, 1.0, 300)
        Th_p  = Thi + (res["Tho_p"]  - Thi) * x
        Tc_p  = Tci + (res["Tco_p"]  - Tci) * x
        Th_cc = Thi + (res["Tho_cc"] - Thi) * x
        Tc_cc = res["Tco_cc"] + (Tci - res["Tco_cc"]) * x

        fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
        fig.patch.set_facecolor(BG_FIG)

        for ax, titulo, Th_x, Tc_x, c_fr, dT1k, dT2k, Qk, LMTDk in [
            (axes[0], "Flujo Paralelo",
             Th_p, Tc_p, C_COLD_P,
             "dT1_p", "dT2_p", "Qp", "LMTDp"),
            (axes[1], "Flujo a Contracorriente",
             Th_cc, Tc_cc, C_COLD_C,
             "dT1_cc", "dT2_cc", "Qcc", "LMTDcc"),
        ]:
            ax_estilo(ax, titulo)
            Tho_k = res["Tho_p"] if "p" in Qk else res["Tho_cc"]
            Tco_k = res["Tco_p"] if "p" in Qk else res["Tco_cc"]
            ax.plot(x, Th_x, color=C_HOT, lw=2.5,
                    label=f"Fluido caliente   Th,o = {Tho_k:.1f} °C")
            ax.plot(x, Tc_x, color=c_fr,  lw=2.5,
                    label=f"Fluido frío          Tc,o = {Tco_k:.1f} °C")
            ax.fill_between(x, np.minimum(Tc_x, Th_x),
                            np.maximum(Tc_x, Th_x), alpha=0.07, color=C_HOT)
            for xp, lbl, th_v, tc_v in [
                (0.02, f"ΔT₁={res[dT1k]:.1f}°C", Th_x[0],  Tc_x[0]),
                (0.98, f"ΔT₂={res[dT2k]:.1f}°C", Th_x[-1], Tc_x[-1]),
            ]:
                ha = "left" if xp < 0.5 else "right"
                ax.annotate("",
                            xy=(xp, min(tc_v, th_v)),
                            xytext=(xp, max(tc_v, th_v)),
                            arrowprops=dict(arrowstyle="<->",
                                            color="#ffdd57", lw=1.5))
                ax.text(xp + (0.02 if xp < 0.5 else -0.02),
                        (th_v + tc_v) / 2, lbl,
                        color="#ffdd57", fontsize=8, va="center", ha=ha)
            ax.text(0.5, 0.06,
                    f"ΔTlm={res[LMTDk]:.2f}°C   Q={res[Qk]/1000:.2f}kW",
                    transform=ax.transAxes, color="white", fontsize=9,
                    ha="center",
                    bbox=dict(facecolor="#2d3147", edgecolor=c_fr,
                              boxstyle="round,pad=0.4", alpha=0.9))
            ax.legend(fontsize=8, facecolor=BG_AX, edgecolor=GRID_C,
                      labelcolor=TEXT_C)

        plt.tight_layout(pad=1.5)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # ── Tab 2: NTU-ε ──────────────────────────────────────────────────────
    with tab2:
        NTU_v = np.linspace(0.001, 6.5, 500)
        c     = res["c"]
        eps_p_v = (1.0 - np.exp(-NTU_v * (1.0 + c))) / (1.0 + c)
        if abs(c - 1.0) < 1e-6:
            eps_cc_v = NTU_v / (1.0 + NTU_v)
        else:
            eps_cc_v = ((1.0 - np.exp(-NTU_v * (1.0 - c))) /
                        (1.0 - c * np.exp(-NTU_v * (1.0 - c))))

        fig, ax = plt.subplots(figsize=(10, 5.5))
        fig.patch.set_facecolor(BG_FIG)
        ax.set_facecolor(BG_AX)
        ax.plot(NTU_v, eps_p_v  * 100, color=C_COLD_P, lw=2.5,
                label=f"Paralelo  (c = {c:.3f})")
        ax.plot(NTU_v, eps_cc_v * 100, color=C_COLD_C, lw=2.5,
                label=f"Contracorriente  (c = {c:.3f})")
        for eps_op, col in [
            (res["eps_p"],  C_COLD_P),
            (res["eps_cc"], C_COLD_C),
        ]:
            ax.scatter(res["NTU"], eps_op * 100, color=col, s=130,
                       zorder=6, edgecolors="white", lw=1.8)
        ax.axvline(res["NTU"], color="white", lw=0.8, ls=":", alpha=0.45)
        ax.text(res["NTU"] + 0.09, res["eps_p"]  * 100 + 1.0,
                f"{res['eps_p']*100:.1f}%",  color=C_COLD_P,
                fontsize=9, fontfamily="monospace")
        ax.text(res["NTU"] + 0.09, res["eps_cc"] * 100 - 3.5,
                f"{res['eps_cc']*100:.1f}%", color=C_COLD_C,
                fontsize=9, fontfamily="monospace")
        ax.set_xlim(0, 6.5); ax.set_ylim(0, 105)
        ax.set_xlabel("NTU = U·As / Cmin", color=TEXT_C, fontsize=10)
        ax.set_ylabel("Efectividad  ε  (%)", color=TEXT_C, fontsize=10)
        ax.set_title(f"ε vs. NTU  —  c = Cmin/Cmax = {c:.4f}",
                     color=TEXT_C, fontsize=11, fontfamily="monospace", pad=10)
        ax.tick_params(colors=TEXT_C)
        for sp in ax.spines.values():
            sp.set_color(GRID_C)
        ax.grid(color=GRID_C, linestyle="--", linewidth=0.5, alpha=0.7)
        ax.legend(fontsize=9, facecolor=BG_AX, edgecolor=GRID_C,
                  labelcolor=TEXT_C, loc="lower right")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # ── Tab 3: Tabla resumen ───────────────────────────────────────────────
    with tab3:
        data = {
            "Parámetro": [
                "ṁh  (kg/s)", "ṁc  (kg/s)",
                "Ch = ṁh·Cph  (W/K)", "Cc = ṁc·Cpc  (W/K)",
                "Cmin  (W/K)", "Cmax  (W/K)", "c = Cmin/Cmax",
                "NTU = U·As / Cmin", "Qmax  (kW)",
                "ε  Paralelo  (%)", "ε  Contracorriente  (%)",
                "Q  Paralelo  (kW)", "Q  Contracorriente  (kW)",
                "ΔTlm  Paralelo  (°C)", "ΔTlm  Contracorriente  (°C)",
                "Th,o  Paralelo  (°C)", "Th,o  Contracorriente  (°C)",
                "Tc,o  Paralelo  (°C)", "Tc,o  Contracorriente  (°C)",
            ],
            "Valor": [
                f"{mh:.3f}", f"{mc:.3f}",
                f"{res['Ch']:.2f}", f"{res['Cc']:.2f}",
                f"{res['Cmin']:.2f}", f"{res['Cmax']:.2f}",
                f"{res['c']:.5f}", f"{res['NTU']:.5f}",
                f"{res['Qmax']/1000:.4f}",
                f"{res['eps_p']*100:.3f}", f"{res['eps_cc']*100:.3f}",
                f"{res['Qp']/1000:.4f}", f"{res['Qcc']/1000:.4f}",
                f"{res['LMTDp']:.3f}", f"{res['LMTDcc']:.3f}",
                f"{res['Tho_p']:.3f}", f"{res['Tho_cc']:.3f}",
                f"{res['Tco_p']:.3f}", f"{res['Tco_cc']:.3f}",
            ],
        }
        st.dataframe(pd.DataFrame(data), use_container_width=True,
                     hide_index=True)
        st.markdown(
            f"> **Ganancia de efectividad al usar contracorriente:** "
            f"{de:.2f} pp  →  ΔQ = {dq:+.3f} kW adicionales"
        )


# ══════════════════════════════════════════════════════════════════════════
# ENRUTADOR
# ══════════════════════════════════════════════════════════════════════════
if st.session_state.paso == "seleccion":
    pantalla_seleccion()
elif st.session_state.comparativa:
    pantalla_comparativa()
else:
    pantalla_simulador(st.session_state.paso)
