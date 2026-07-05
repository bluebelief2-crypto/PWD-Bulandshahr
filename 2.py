"""
Stochastic EOQ Model — PWD Bulandshahr Procurement Risk Tool
=============================================================

Models material procurement (e.g. TMT steel, VG-40 bitumen) under
lead-time uncertainty, following the Hadley-Whitin backorder
formulation of the stochastic Economic Order Quantity (EOQ) model:

    E[TC](Q, k) = (D/Q) * S  +  h * (Q/2 + k*sigma_L)  +  (D/Q) * pi * sigma_L * psi(k)

where psi(k) = phi(k) - k*(1 - Phi(k)) is the standardized normal loss
function (expected number of units short per order cycle).

Runs 100% offline — no network calls, no external APIs. Everything is
computed locally with numpy/scipy; charts are rendered with matplotlib.

Usage:
    pip install -r requirements.txt
    streamlit run app.py
"""

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy.stats import norm, gamma as gamma_dist, lognorm

st.set_page_config(page_title="PWD Bulandshahr — Stochastic EOQ", layout="wide")

# -----------------------------------------------------------------------
# Core model functions
# -----------------------------------------------------------------------

def std_normal_loss(k):
    """Standardized normal loss function psi(k) = phi(k) - k*(1 - Phi(k))."""
    return norm.pdf(k) - k * (1 - norm.cdf(k))


def deterministic_eoq(D, S, h):
    """Classical EOQ ignoring lead-time uncertainty."""
    return np.sqrt(2 * D * S / h)


def solve_stochastic_eoq(D, S, h, pi, sigma_L, max_iter=100, tol=1e-6):
    """
    Hadley-Whitin iterative algorithm for the joint optimal order
    quantity Q* and safety factor k* under stochastic lead-time demand.

    Returns a dict with Q, k, safety_stock, reorder_point components,
    total cost, and the iteration history (for display/debugging).
    """
    Q = deterministic_eoq(D, S, h)
    k = 0.0
    history = []

    for i in range(max_iter):
        # Step 1: given Q, find k from the marginal cost balance condition
        # 1 - Phi(k) = h*Q / (D*pi)
        rhs = h * Q / (D * pi)
        rhs = min(max(rhs, 1e-9), 1 - 1e-9)  # keep in valid probability range
        k_new = norm.ppf(1 - rhs)

        # Step 2: given k, update Q
        loss = std_normal_loss(k_new)
        Q_new = np.sqrt(2 * D * (S + pi * sigma_L * loss) / h)

        history.append({"iter": i + 1, "Q": Q_new, "k": k_new})

        if abs(Q_new - Q) < tol and abs(k_new - k) < tol:
            Q, k = Q_new, k_new
            break
        Q, k = Q_new, k_new

    safety_stock = k * sigma_L
    total_cost = expected_total_cost(Q, k, D, S, h, pi, sigma_L)
    service_level = norm.cdf(k)

    return {
        "Q": Q,
        "k": k,
        "safety_stock": safety_stock,
        "service_level": service_level,
        "total_cost": total_cost,
        "history": pd.DataFrame(history),
    }


def expected_total_cost(Q, k, D, S, h, pi, sigma_L):
    """E[TC](Q, k) per the stochastic EOQ formula."""
    ordering = (D / Q) * S
    holding = h * (Q / 2 + k * sigma_L)
    shortage = (D / Q) * pi * sigma_L * std_normal_loss(k)
    return ordering + holding + shortage


def cost_curve(D, S, h, pi, sigma_L, k_fixed, q_range):
    return np.array([expected_total_cost(q, k_fixed, D, S, h, pi, sigma_L) for q in q_range])


# -----------------------------------------------------------------------
# Monte Carlo validation
# -----------------------------------------------------------------------

def simulate_cycles(D, Q, k, sigma_L, mean_L, dist, n_cycles=20000, seed=42):
    """
    Simulate n_cycles order cycles with random lead-time demand and
    estimate the empirical stockout probability and average shortage,
    to check the analytical result against a direct simulation.
    """
    rng = np.random.default_rng(seed)
    daily_demand = D / 365.0
    reorder_point = daily_demand * mean_L + k * sigma_L

    if dist == "Normal":
        lead_times = rng.normal(mean_L, max(mean_L * 0.15, 1e-6), n_cycles)
    elif dist == "Lognormal":
        # match mean/std via method-of-moments for lognormal
        sigma2 = np.log(1 + (mean_L * 0.4 / mean_L) ** 2)
        mu = np.log(mean_L) - sigma2 / 2
        lead_times = rng.lognormal(mu, np.sqrt(sigma2), n_cycles)
    else:  # Gamma / Frechet-like heavy tail proxy
        shape = 2.0
        scale = mean_L / shape
        lead_times = rng.gamma(shape, scale, n_cycles) * (1 + rng.pareto(3.0, n_cycles) * 0.05)

    lead_times = np.clip(lead_times, 0.1, None)
    demand_during_lead_time = daily_demand * lead_times
    shortage = np.maximum(demand_during_lead_time - reorder_point, 0)

    stockout_prob = np.mean(shortage > 0)
    avg_shortage = np.mean(shortage)
    avg_lead_time = np.mean(lead_times)

    return {
        "stockout_prob": stockout_prob,
        "avg_shortage": avg_shortage,
        "avg_lead_time": avg_lead_time,
        "lead_times": lead_times,
        "reorder_point": reorder_point,
    }


# -----------------------------------------------------------------------
# Sidebar — inputs
# -----------------------------------------------------------------------

st.sidebar.header("Material & Cost Parameters")

material = st.sidebar.selectbox(
    "Material",
    ["TMT Steel (bridge project)", "VG-40 Bitumen (paving)", "Cement (OPC 43/53)", "Custom"],
)

presets = {
    "TMT Steel (bridge project)": dict(D=1200.0, S=45000.0, h=3500.0, pi=9000.0, mean_L=25.0, sigma_L=6.0),
    "VG-40 Bitumen (paving)": dict(D=800.0, S=60000.0, h=5200.0, pi=15000.0, mean_L=20.0, sigma_L=9.0),
    "Cement (OPC 43/53)": dict(D=3000.0, S=20000.0, h=1200.0, pi=4000.0, mean_L=10.0, sigma_L=3.0),
    "Custom": dict(D=1000.0, S=40000.0, h=3000.0, pi=8000.0, mean_L=20.0, sigma_L=5.0),
}
p = presets[material]

D = st.sidebar.number_input("Annual demand D (tonnes/year)", min_value=1.0, value=p["D"])
S = st.sidebar.number_input("Ordering cost S (₹ per order)", min_value=1.0, value=p["S"])
h = st.sidebar.number_input("Holding cost h (₹ per tonne per year)", min_value=1.0, value=p["h"])
pi = st.sidebar.number_input("Shortage penalty π (₹ per tonne short)", min_value=1.0, value=p["pi"])

st.sidebar.header("Lead Time Under Disruption")
mean_L = st.sidebar.number_input("Mean lead time (days)", min_value=1.0, value=p["mean_L"])

crisis_mode = st.sidebar.toggle(
    "Crisis scenario (Suez / Red Sea / Hormuz disruption)",
    value=False,
    help="Multiplies lead-time variability to reflect a geopolitical shock, "
         "e.g. Cape of Good Hope rerouting adding 14-20 days of uncertainty.",
)
sigma_L_base = st.sidebar.number_input("Lead time std. dev., normal conditions (days)", min_value=0.1, value=p["sigma_L"])
crisis_multiplier = st.sidebar.slider("Crisis variability multiplier", 1.0, 6.0, 3.0, 0.5, disabled=not crisis_mode)
sigma_L = sigma_L_base * (crisis_multiplier if crisis_mode else 1.0)

lead_time_dist = st.sidebar.selectbox(
    "Lead time distribution",
    ["Normal", "Lognormal", "Gamma (heavy-tailed / crisis)"],
    index=2 if crisis_mode else 0,
)
dist_key = "Gamma" if lead_time_dist.startswith("Gamma") else lead_time_dist

n_cycles = st.sidebar.slider("Monte Carlo cycles", 2000, 50000, 20000, 1000)

# -----------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------

st.title("Stochastic EOQ Model — PWD Bulandshahr Procurement")
st.caption(
    "Illustrative decision-support tool built for the internship study on global supply "
    "chain disruptions and PWD infrastructure procurement. All figures below are "
    "user-configurable inputs, not verified departmental data."
)

if crisis_mode:
    st.warning(
        f"Crisis scenario active — lead-time variability increased {crisis_multiplier:g}x "
        f"over normal conditions (σL = {sigma_L:.1f} days)."
    )

# -----------------------------------------------------------------------
# Compute analytical solution
# -----------------------------------------------------------------------

result = solve_stochastic_eoq(D, S, h, pi, sigma_L)
Q_star, k_star = result["Q"], result["k"]
Q_det = deterministic_eoq(D, S, h)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Deterministic EOQ (Q₀)", f"{Q_det:,.1f} t")
col2.metric("Stochastic optimal order qty (Q*)", f"{Q_star:,.1f} t", f"{Q_star - Q_det:+.1f} t vs Q₀")
col3.metric("Safety stock (k·σL)", f"{result['safety_stock']:,.1f} t")
col4.metric("Implied service level Φ(k)", f"{result['service_level']*100:.1f}%")

st.markdown(
    f"**Optimal safety factor k\\* = {k_star:.3f}** — the department should carry "
    f"**{result['safety_stock']:.1f} tonnes** of extra stock beyond expected lead-time demand, "
    f"at a total expected cost of **₹{result['total_cost']:,.0f}/year** "
    f"(ordering + holding + shortage combined)."
)

# -----------------------------------------------------------------------
# Cost curve vs Q
# -----------------------------------------------------------------------

st.subheader("Total Cost vs. Order Quantity")
q_range = np.linspace(max(Q_star * 0.2, 1), Q_star * 2.5, 200)
costs_stochastic = cost_curve(D, S, h, pi, sigma_L, k_star, q_range)
costs_det_k0 = cost_curve(D, S, h, pi, sigma_L, 0.0, q_range)

fig1, ax1 = plt.subplots(figsize=(9, 4.2))
ax1.plot(q_range, costs_stochastic, label="E[TC] with optimal safety stock (k*)", color="#1F3864", linewidth=2)
ax1.plot(q_range, costs_det_k0, label="E[TC] with zero safety stock (k=0)", color="#C00000", linestyle="--")
ax1.axvline(Q_star, color="#1F3864", linestyle=":", alpha=0.7)
ax1.axvline(Q_det, color="gray", linestyle=":", alpha=0.7)
ax1.scatter([Q_star], [result["total_cost"]], color="#1F3864", zorder=5)
ax1.set_xlabel("Order quantity Q (tonnes)")
ax1.set_ylabel("Expected total cost (₹/year)")
ax1.legend()
ax1.grid(alpha=0.3)
st.pyplot(fig1)

# -----------------------------------------------------------------------
# Sensitivity: safety stock vs lead time variability
# -----------------------------------------------------------------------

st.subheader("Safety Stock Sensitivity to Lead-Time Variability")
sigma_range = np.linspace(sigma_L_base * 0.5, sigma_L_base * 6, 40)
safety_stocks, service_levels = [], []
for s in sigma_range:
    r = solve_stochastic_eoq(D, S, h, pi, s)
    safety_stocks.append(r["safety_stock"])
    service_levels.append(r["service_level"])

fig2, ax2 = plt.subplots(figsize=(9, 4.2))
ax2.plot(sigma_range, safety_stocks, color="#2E5395", linewidth=2)
ax2.axvline(sigma_L_base, color="gray", linestyle=":", label="Normal conditions")
if crisis_mode:
    ax2.axvline(sigma_L, color="#C00000", linestyle=":", label="Current crisis setting")
ax2.set_xlabel("Lead-time standard deviation σL (days)")
ax2.set_ylabel("Optimal safety stock (tonnes)")
ax2.legend()
ax2.grid(alpha=0.3)
st.pyplot(fig2)
st.caption(
    "As lead-time uncertainty rises — the effect of a Suez/Hormuz-style disruption — "
    "the required safety stock grows faster than linearly, which is the mathematical "
    "case for pre-positioning stock ahead of the monsoon/tender season rather than "
    "relying on just-in-time delivery."
)

# -----------------------------------------------------------------------
# Monte Carlo validation
# -----------------------------------------------------------------------

st.subheader("Monte Carlo Validation")
st.write(
    "Simulates thousands of independent order cycles with random lead times drawn from "
    "the selected distribution, to check the analytical stockout risk against direct simulation."
)

sim = simulate_cycles(D, Q_star, k_star, sigma_L, mean_L, dist_key, n_cycles=n_cycles)
analytical_stockout_risk = 1 - result["service_level"]

sc1, sc2, sc3 = st.columns(3)
sc1.metric("Analytical stockout risk (1-Φ(k))", f"{analytical_stockout_risk*100:.2f}%")
sc2.metric("Simulated stockout probability", f"{sim['stockout_prob']*100:.2f}%")
sc3.metric("Simulated mean lead time", f"{sim['avg_lead_time']:.1f} days")

fig3, ax3 = plt.subplots(figsize=(9, 3.8))
ax3.hist(sim["lead_times"], bins=60, color="#2E5395", alpha=0.75)
ax3.axvline(mean_L, color="gray", linestyle=":", label="Mean lead time (input)")
ax3.axvline(sim["avg_lead_time"], color="#C00000", linestyle=":", label="Simulated mean")
ax3.set_xlabel("Lead time (days)")
ax3.set_ylabel("Simulated cycles")
ax3.set_title(f"Simulated lead-time distribution — {lead_time_dist}")
ax3.legend()
st.pyplot(fig3)

st.caption(
    "Analytical and simulated stockout figures should be close under Normal lead times; "
    "they diverge more under Gamma/heavy-tailed settings, which is expected since the "
    "closed-form model assumes normality — the simulation is the more honest picture "
    "of risk once lead times develop the fat tails typical of a real supply shock."
)

# -----------------------------------------------------------------------
# Summary table for report use
# -----------------------------------------------------------------------

st.subheader("Summary (for copying into a report)")
summary_df = pd.DataFrame({
    "Metric": [
        "Deterministic EOQ (Q0)", "Stochastic optimal order qty (Q*)",
        "Optimal safety factor (k*)", "Safety stock (k*·σL)",
        "Implied service level", "Expected total cost (₹/year)",
        "Lead time distribution", "Lead time σL used (days)",
    ],
    "Value": [
        f"{Q_det:,.2f} t", f"{Q_star:,.2f} t", f"{k_star:.3f}",
        f"{result['safety_stock']:,.2f} t", f"{result['service_level']*100:.1f}%",
        f"₹{result['total_cost']:,.0f}", lead_time_dist, f"{sigma_L:.2f}",
    ],
})
st.table(summary_df)

st.download_button(
    "Download summary as CSV",
    summary_df.to_csv(index=False).encode("utf-8"),
    file_name="pwd_eoq_summary.csv",
    mime="text/csv",
)