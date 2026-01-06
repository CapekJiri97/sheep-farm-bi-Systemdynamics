import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass

# --- KONFIGURACE A STYL ---
st.set_page_config(page_title="Sheep Farm 9.0 BI", layout="wide", page_icon="🚜")
plt.style.use('dark_background')

# --- POMOCNÉ FUNKCE ---
def get_stochastic_value(mean, std, min_val=0.0):
    val = np.random.normal(mean, std)
    return max(min_val, val)

@dataclass
class HardDataConfig:
    # --- UŽIVATELSKÉ VSTUPY ---
    sim_years: int
    land_area: float
    barn_capacity: int
    capital: float
    price_meat_avg: float
    enable_forecasting: bool
    safety_margin: float

    # --- BIOLOGIE (Hard Data) ---
    carrying_capacity_mean: float = 7.0
    carrying_capacity_std: float = 1.0
    fertility_mean: float = 1.5
    fertility_std: float = 0.3
    mortality_lamb_mean: float = 0.12
    mortality_lamb_std: float = 0.04
    mortality_ewe_mean: float = 0.04
    mortality_ewe_std: float = 0.01
    feed_intake_mean: float = 2.2 # kg sušiny/den
    feed_intake_std: float = 0.3
    
    # --- EKONOMIKA ---
    price_meat_std: float = 8.0 
    
    # Náklady na krmivo
    cost_feed_own_mean: float = 2.5
    cost_feed_own_std: float = 0.5
    cost_feed_market_mean: float = 8.0
    cost_feed_market_std: float = 2.0
    
    # Dotace (SZIF)
    subsidy_ha_mean: float = 4500.0
    subsidy_ha_std: float = 200.0
    subsidy_sheep_mean: float = 580.0
    subsidy_sheep_std: float = 50.0
    
    # Režie a Fixní
    fixed_cost_ewe_mean: float = 900.0 
    fixed_cost_ewe_std: float = 100.0
    overhead_base_year: float = 40000.0 
    
    # Šoky (Rizika)
    shock_prob_daily: float = 0.005 # 0.5% šance denně
    shock_cost_mean: float = 15000.0
    shock_cost_std: float = 5000.0

class FarmBIModel:
    def __init__(self, cfg: HardDataConfig):
        self.cfg = cfg
        self.date = pd.Timestamp("2025-01-01")
        self.ewes = 20
        self.lambs = 0
        self.cash = cfg.capital
        self.bcs = 3.0 
        
        # Logy pro BI
        self.history = []
        self.feed_log = {"Grazing": 0, "Stored": 0, "Market": 0}

        self.grass_curve = {
            1:0, 2:0, 3:0.1, 4:0.6, 5:1.2, 6:1.1, 
            7:0.7, 8:0.5, 9:0.8, 10:0.4, 11:0.1, 12:0
        }

    def _get_seasonal_overhead(self, month):
        base_daily = self.cfg.overhead_base_year / 365
        if month in [6, 7, 8]: return base_daily * 1.5
        elif month in [1, 2, 12]: return base_daily * 1.3
        else: return base_daily * 0.8

    def _perform_forecast(self):
        winter_feed = 180 * self.cfg.feed_intake_mean * self.cfg.cost_feed_market_mean * self.ewes
        winter_overhead = (self.cfg.overhead_base_year / 2) * 1.2
        return (winter_feed + winter_overhead) * (1.0 + self.cfg.safety_margin)

    def step(self):
        month = self.date.month
        current_meat_price = get_stochastic_value(self.cfg.price_meat_avg, self.cfg.price_meat_std)
        
        # --- 1. BIOLOGIE & KRMENÍ ---
        total_sheep = self.ewes + self.lambs
        current_intake = get_stochastic_value(self.cfg.feed_intake_mean, self.cfg.feed_intake_std)
        feed_demand = total_sheep * current_intake
        
        season_factor = self.grass_curve[month]
        available_grass = self.cfg.land_area * 30.0 * season_factor * np.random.normal(1.0, 0.2)
        
        feed_cost = 0.0
        feed_source = ""
        
        if available_grass >= feed_demand:
            feed_cost = feed_demand * get_stochastic_value(self.cfg.cost_feed_own_mean, 0.5)
            self.bcs = min(4.0, self.bcs + 0.003)
            feed_source = "Grazing"
        else:
            if month in [12,1,2,3]:
                # Zima - Vlastní zásoby (simulované cenou)
                price = get_stochastic_value(self.cfg.cost_feed_own_mean * 1.5, 0.5)
                feed_cost = feed_demand * price
                self.bcs = max(2.0, self.bcs - 0.001)
                feed_source = "Stored"
            else:
                # Léto/Sucho - Nákup z trhu
                price = get_stochastic_value(self.cfg.cost_feed_market_mean, self.cfg.cost_feed_market_std)
                # Koupíme jen 80%, zbytek hubnou
                feed_cost = (available_grass * self.cfg.cost_feed_own_mean) + ((feed_demand - available_grass) * 0.8 * price)
                self.bcs = max(1.5, self.bcs - 0.01)
                feed_source = "Market"

        self.feed_log[feed_source] += 1

        # Mortalita
        mort_prob = (self.cfg.mortality_ewe_mean / 365) * (2 if self.bcs < 2.5 else 1)
        deaths = np.random.binomial(total_sheep, mort_prob)
        if total_sheep > 0: self.ewes = max(0, self.ewes - deaths)
        
        # Reprodukce
        new_lambs = 0
        if month == 3 and self.date.day == 15:
            new_lambs = int(self.ewes * get_stochastic_value(self.cfg.fertility_mean, 0.3))
            self.lambs += new_lambs
            self.bcs -= 0.5

        # --- 2. CASHFLOW ---
        income_sales = 0.0
        income_subsidy = 0.0
        daily_overhead = self._get_seasonal_overhead(month)
        
        # Šoky
        shock_cost = 0.0
        if np.random.random() < self.cfg.shock_prob_daily:
            shock_cost = get_stochastic_value(self.cfg.shock_cost_mean, self.cfg.shock_cost_std)
        
        # Dotace
        if month == 11 and self.date.day == 20: 
            income_subsidy += ((self.cfg.land_area * self.cfg.subsidy_ha_mean) + (self.ewes * self.cfg.subsidy_sheep_mean)) * 0.7
        if month == 4 and self.date.day == 20: 
            income_subsidy += ((self.cfg.land_area * self.cfg.subsidy_ha_mean) + (self.ewes * self.cfg.subsidy_sheep_mean)) * 0.3

        # Prodej
        if month == 10 and self.date.day == 15:
            lamb_income = self.lambs * 35.0 * current_meat_price
            
            cull_extra = 0
            if self.cfg.enable_forecasting:
                liability = self._perform_forecast()
                balance = self.cash + lamb_income - liability
                if balance < 0:
                    cull_extra = int(abs(balance) / (60 * current_meat_price)) + 1
            
            cull = int(self.ewes * 0.15) + cull_extra
            sold_ewes = min(cull, self.ewes)
            
            income_sales += lamb_income
            income_sales += sold_ewes * 60.0 * (current_meat_price * 0.8)
            
            self.ewes = (self.ewes - sold_ewes) + int(self.ewes * 0.15)
            self.lambs = 0
            # Roční fixní veterinární poplatky
            feed_cost += self.ewes * get_stochastic_value(self.cfg.fixed_cost_ewe_mean, 50)

        total_income = income_sales + income_subsidy
        total_expense = feed_cost + daily_overhead + shock_cost
        self.cash += total_income - total_expense
        
        self.history.append({
            "Date": self.date,
            "Cash": self.cash,
            "Total Sheep": self.ewes + self.lambs,
            "BCS": self.bcs,
            # Breakdown pro BI grafy
            "Income_Sales": income_sales,
            "Income_Subsidy": income_subsidy,
            "Exp_Feed": feed_cost,
            "Exp_Overhead": daily_overhead,
            "Exp_Shock": shock_cost,
            "Meat_Price": current_meat_price
        })
        self.date += pd.Timedelta(days=1)

    def run(self):
        for _ in range(self.cfg.sim_years * 365): self.step()
        return pd.DataFrame(self.history).set_index("Date")

# --- SIDEBAR UI (KOMPLETNÍ VÝPIS) ---
with st.sidebar:
    st.title("🚜 Sheep Farm BI")
    
    st.header("🎛️ Vstupy")
    area = st.slider("Plocha (ha)", 5.0, 100.0, 15.0)
    barn = st.slider("Ovčín (ks)", 20, 500, 100)
    cap = st.number_input("Kapitál (Kč)", value=200000)
    meat_price = st.slider("Cena masa (Kč)", 40.0, 120.0, 75.0)
    use_forecast = st.toggle("Cashflow Plánovač", value=True)
    
    st.markdown("---")
    st.header("📋 Parametry Modelu (Audit)")
    st.caption("Hodnoty: Střední hodnota ± Směrodatná odchylka")

    # Instance pro výpis defaultů
    d = HardDataConfig(0,0,0,0,0,0,0) 

    with st.expander("🧬 1. Biologická Data", expanded=False):
        st.code(f"""
Carrying Capacity: {d.carrying_capacity_mean} ± {d.carrying_capacity_std} ks/ha
Fertility Rate:    {d.fertility_mean} ± {d.fertility_std}
Mortality Lamb:    {d.mortality_lamb_mean*100:.0f}% ± {d.mortality_lamb_std*100:.0f}%
Mortality Ewe:     {d.mortality_ewe_mean*100:.0f}% ± {d.mortality_ewe_std*100:.0f}%
Feed Intake:       {d.feed_intake_mean} ± {d.feed_intake_std} kg/den
        """)

    with st.expander("💰 2. Ekonomická Data", expanded=False):
        st.code(f"""
Meat Price Vol:    ± {d.price_meat_std} Kč
SAPS Subsidy:      {d.subsidy_ha_mean:,.0f} ± {d.subsidy_ha_std} Kč/ha
VDJ Subsidy:       {d.subsidy_sheep_mean:,.0f} ± {d.subsidy_sheep_std} Kč/ks
        """)

    with st.expander("📉 3. Nákladová Data", expanded=False):
        st.code(f"""
Feed (Own):        {d.cost_feed_own_mean} ± {d.cost_feed_own_std} Kč/kg
Feed (Market):     {d.cost_feed_market_mean} ± {d.cost_feed_market_std} Kč/kg
Fixed Cost/Ewe:    {d.fixed_cost_ewe_mean} ± {d.fixed_cost_ewe_std} Kč
Overhead Base:     {d.overhead_base_year:,.0f} Kč/rok
Shock Prob:        {d.shock_prob_daily*100:.1f}% / den
Shock Cost:        {d.shock_cost_mean:,.0f} ± {d.shock_cost_std:,.0f} Kč
        """)

# --- SPUŠTĚNÍ ---
cfg = HardDataConfig(
    sim_years=5, land_area=area, barn_capacity=barn, capital=cap,
    price_meat_avg=meat_price, enable_forecasting=use_forecast, safety_margin=0.2
)
model = FarmBIModel(cfg)
df = model.run()

# --- BI DASHBOARD ---
st.title("📊 Farm Management Dashboard")

# 1. KPI ROW
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
final_cash = df["Cash"].iloc[-1]
kpi1.metric("Likvidita (Cash)", f"{final_cash:,.0f} Kč", delta=f"{final_cash-cap:,.0f}")
kpi2.metric("Stav Stáda", int(df["Total Sheep"].iloc[-1]), delta=int(df["Total Sheep"].iloc[-1]-20))
kpi3.metric("Náklady na Havárie", f"{df['Exp_Shock'].sum():,.0f} Kč")
kpi4.metric("Celkové Dotace", f"{df['Income_Subsidy'].sum():,.0f} Kč")

# 2. FINANČNÍ STRUKTURA (Stacked Area)
st.subheader("💸 Struktura Cashflow")
col_chart, col_pie = st.columns([3, 1])

with col_chart:
    # Agregace po měsících pro hezčí graf
    df_monthly = df.resample("M").sum()
    
    fig, ax = plt.subplots(figsize=(10, 4))
    # Příjmy (Positive)
    ax.bar(df_monthly.index, df_monthly["Income_Sales"], label="Prodej Masa", color="#2ecc71", width=20)
    ax.bar(df_monthly.index, df_monthly["Income_Subsidy"], bottom=df_monthly["Income_Sales"], label="Dotace", color="#f1c40f", width=20)
    
    # Výdaje (Negative)
    ax.bar(df_monthly.index, -df_monthly["Exp_Feed"], label="Krmivo", color="#e67e22", width=20)
    ax.bar(df_monthly.index, -df_monthly["Exp_Overhead"], bottom=-df_monthly["Exp_Feed"], label="Režie", color="#3498db", width=20)
    ax.bar(df_monthly.index, -df_monthly["Exp_Shock"], bottom=-(df_monthly["Exp_Feed"]+df_monthly["Exp_Overhead"]), label="Havárie", color="#e74c3c", width=20)
    
    ax.axhline(0, color="white", linewidth=0.5)
    ax.set_ylabel("CZK")
    ax.legend(loc="upper left", bbox_to_anchor=(1,1))
    ax.grid(True, alpha=0.1)
    st.pyplot(fig)

with col_pie:
    st.markdown("**Rozložení Výdajů**")
    total_exp = df[["Exp_Feed", "Exp_Overhead", "Exp_Shock"]].sum()
    fig_pie, ax_pie = plt.subplots()
    ax_pie.pie(total_exp, labels=["Krmivo", "Režie", "Šoky"], autopct='%1.1f%%', colors=["#e67e22", "#3498db", "#e74c3c"], startangle=90)
    st.pyplot(fig_pie)

# 3. SEZÓNNÍ ANALÝZA & TRH
st.subheader("📅 Sezónnost a Trh")
col_seas, col_market = st.columns(2)

with col_seas:
    st.markdown("**Průměrné měsíční Cashflow**")
    # Získáme průměrné saldo pro každý měsíc v roce
    df["Month"] = df.index.month
    df["Net_Flow"] = (df["Income_Sales"] + df["Income_Subsidy"]) - (df["Exp_Feed"] + df["Exp_Overhead"] + df["Exp_Shock"])
    seasonal_data = df.groupby("Month")["Net_Flow"].mean()
    
    fig_s, ax_s = plt.subplots(figsize=(6, 3))
    colors = ["red" if x < 0 else "green" for x in seasonal_data]
    ax_s.bar(seasonal_data.index, seasonal_data, color=colors)
    ax_s.set_xticks(range(1, 13))
    ax_s.set_ylabel("Průměrný Zisk/Ztráta (CZK)")
    ax_s.grid(True, alpha=0.2)
    st.pyplot(fig_s)
    st.caption("Červená = měsíce, kdy farma 'krvácí' peníze (zima/jaro).")

with col_market:
    st.markdown("**Volatilita Tržní Ceny**")
    fig_m, ax_m = plt.subplots(figsize=(6, 3))
    sns.histplot(df["Meat_Price"], kde=True, ax=ax_m, color="purple", bins=30)
    ax_m.axvline(cfg.price_meat_avg, color="white", linestyle="--", label="Průměr")
    ax_m.set_xlabel("Cena masa (Kč/kg)")
    st.pyplot(fig_m)

# 4. PROVOZNÍ EFEKTIVITA
st.subheader("🌾 Efektivita Krmení")
feed_counts = pd.Series(model.feed_log)
st.bar_chart(feed_counts)
st.caption("**Grazing:** Pastva (levné) | **Stored:** Zásoby (střední) | **Market:** Nákup (drahé - krize)")