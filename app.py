import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass

# --- CONFIGURATION AND STYLE ---
st.set_page_config(page_title="Sheep Farm 9.0 BI", layout="wide", page_icon="🚜")
plt.style.use('dark_background')

# --- HELPER FUNCTIONS ---
def get_stochastic_value(mean, std, min_val=0.0):
    val = np.random.normal(mean, std)
    return max(min_val, val)

@dataclass
class HardDataConfig:
    # --- USER INPUTS ---
    sim_years: int
    land_area: float
    meadow_share: float  # NEW: % of land reserved only for hay production
    barn_capacity: int
    capital: float
    price_meat_avg: float
    enable_forecasting: bool
    safety_margin: float

    # --- BIOLOGY (Hard Data) ---
    carrying_capacity_mean: float = 7.0
    carrying_capacity_std: float = 1.0
    fertility_mean: float = 1.5
    fertility_std: float = 0.3
    mortality_lamb_mean: float = 0.12
    mortality_lamb_std: float = 0.04
    mortality_ewe_mean: float = 0.04
    mortality_ewe_std: float = 0.01
    feed_intake_mean: float = 2.2 # kg dry matter/day
    feed_intake_std: float = 0.3
    # NEW: Parameters for physical bales
    hay_yield_ha_mean: float = 10.0 # Bales per hectare (during haymaking)
    hay_yield_ha_std: float = 3.0
    bale_weight_kg: float = 200.0   # Weight of one bale
    
    # --- ECONOMICS ---
    price_meat_std: float = 8.0 
    
    # Feed costs
    cost_feed_own_mean: float = 2.5
    cost_feed_own_std: float = 0.5
    cost_feed_market_mean: float = 8.0
    cost_feed_market_std: float = 2.0
    
    # Subsidies (SZIF)
    subsidy_ha_mean: float = 4500.0
    subsidy_ha_std: float = 200.0
    subsidy_sheep_mean: float = 580.0
    subsidy_sheep_std: float = 50.0
    
    # Overheads and Fixed
    fixed_cost_ewe_mean: float = 900.0 
    fixed_cost_ewe_std: float = 100.0
    overhead_base_year: float = 40000.0 
    
    # Shocks (Risks)
    shock_prob_daily: float = 0.005 # 0.5% chance daily
    shock_cost_mean: float = 15000.0
    shock_cost_std: float = 5000.0

class FarmBIModel:
    def __init__(self, cfg: HardDataConfig):
        self.cfg = cfg
        self.date = pd.Timestamp("2025-01-01")
        self.ewes = cfg.barn_capacity  # Přiřadí se z uživatelského inputu
        self.lambs = 0
        self.cash = cfg.capital
        self.bcs = 3.0 
        
        # NEW: Physical hay stock
        self.hay_stock = 25.0 # Starting bales
        
        # LAND SPLIT (Key change)
        self.area_meadow = cfg.land_area * cfg.meadow_share     # Only for hay
        self.area_pasture = cfg.land_area * (1 - cfg.meadow_share) # Only for grazing
        
        self.history = []
        self.feed_log = {"Grazing": 0, "Stored": 0, "Market": 0}

        self.grass_curve = {
            1:0, 2:0, 3:0.1, 4:0.6, 5:1.2, 6:1.1, 
            7:0.7, 8:0.5, 9:0.7, 10:0.4, 11:0.1, 12:0
        }

    def _get_seasonal_overhead(self, month):
        base_daily = self.cfg.overhead_base_year / 365
        if month in [6, 7, 8]: return base_daily * 1.5
        elif month in [1, 2, 12]: return base_daily * 1.3
        else: return base_daily * 0.8

    def _perform_forecast(self):
        # Estimate: 180 days of winter * consumption * market price (to be safe)
        winter_feed_cost = 180 * self.cfg.feed_intake_mean * self.cfg.cost_feed_market_mean * self.ewes
        winter_overhead = (self.cfg.overhead_base_year / 2) * 1.2
        return (winter_feed_cost + winter_overhead) * (1.0 + self.cfg.safety_margin)

    def step(self):
        month = self.date.month
        current_meat_price = get_stochastic_value(self.cfg.price_meat_avg, self.cfg.price_meat_std)
        
        # --- 1. HAYMAKING (Only from Meadows!) ---
        # If it is June 15th, make hay for winter
        if month == 6 and self.date.day == 15:
            yield_per_ha = get_stochastic_value(self.cfg.hay_yield_ha_mean, self.cfg.hay_yield_ha_std)
            # Hay is made ONLY from meadow area (area_meadow), sheep don't graze there
            new_bales = self.area_meadow * yield_per_ha
            self.hay_stock += new_bales
            # Cost of bale production (net, diesel)
            cost_production = new_bales * 300 # 300 CZK direct cost for production
            self.cash -= cost_production

        # --- 2. BIOLOGY & FEEDING ---
        total_sheep = self.ewes + self.lambs
        current_intake = get_stochastic_value(self.cfg.feed_intake_mean, self.cfg.feed_intake_std)
        feed_demand_kg = total_sheep * current_intake
        
        season_factor = self.grass_curve[month]
        # Grass grows only on PASTURES (area_pasture), not on meadows!
        available_grass = self.area_pasture * 35.0 * season_factor * np.random.normal(1.0, 0.2)
        
        feed_cost = 0.0
        feed_source = ""
        
        # A) Is there enough grass on the pasture?
        if available_grass >= feed_demand_kg:
            # We graze (cheap)
            feed_cost = feed_demand_kg * get_stochastic_value(self.cfg.cost_feed_own_mean, 0.5)
            self.bcs = min(4.0, self.bcs + 0.003)
            feed_source = "Grazing"
        else:
            # B) Not enough grass -> Feed from storage (Hay)
            # Deficit that needs to be covered
            deficit_kg = feed_demand_kg - available_grass
            # Grass they ate (we pay for it as 'own feed cost')
            feed_cost += available_grass * self.cfg.cost_feed_own_mean
            
            # Account for hay waste (25% loss from spillage, trampling in feeder)
            hay_waste_factor = 1.45  # 45% waste
            needed_bales = (deficit_kg * hay_waste_factor) / self.cfg.bale_weight_kg
            
            # Do we have enough bales?
            # FIX: Handling negative stocks
            bales_from_stock = min(self.hay_stock, needed_bales)
            self.hay_stock -= bales_from_stock
            
            # Handling fee for hay feeding
            feed_cost += (bales_from_stock * self.cfg.bale_weight_kg) * 0.5 
            
            if bales_from_stock < needed_bales:
                # C) OUT OF HAY -> Market purchase (Crisis)
                missing_bales = needed_bales - bales_from_stock
                market_price_kg = get_stochastic_value(self.cfg.cost_feed_market_mean, self.cfg.cost_feed_market_std)
                feed_cost += (missing_bales * self.cfg.bale_weight_kg) * market_price_kg
                self.bcs = max(2.0, self.bcs - 0.002)
                feed_source = "Market"
            else:
                if month in [12,1,2]: self.bcs = max(2.5, self.bcs - 0.0005)
                feed_source = "Stored"

        self.feed_log[feed_source] += 1

        # Mortality
        mort_prob = (self.cfg.mortality_ewe_mean / 365) * (2 if self.bcs < 2.5 else 1)
        deaths = np.random.binomial(total_sheep, mort_prob)
        if total_sheep > 0: self.ewes = max(0, self.ewes - deaths)
        
        # Reproduction (Spring)
        if month == 3 and self.date.day == 15:
            new_lambs = int(self.ewes * get_stochastic_value(self.cfg.fertility_mean, 0.3))
            self.lambs += new_lambs
            self.bcs -= 0.5

        # Cashflow
        income_sales = 0.0
        income_subsidy = 0.0
        daily_overhead = self._get_seasonal_overhead(month)
        shock_cost = 0.0
        
        if np.random.random() < self.cfg.shock_prob_daily:
            shock_cost = get_stochastic_value(self.cfg.shock_cost_mean, self.cfg.shock_cost_std)

        # Subsidies
        if month == 11 and self.date.day == 20: 
            income_subsidy += ((self.cfg.land_area * self.cfg.subsidy_ha_mean) + (self.ewes * self.cfg.subsidy_sheep_mean)) * 0.7
        if month == 4 and self.date.day == 20: 
            income_subsidy += ((self.cfg.land_area * self.cfg.subsidy_ha_mean) + (self.ewes * self.cfg.subsidy_sheep_mean)) * 0.3

        # Sales (October)
        emergency_sale = 0
        if month == 10 and self.date.day == 15:
            lamb_income = self.lambs * 35.0 * current_meat_price
            
            if self.cfg.enable_forecasting:
                liability = self._perform_forecast()
                balance = self.cash + lamb_income - liability
                if balance < 0:
                    emergency_sale = int(abs(balance) / (60 * current_meat_price)) + 1
            
            cull = int(self.ewes * 0.15) + emergency_sale
            sold_ewes = min(cull, self.ewes)
            
            income_sales += lamb_income
            income_sales += sold_ewes * 60.0 * (current_meat_price * 0.8)
            
            self.ewes = (self.ewes - sold_ewes) + int(self.ewes * 0.15)
            self.lambs = 0
            feed_cost += self.ewes * get_stochastic_value(self.cfg.fixed_cost_ewe_mean, 50)

        total_income = income_sales + income_subsidy
        total_expense = feed_cost + daily_overhead + shock_cost
        self.cash += total_income - total_expense
        
        self.history.append({
            "Date": self.date,
            "Cash": self.cash,
            "Total Sheep": self.ewes + self.lambs,
            "Ewes": self.ewes,   # <--- Logging ewes (mothers)
            "Lambs": self.lambs, # <--- Logging lambs (offspring)
            "Hay Stock": self.hay_stock, # <--- Logging bales here!
            "BCS": self.bcs,
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

# --- SIDEBAR UI (COMPLETE LISTING) ---
with st.sidebar:
    st.title("🚜 Sheep Farm BI")
    
    st.header("🎛️ Inputs")
    st.subheader("1. Land Management")
    area = st.slider("Total Area (ha)", 5.0, 100.0, 15.0)
    meadow_pct = st.slider("Share of Meadows for Hay (%)", 0, 100, 20, help="This portion of land is reserved ONLY for hay production. Sheep do not graze here.")
    
    st.subheader("2. Herd and Finance")
    barn = st.slider("Barn (heads)", 10, 250, 25)
    cap = st.number_input("Capital (CZK)", value=200000)
    meat_price = st.slider("Meat Price (CZK)", 40.0, 120.0, 75.0)
    use_forecast = st.toggle("Cashflow Planner", value=True)
    
    st.markdown("---")
    st.header("📋 Model Parameters (Audit)")
    st.caption("Values: Mean ± Standard Deviation")

    # Instance for listing defaults
    d = HardDataConfig(0,0,0,0,0,0,0,0) 

    with st.expander("🧬 1. Biological Data", expanded=False):
        st.code(f"""
Carrying Capacity: {d.carrying_capacity_mean} ± {d.carrying_capacity_std} ks/ha
Fertility Rate:    {d.fertility_mean} ± {d.fertility_std}
Mortality Lamb:    {d.mortality_lamb_mean*100:.0f}% ± {d.mortality_lamb_std*100:.0f}%
Mortality Ewe:     {d.mortality_ewe_mean*100:.0f}% ± {d.mortality_ewe_std*100:.0f}%
Feed Intake:       {d.feed_intake_mean} ± {d.feed_intake_std} kg/day
        """)

    with st.expander("💰 2. Economic Data", expanded=False):
        st.code(f"""
Meat Price Vol:    ± {d.price_meat_std} CZK
SAPS Subsidy:      {d.subsidy_ha_mean:,.0f} ± {d.subsidy_ha_std} CZK/ha
VDJ Subsidy:       {d.subsidy_sheep_mean:,.0f} ± {d.subsidy_sheep_std} CZK/ks
        """)

    with st.expander("📉 3. Cost Data", expanded=False):
        st.code(f"""
Feed (Own):        {d.cost_feed_own_mean} ± {d.cost_feed_own_std} CZK/kg
Feed (Market):     {d.cost_feed_market_mean} ± {d.cost_feed_market_std} CZK/kg
Fixed Cost/Ewe:    {d.fixed_cost_ewe_mean} ± {d.fixed_cost_ewe_std} CZK
Overhead Base:     {d.overhead_base_year:,.0f} CZK/year
Shock Prob:        {d.shock_prob_daily*100:.1f}% / day
Shock Cost:        {d.shock_cost_mean:,.0f} ± {d.shock_cost_std:,.0f} CZK
        """)
        
    st.markdown("---")
    st.header("💾 Export Data")
    st.caption("Download the simulation for debugging in Excel.")

# --- RUNNING ---
cfg = HardDataConfig(
    sim_years=5, land_area=area, meadow_share=meadow_pct/100.0, barn_capacity=barn, capital=cap,
    price_meat_avg=meat_price, enable_forecasting=use_forecast, safety_margin=0.2
)
model = FarmBIModel(cfg)
df = model.run()

# --- DEBUG & EXPORT BUTTON ---
with st.sidebar:
    # CSV Convert
    csv = df.to_csv().encode('utf-8')
    st.download_button(
        label="Download CSV (Debug)",
        data=csv,
        file_name='sheep_simulation_debug.csv',
        mime='text/csv',
    )
    if st.checkbox("Show Raw Data Head"):
        st.dataframe(df.head())

# --- BI DASHBOARD ---
st.title("📊 Farm Management Dashboard")

# Land Split Display (Visual Check)
col_land1, col_land2 = st.columns(2)
with col_land1:
    st.metric("Pastures (For sheep)", f"{model.area_pasture:.1f} ha")
with col_land2:
    st.metric("Meadows (For hay)", f"{model.area_meadow:.1f} ha")

# 1. KPI ROW
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
final_cash = df["Cash"].iloc[-1]
kpi1.metric("Liquidity (Cash)", f"{final_cash:,.0f} CZK", delta=f"{final_cash-cap:,.0f}")
kpi2.metric("Herd Status", int(df["Total Sheep"].iloc[-1]), delta=int(df["Total Sheep"].iloc[-1]-barn))
kpi3.metric("Cost of Shocks", f"{df['Exp_Shock'].sum():,.0f} CZK")
kpi4.metric("Total Subsidies", f"{df['Income_Subsidy'].sum():,.0f} CZK")

# 2. FINANCIAL STRUCTURE (Stacked Area)
st.subheader("💸 Cashflow Structure")
col_chart, col_pie = st.columns([3, 1])

with col_chart:
    # Monthly aggregation for nicer chart
    df_monthly = df.resample("ME").sum()
    
    fig, ax = plt.subplots(figsize=(10, 4))
    # Income (Positive)
    ax.bar(df_monthly.index, df_monthly["Income_Sales"], label="Meat Sales", color="#2ecc71", width=20)
    ax.bar(df_monthly.index, df_monthly["Income_Subsidy"], bottom=df_monthly["Income_Sales"], label="Subsidies", color="#f1c40f", width=20)
    
    # Expenses (Negative)
    ax.bar(df_monthly.index, -df_monthly["Exp_Feed"], label="Feed", color="#e67e22", width=20)
    ax.bar(df_monthly.index, -df_monthly["Exp_Overhead"], bottom=-df_monthly["Exp_Feed"], label="Overhead", color="#3498db", width=20)
    ax.bar(df_monthly.index, -df_monthly["Exp_Shock"], bottom=-(df_monthly["Exp_Feed"]+df_monthly["Exp_Overhead"]), label="Shocks", color="#e74c3c", width=20)
    
    ax.axhline(0, color="white", linewidth=0.5)
    ax.set_ylabel("CZK")
    ax.legend(loc="upper left", bbox_to_anchor=(1,1))
    ax.grid(True, alpha=0.1)
    st.pyplot(fig)

with col_pie:
    st.markdown("**Expense Breakdown**")
    total_exp = df[["Exp_Feed", "Exp_Overhead", "Exp_Shock"]].sum()
    fig_pie, ax_pie = plt.subplots()
    ax_pie.pie(total_exp, labels=["Feed", "Overhead", "Shocks"], autopct='%1.1f%%', colors=["#e67e22", "#3498db", "#e74c3c"], startangle=90)
    st.pyplot(fig_pie)

# 3. HERD STRUCTURE - Modern Line Chart
st.subheader("🐑 Herd Structure")

fig_herd, ax_herd = plt.subplots(figsize=(14, 5))

# Modern line chart
ax_herd.plot(df.index, df["Ewes"], 
             label="Ewes (Mothers)", color="#3498db", linewidth=3, marker="o", markersize=2)
ax_herd.plot(df.index, df["Lambs"], 
             label="Lambs (Offspring)", color="#2ecc71", linewidth=3, marker="s", markersize=2)
ax_herd.plot(df.index, df["Total Sheep"], 
             label="Total Herd", color="#e74c3c", linewidth=2.5, linestyle="--", alpha=0.7)

ax_herd.set_ylabel("Number of Heads", color="white", fontsize=12, fontweight="bold")
ax_herd.set_xlabel("Date", color="white", fontsize=11)
ax_herd.tick_params(axis='y', labelcolor="white")
ax_herd.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
ax_herd.legend(loc="upper left", fontsize=11, framealpha=0.9)

st.pyplot(fig_herd)
st.caption("Blue line = breeding ewes, Green line = lambs born in spring, Red dashed = total. October drop = sales period.")

# 4. HAY STOCK - Modern Line Chart
st.subheader("🌾 Hay Stock Management")

fig_hay, ax_hay = plt.subplots(figsize=(14, 5))

# Modern gradient-like effect with fill and line
ax_hay.fill_between(df.index, 0, df["Hay Stock"], 
                     color="#f39c12", alpha=0.3, label="Hay Stock Level")
ax_hay.plot(df.index, df["Hay Stock"], 
            color="#f39c12", linewidth=3, marker="o", markersize=2, markerfacecolor="#e67e22", label="Bales Count")

# Add critical threshold line (visual reference)
ax_hay.axhline(y=0, color="red", linestyle="--", linewidth=2, alpha=0.5, label="Critical (Empty)")

ax_hay.set_ylabel("Number of Bales", color="white", fontsize=12, fontweight="bold")
ax_hay.set_xlabel("Date", color="white", fontsize=11)
ax_hay.tick_params(axis='y', labelcolor="white")
ax_hay.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
ax_hay.legend(loc="upper right", fontsize=11, framealpha=0.9)
ax_hay.set_ylim(bottom=-5)  # Small margin below zero

st.pyplot(fig_hay)
st.caption("June spike = haymaking season. Winter decline = feeding season. If hits red line = must buy feed from market (expensive).")

# 5. SEASONAL ANALYSIS & MARKET
st.subheader("📅 Seasonality and Market")
col_seas, col_market = st.columns(2)

with col_seas:
    st.markdown("**Average Monthly Cashflow**")
    # Get average balance for each month of the year
    df["Month"] = df.index.month
    df["Net_Flow"] = (df["Income_Sales"] + df["Income_Subsidy"]) - (df["Exp_Feed"] + df["Exp_Overhead"] + df["Exp_Shock"])
    seasonal_data = df.groupby("Month")["Net_Flow"].mean()
    
    fig_s, ax_s = plt.subplots(figsize=(6, 3))
    colors = ["red" if x < 0 else "green" for x in seasonal_data]
    ax_s.bar(seasonal_data.index, seasonal_data, color=colors)
    ax_s.set_xticks(range(1, 13))
    ax_s.set_ylabel("Avg Profit/Loss (CZK)")
    ax_s.grid(True, alpha=0.2)
    st.pyplot(fig_s)
    st.caption("Red = months when the farm is 'bleeding' money (winter/spring).")

with col_market:
    st.markdown("**Market Price Volatility**")
    fig_m, ax_m = plt.subplots(figsize=(6, 3))
    sns.histplot(df["Meat_Price"], kde=True, ax=ax_m, color="purple", bins=30)
    ax_m.axvline(cfg.price_meat_avg, color="white", linestyle="--", label="Average")
    ax_m.set_xlabel("Meat Price (CZK/kg)")
    st.pyplot(fig_m)

# 6. OPERATIONAL EFFICIENCY
st.subheader("📊 Feeding Efficiency")

# Calculate percentages
total_days = model.feed_log["Grazing"] + model.feed_log["Stored"] + model.feed_log["Market"]
grazing_pct = (model.feed_log["Grazing"] / total_days * 100) if total_days > 0 else 0
stored_pct = (model.feed_log["Stored"] / total_days * 100) if total_days > 0 else 0
market_pct = (model.feed_log["Market"] / total_days * 100) if total_days > 0 else 0

# Create two columns: chart and explanation
col_feed_chart, col_feed_info = st.columns([2, 1])

with col_feed_chart:
    # Create horizontal bar chart for better readability
    fig_feed, ax_feed = plt.subplots(figsize=(12, 4))
    
    feed_data = [model.feed_log["Grazing"], model.feed_log["Stored"], model.feed_log["Market"]]
    feed_labels = [f"Grazing\n({grazing_pct:.0f}%)", f"Stored\n({stored_pct:.0f}%)", f"Market\n({market_pct:.0f}%)"]
    feed_colors = ["#2ecc71", "#f39c12", "#e74c3c"]
    
    bars = ax_feed.barh(feed_labels, feed_data, color=feed_colors, edgecolor="white", linewidth=2)
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, feed_data)):
        ax_feed.text(value/2, i, f"{int(value)} days", 
                    ha="center", va="center", fontsize=12, fontweight="bold", color="white")
    
    ax_feed.set_xlabel("Days per Year", color="white", fontsize=11, fontweight="bold")
    ax_feed.tick_params(axis='x', labelcolor="white")
    ax_feed.grid(axis='x', alpha=0.2, linestyle='--')
    ax_feed.set_xlim(0, total_days * 1.1)
    
    st.pyplot(fig_feed)

with col_feed_info:
    st.markdown("**📌 What it means:**")
    st.markdown(f"""
    ✅ **Grazing**: {grazing_pct:.0f}%
    - Sheep on pasture (cheap)
    - ~2-3 Kč/kg
    
    ⚠️ **Stored**: {stored_pct:.0f}%
    - Your hay from June
    - ~0.5 Kč/kg
    
    ❌ **Market**: {market_pct:.0f}%
    - Bought feed (expensive!)
    - ~8 Kč/kg
    """)
    
    if market_pct > 20:
        st.error(f"⚠️ High market buying ({market_pct:.0f}%)! Increase meadow % or reduce flock.")
    elif market_pct > 5:
        st.warning(f"⚡ Some market buying ({market_pct:.0f}%). Consider adjusting land split.")
    else:
        st.success("✓ Good! Minimal market purchases.")

st.caption("Green = sustainable pasture feeding | Orange = efficient hay management | Red = crisis/expensive emergency purchases")


# --- VALIDATION (Benchmark) ---
st.markdown("---")
st.subheader("✅ Komplexní Validace (Model vs. Realita ČR)")

# 1. Benchmark Data (Rozšířená sada - Zdroje: SCHOK, ÚZEI, FADN)
benchmark_data = {
    "1. Náklady Krmivo (Kč/ks)": 1750,
    "2. Náklady Veterina/Režie (Kč/ks)": 750,  # 350 vet + 400 ost.
    "3. Tržby Maso (Kč/ks)": 2900, 
    "4. Zisk bez dotací (Kč/ks)": -1150,
    "5. Odchov (ks jehňat/matku)": 1.35,       # Biologická efektivita
    "6. Závislost na dotacích (%)": 65.0       # Kolik % příjmu tvoří dotace
}

# 2. Výpočet metrik z tvého modelu
avg_ewes = df["Ewes"].mean()
if avg_ewes == 0: avg_ewes = 1 # Anti-zero division

# Ekonomika na 1 bahnici
model_feed = df["Exp_Feed"].sum() / (avg_ewes * cfg.sim_years)
model_overhead = (df["Exp_Overhead"].sum() + df["Exp_Shock"].sum()) / (avg_ewes * cfg.sim_years)
model_meat = df["Income_Sales"].sum() / (avg_ewes * cfg.sim_years)
model_profit_no_sub = model_meat - (model_feed + model_overhead)

# Biologie (Odchov)
# Počet prodaných jehňat za celou dobu / počet bahnic / roky
# Pozn: V modelu prodáváme v říjnu. Income_Sales > 0 indikuje prodej.
# Zjednodušený odhad: počet jehňat v létě (peak) / počet matek
avg_lamb_peak = df[df.index.month == 6]["Lambs"].mean()
model_rearing = avg_lamb_peak / avg_ewes if avg_ewes > 0 else 0

# Závislost na dotacích
total_income = df["Income_Sales"].sum() + df["Income_Subsidy"].sum()
model_subsidy_dep = (df["Income_Subsidy"].sum() / total_income * 100) if total_income > 0 else 0

# 3. Dataframe pro tabulku
validation_df = pd.DataFrame({
    "Metrika": list(benchmark_data.keys()),
    "Průměr ČR (Realita)": list(benchmark_data.values()),
    "Tvůj Model": [model_feed, model_overhead, model_meat, model_profit_no_sub, model_rearing, model_subsidy_dep]
})

# Výpočet odchylky
validation_df["Odchylka"] = validation_df["Tvůj Model"] - validation_df["Průměr ČR (Realita)"]

# Formátování a vykreslení
col_val1, col_val2 = st.columns([4, 3])

with col_val1:
    st.markdown("### 📋 Detailní Srovnání")
    
    def color_diff(val):
        """Barví odchylku: Zelená (malá), Červená (velká)"""
        color = 'green' if abs(val) < 200 else 'red' # Tolerance pro Kč
        return f'color: {color}'

    st.dataframe(
        validation_df.style.format(
            "{:,.1f}", subset=["Průměr ČR (Realita)", "Tvůj Model", "Odchylka"]
        ),
        use_container_width=True,
        height=300
    )

with col_val2:
    st.markdown("### 🎯 Klíčové KPI")
    
    # 1. Ziskovost
    diff_profit = model_profit_no_sub - benchmark_data["4. Zisk bez dotací (Kč/ks)"]
    if model_profit_no_sub > 0:
        st.error(f"❌ **PŘÍLIŠ ZISKOVÉ!** Model ukazuje zisk bez dotací {model_profit_no_sub:.0f} Kč. Realita v ČR je ztráta cca -1150 Kč. Zřejmě máš příliš levné krmivo nebo vysokou cenu masa.")
    elif abs(diff_profit) < 500:
        st.success("✅ **EKONOMIKA SEDÍ:** Model generuje realistickou ztrátu bez dotací.")
    else:
        st.warning("⚠️ **Odchylka v zisku:** Zkontroluj fixní náklady.")

    # 2. Biologie
    st.write("---")
    col_bio1, col_bio2 = st.columns(2)
    col_bio1.metric("Odchov (Model)", f"{model_rearing:.2f}", delta=f"{model_rearing - 1.35:.2f}")
    col_bio2.metric("Odchov (ČR)", "1.35")
    
    if model_rearing > 1.6:
        st.warning("⚠️ **Super-ovce?** Odchov > 1.6 je v extenzivním chovu velmi vzácný. Sniž plodnost nebo zvyš úmrtnost.")
    
    # 3. Dotace
    st.write("---")
    st.metric("Závislost na dotacích", f"{model_subsidy_dep:.1f} %", delta=f"{model_subsidy_dep - 65.0:.1f} %")
    st.caption("Pokud je toto číslo pod 50 %, tvůj model je příliš tržně optimistický.")