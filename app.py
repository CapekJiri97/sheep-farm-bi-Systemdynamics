import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass

# --- CONFIGURATION ---
st.set_page_config(page_title="Sheep Farm 12.0 - Ultimate Reality", layout="wide", page_icon="🚜")
plt.style.use('dark_background')

# --- HELPER FUNCTIONS ---
def get_stochastic_value(mean, std, min_val=0.0):
    val = np.random.normal(mean, std)
    return max(min_val, val)

@dataclass
class FarmConfig:
    # 1. SCALE & LAND
    sim_years: int
    land_area: float
    meadow_share: float
    barn_capacity: int
    initial_ewes: int
    barn_area_m2: float
    hay_barn_area_m2: float
    
    # 2. ECONOMICS
    capital: float
    price_meat_avg: float
    meat_price_std: float = 10.0
    initial_hay_bales: float = 25.0
    price_bale_sell_winter: float = 1200.0   # Buy/Sell prices
    price_bale_sell_summer: float = 600.0   
    price_ram_purchase: float = 10000.0
    
    # 3. STRATEGY (New!)
    machinery_mode: str = "Services" # "Services" or "Own"
    climate_profile: str = "Normal"  # "Normal", "Dry", "Mountain"
    include_labor_cost: bool = False
    safety_margin: float = 0.2
    enable_forecasting: bool = True
    
    # 4. COSTS (Base rates)
    cost_feed_own_mean: float = 2.5
    cost_feed_market_mean: float = 8.0
    
    cost_vet_base: float = 350.0
    cost_shearing: float = 50.0
    admin_base_cost: float = 5000.0       # Base admin cost per year
    admin_complexity_factor: float = 1.5  # Exponent (1.0=linear, 1.5=progressive)
    
    # Machinery / Services Rates
    service_mow_ha: float = 3000.0   # Price per ha (service)
    service_bale_pcs: float = 200.0  # Price per bale (service)
    
    own_mow_fuel_ha: float = 400.0   # Fuel only
    own_bale_material: float = 50.0  # Net wrap
    own_machine_capex: float = 1600000.0 # Tractor + Equipment cost
    own_machine_life: float = 10.0   # Years depreciation
    machinery_repair_mean: float = 25000.0
    machinery_repair_std: float = 5000.0
    machinery_failure_prob_daily: float = 0.0001  # Daily breakdown risk
    
    # 5. BIOLOGY
    fertility_mean: float = 1.5
    fertility_std: float = 0.2
    mortality_lamb_mean: float = 0.10
    mortality_ewe_mean: float = 0.04
    feed_intake_ewe: float = 2.2
    feed_intake_lamb: float = 1.2
    hay_yield_ha_mean: float = 12.0
    hay_yield_ha_std: float = 3.0
    bale_weight_kg: float = 250.0
    bale_volume_m3: float = 1.4
    
    # 6. SUBSIDIES
    subsidy_ha_mean: float = 8500.0
    subsidy_ha_std: float = 200.0
    subsidy_sheep_mean: float = 603.0
    subsidy_sheep_std: float = 50.0
    
    # 7. FIXED & SHOCKS
    tax_land_ha: float = 500.0
    tax_building_m2: float = 15.0
    fixed_cost_ewe_mean: float = 900.0 
    overhead_base_year: float = 40000.0
    barn_maintenance_m2_year: float = 60.0
    shock_prob_daily: float = 0.005
    shock_cost_mean: float = 15000.0
    shock_cost_std: float = 5000.0
    
    # Labor
    wage_hourly: float = 200.0
    labor_hours_per_ewe_year: float = 8.0
    labor_hours_per_ha_year: float = 10.0
    labor_hours_fix_year: float = 200.0
    labor_hours_barn_m2_year: float = 0.5

class FarmModel:
    def __init__(self, cfg: FarmConfig):
        self.cfg = cfg
        self.date = pd.Timestamp("2025-01-01")
        
        # --- HERD ---
        self.ewes = cfg.initial_ewes
        self.ewe_ages = [3.0] * cfg.initial_ewes
        self.rams_breeding = max(1, int(cfg.initial_ewes / 30))
        self.ram_age = 3.0
        self.lambs_male = 0
        self.lambs_female = 0
        self.lamb_age = 0.0
        
        # --- ASSETS ---
        self.cash = cfg.capital
        self.hay_stock_bales = cfg.initial_hay_bales
        self.bcs = 3.0
        
        # --- CLIMATE SETUP ---
        if cfg.climate_profile == "Dry":
            self.grass_mod = 0.7       # Less grass
            self.drought_chance = 0.02 # 2% daily chance of total burnout in summer
            self.winter_len_mod = 0.8  # Shorter winter
        elif cfg.climate_profile == "Mountain":
            self.grass_mod = 1.2       # More grass
            self.drought_chance = 0.001
            self.winter_len_mod = 1.3  # Longer winter
        else:
            self.grass_mod = 1.0
            self.drought_chance = 0.005
            self.winter_len_mod = 1.0

        self.is_winter = True
        self.winter_end_day = int(80 * self.winter_len_mod)
        self.winter_start_day = int(365 - (60 * self.winter_len_mod))
        
        # --- LAND ---
        self.area_meadow = cfg.land_area * cfg.meadow_share
        self.area_pasture = cfg.land_area * (1 - cfg.meadow_share)
        
        self.history = []
        self.event_log = []
        self.feed_log = {"Grazing": 0, "Stored": 0, "Market": 0}
        
        # Grass curve (Month 1-12)
        self.grass_curve = {1:0, 2:0, 3:0.1, 4:0.5, 5:1.2, 6:1.1, 7:0.8, 8:0.6, 9:0.8, 10:0.4, 11:0.1, 12:0}

    def _check_barn_capacity(self):
        max_volume = self.cfg.hay_barn_area_m2 * 3.0
        max_bales = int(max_volume / self.cfg.bale_volume_m3)
        if self.hay_stock_bales > max_bales:
            excess = self.hay_stock_bales - max_bales
            income = excess * self.cfg.price_bale_sell_summer
            self.cash += income
            self.hay_stock_bales = max_bales
            self.event_log.append(f"{self.date.date()}: ⚠️ Seník plný! Prodáno {int(excess)} balíků.")

    def _get_seasonal_overhead(self, month):
        base = self.cfg.overhead_base_year / 365
        # Add barn maintenance (spread daily)
        total_m2 = self.cfg.barn_area_m2 + self.cfg.hay_barn_area_m2
        barn_maint = (total_m2 * self.cfg.barn_maintenance_m2_year) / 365
        return (base * (1.5 if month in [6,7,8] else 0.8)) + barn_maint

    def _perform_forecast(self):
        total_adults = self.ewes + self.rams_breeding
        winter_feed_cost = 180 * self.cfg.feed_intake_ewe * self.cfg.cost_feed_market_mean * total_adults
        return (winter_feed_cost + 40000) * (1.0 + self.cfg.safety_margin)

    def step(self):
        month = self.date.month
        day = self.date.dayofyear
        
        # 1. MEAT PRICE
        base_meat_price = get_stochastic_value(self.cfg.price_meat_avg, self.cfg.meat_price_std)
        if month in [3, 4]: base_meat_price *= 1.25 # Easter premium
        
        # 2. SEASON CONTROL
        if self.is_winter and day > self.winter_end_day:
            if np.random.random() < 0.1: 
                self.is_winter = False
                self.event_log.append(f"{self.date.date()}: 🌱 Jaro ({self.cfg.climate_profile})")
        
        if not self.is_winter and day > self.winter_start_day:
            if np.random.random() < 0.1:
                self.is_winter = True
                self.winter_end_day = int(get_stochastic_value(80 * self.winter_len_mod, 10))
                self.event_log.append(f"{self.date.date()}: ❄️ Zima")

        # 3. FEEDING & BCS
        total_adults = self.ewes + self.rams_breeding
        total_lambs = self.lambs_male + self.lambs_female
        demand_kg = (total_adults * self.cfg.feed_intake_ewe) + (total_lambs * self.cfg.feed_intake_lamb)
        
        feed_cost = 0.0
        feed_source = ""
        
        # Daily cost trackers
        day_vet = 0.0
        day_mow = 0.0
        day_shearing = 0.0
        day_ram_purchase = 0.0
        day_machinery = 0.0
        day_admin = 0.0
        
        # Drought simulation
        is_drought = False
        if not self.is_winter and month in [6,7,8]:
            if np.random.random() < self.drought_chance:
                is_drought = True
                self.event_log.append(f"{self.date.date()}: ☀️ Sucho! Tráva neroste.")

        if self.is_winter or is_drought:
            # Feeding Hay
            needed_bales = (demand_kg * 1.2) / self.cfg.bale_weight_kg # 20% waste
            fed = min(self.hay_stock_bales, needed_bales)
            self.hay_stock_bales -= fed
            
            feed_cost = fed * 50 # Handling
            
            if fed < needed_bales:
                buy_bales = needed_bales - fed
                feed_cost += buy_bales * get_stochastic_value(self.cfg.price_bale_sell_winter, 100)
                self.bcs = max(1.5, self.bcs - 0.002) # Hunger
                feed_source = "Market"
            else:
                self.bcs = max(2.5, self.bcs - 0.001)
                feed_source = "Stored"
        else:
            # Grazing
            growth = self.grass_curve[month] * self.grass_mod
            avail = self.area_pasture * 35.0 * growth * np.random.normal(1.0, 0.2)
            
            if avail >= demand_kg:
                self.bcs = min(4.0, self.bcs + 0.004)
                feed_cost = demand_kg * 0.2 # Salt/Water
                feed_source = "Grazing"
            else:
                # Supplement
                deficit = demand_kg - avail
                needed_bales = (deficit * 1.4) / self.cfg.bale_weight_kg
                fed = min(self.hay_stock_bales, needed_bales)
                self.hay_stock_bales -= fed
                
                feed_cost = (avail * 0.2) + (fed * 50)
                
                if fed < needed_bales: 
                    self.bcs -= 0.002
                    missing = needed_bales - fed
                    feed_cost += missing * self.cfg.price_bale_sell_summer
                    feed_source = "Market"
                else:
                    feed_source = "Stored"
        
        self.feed_log[feed_source] += 1

        # 4. HEALTH & COSTS (BCS Impact)
        vet_multiplier = 1.0
        if self.bcs < 2.5: vet_multiplier = 2.0 # Nutná léčba
        
        # 5. MACHINERY COST (Make or Buy)
        if self.cfg.machinery_mode == "Own":
            # Depreciation (Odpisy)
            daily_depreciation = (self.cfg.own_machine_capex / self.cfg.own_machine_life) / 365
            day_machinery += daily_depreciation
            
            # Breakdown risk (increases with land area used)
            if np.random.random() < (self.cfg.machinery_failure_prob_daily * self.cfg.land_area):
                repair = get_stochastic_value(self.cfg.machinery_repair_mean, self.cfg.machinery_repair_std)
                day_machinery += repair
                self.event_log.append(f"{self.date.date()}: 🔧 Porucha traktoru! Oprava: {int(repair)} Kč.")

        # 6. ADMIN PENALTY (Diseconomies of Scale)
        total_animals = total_adults + total_lambs
        # Power law: Base * (N/50)^1.5 ... costs grow faster than linearly
        admin_scale = (max(1, total_animals) / 50.0) ** self.cfg.admin_complexity_factor
        day_admin = (self.cfg.admin_base_cost * admin_scale) / 365

        # --- EVENTS ---
        income = 0.0
        var_cost = 0.0
        
        # Lambing
        if month == 3 and self.date.day == 15:
            f = get_stochastic_value(self.cfg.fertility_mean, self.cfg.fertility_std)
            if self.bcs < 2.5: f *= 0.6
            born = int(self.ewes * f)
            self.lambs_male += int(born/2)
            self.lambs_female += (born - int(born/2))
            self.bcs -= 0.5
            
            # Spring Vet
            day_vet += (total_adults + born) * (self.cfg.cost_vet_base * vet_multiplier / 2)
            self.event_log.append(f"{self.date.date()}: 🍼 Narozeno {born} jehňat.")

        # Shearing
        if month == 5 and self.date.day == 15:
            day_shearing += total_adults * self.cfg.cost_shearing
            self.event_log.append(f"{self.date.date()}: ✂️ Stříhání.")

        # Mowing (June + Sept)
        if (month == 6 or month == 9) and self.date.day == 10:
            yield_h = get_stochastic_value(self.cfg.hay_yield_ha_mean, self.cfg.hay_yield_ha_std) * self.grass_mod * (0.6 if month==9 else 1.0)
            bales = self.area_meadow * yield_h
            self.hay_stock_bales += bales
            
            # Cost calculation based on Mode
            if self.cfg.machinery_mode == "Services":
                cost = (self.area_meadow * self.cfg.service_mow_ha) + (bales * self.cfg.service_bale_pcs)
            else:
                cost = (self.area_meadow * self.cfg.own_mow_fuel_ha) + (bales * self.cfg.own_bale_material)
            
            day_mow += cost
            self._check_barn_capacity()
            self.event_log.append(f"{self.date.date()}: 🚜 Seč ({self.cfg.machinery_mode}). {int(bales)} balíků.")

        # Sales (October)
        if month == 10 and self.date.day == 15:
            # Planner logic
            if self.cfg.enable_forecasting:
                forecast = self._perform_forecast()
                projected = self.cash - forecast
                if projected < 0 and self.hay_stock_bales > 0:
                    needed = int(abs(projected) / self.cfg.price_bale_sell_summer) + 1
                    sold_hay = min(self.hay_stock_bales * 0.4, needed)
                    income += sold_hay * self.cfg.price_bale_sell_summer
                    self.hay_stock_bales -= sold_hay
            
            # 1. Sell Males
            income += self.lambs_male * 40 * base_meat_price
            sold_m = self.lambs_male
            self.lambs_male = 0
            
            # 2. Renew Females (S respektem ke kapacitě ovčína)
            cull_count = int(self.ewes * 0.15)
            future_ewes = self.ewes - cull_count
            
            # Kolik můžeme maximálně doplnit?
            max_new_capacity = self.cfg.barn_capacity - future_ewes
            
            # Chceme doplnit max. 80% jehniček, ale ne víc, než se vejde
            potential_keep = int(self.lambs_female * 0.8)
            keep = max(0, min(potential_keep, max_new_capacity))
            
            # Zbytek prodáme
            sell_females = self.lambs_female - keep
            income += sell_females * 35.0 * base_meat_price
            
            # 3. Cull Old Ewes
            self.ewes = max(1, future_ewes + keep)
            self.lambs_female = 0
            income += cull_count * 60 * base_meat_price * 0.7
            
            # 3b. Check Ram Ratio (Growth support)
            needed_rams = max(1, int(self.ewes / 30))
            if self.rams_breeding < needed_rams:
                buy_rams = needed_rams - self.rams_breeding
                day_ram_purchase += buy_rams * self.cfg.price_ram_purchase
                self.rams_breeding += buy_rams
                self.event_log.append(f"{self.date.date()}: 🐏 Růst stáda -> nákup {buy_rams} beranů.")
            
            # 4. Ram Replace (every 2 years)
            if self.date.year % 2 == 0:
                 replace_count = max(1, int(round(self.rams_breeding * 0.5)))
                 cost = replace_count * self.cfg.price_ram_purchase
                 day_ram_purchase += cost
                 self.event_log.append(f"🐏 Výměna beranů.")

            # 5. Fall Vet
            day_vet += (self.ewes + self.rams_breeding) * (self.cfg.cost_vet_base * vet_multiplier / 2)
            
            self.event_log.append(f"{self.date.date()}: 💰 Prodej. Příjem {int(income)}.")

        # Subsidies
        if month == 11 and self.date.day == 20:
            income += ((self.cfg.land_area * get_stochastic_value(self.cfg.subsidy_ha_mean, 200)) + (self.ewes * self.cfg.subsidy_sheep_mean)) * 0.7
        if month == 4 and self.date.day == 20:
             income += ((self.cfg.land_area * get_stochastic_value(self.cfg.subsidy_ha_mean, 200)) + (self.ewes * self.cfg.subsidy_sheep_mean)) * 0.3
        
        # Land Tax
        if month == 12 and self.date.day == 31:
             var_cost += self.cfg.land_area * self.cfg.tax_land_ha
             var_cost += (self.cfg.barn_area_m2 + self.cfg.hay_barn_area_m2) * self.cfg.tax_building_m2

        # Labor
        labor_animals = total_adults * self.cfg.labor_hours_per_ewe_year
        labor_land = self.cfg.land_area * self.cfg.labor_hours_per_ha_year
        labor_barn = (self.cfg.barn_area_m2 + self.cfg.hay_barn_area_m2) * self.cfg.labor_hours_barn_m2_year
        labor_fix = self.cfg.labor_hours_fix_year
        
        daily_hours = (labor_animals + labor_land + labor_fix + labor_barn) / 365
        labor_val = 0
        if self.cfg.include_labor_cost:
            labor_val = daily_hours * self.cfg.wage_hourly

        # Shocks
        shock_val = 0.0
        if np.random.random() < self.cfg.shock_prob_daily:
            shock_val = get_stochastic_value(self.cfg.shock_cost_mean, self.cfg.shock_cost_std)
            self.event_log.append(f"{self.date.date()}: ⚡ Šok!")

        # Finalize
        var_cost += day_vet + day_mow + day_shearing + day_ram_purchase + day_machinery + day_admin
        daily_overhead = self._get_seasonal_overhead(month)
        
        total_out = feed_cost + var_cost + daily_overhead + labor_val + shock_val
        self.cash += income - total_out
        
        # Mortality (Simple daily check)
        mort_prob_ewe = (self.cfg.mortality_ewe_mean / 365)
        if self.bcs < 2.0: mort_prob_ewe *= 5
        elif self.bcs < 2.5: mort_prob_ewe *= 2
        
        deaths_ewes = np.random.binomial(self.ewes, mort_prob_ewe)
        self.ewes = max(0, self.ewes - deaths_ewes)
        
        if self.lambs_male + self.lambs_female > 0:
            mort_prob_lamb = (self.cfg.mortality_lamb_mean / 365) * (2 if self.bcs < 2.5 else 1)
            self.lambs_male = max(0, self.lambs_male - np.random.binomial(self.lambs_male, mort_prob_lamb))
            self.lambs_female = max(0, self.lambs_female - np.random.binomial(self.lambs_female, mort_prob_lamb))

        self.history.append({
            "Date": self.date,
            "Cash": self.cash,
            "Ewes": self.ewes,
            "Lambs": self.lambs_male + self.lambs_female,
            "Lambs Male": self.lambs_male,
            "Lambs Female": self.lambs_female,
            "Total Animals": self.ewes + self.rams_breeding + self.lambs_male + self.lambs_female,
            "Hay Stock": self.hay_stock_bales,
            "Income": income,
            "Exp_Feed": feed_cost,
            "Exp_Vet": day_vet,
            "Exp_Machinery": day_machinery,
            "Exp_Mow": day_mow,
            "Exp_Shearing": day_shearing,
            "Exp_RamPurchase": day_ram_purchase,
            "Exp_Admin": day_admin,
            "Exp_Labor": labor_val,
            "Labor Hours": daily_hours,
            "Exp_Overhead": daily_overhead,
            "Exp_Shock": shock_val,
            "Exp_Variable": day_vet + day_mow + day_shearing + day_ram_purchase + day_machinery,
            "BCS": self.bcs,
            "Meat_Price": base_meat_price
        })
        self.date += pd.Timedelta(days=1)

    def run(self):
        for _ in range(self.cfg.sim_years * 365): self.step()
        return pd.DataFrame(self.history).set_index("Date")

# --- SIDEBAR UI ---
with st.sidebar:
    st.title("Ovčí farma")
    st.markdown("**Proměnné modelu**")
    
    st.header("1. Kapacita a Infrastruktura")
    target_ewes = st.slider("Cílová kapacita (ovčín)", 10, 500, 60, help="Maximální počet bahnic. Určuje velikost potřebné budovy.")
    
    req_m2 = int(target_ewes * 2.5) # 2.5 m2 per ewe
    barn_m2 = st.number_input("Velikost ovčína (m²)", 50, 2000, req_m2, help=f"Pro zvířata. Doporučeno: {req_m2} m² (2.5 m²/ks vč. jehňat a uliček)")
    hay_barn_m2 = st.number_input("Velikost seníku (m²)", 50, 2000, 100, help="Pro uskladnění sena. 100 m² pojme cca 200 balíků (při stohování 3m).")
    
    area = st.number_input("Celková plocha (ha)", 5.0, 100.0, 15.0)
    meadow_pct = st.slider("Podíl luk na seno (%)", 0, 100, 40, help="Část plochy jen na výrobu sena (pastva zakázana)")
    
    st.header("2. Stádo a ekonomika")
    start_ewes = st.slider("Počet bahnic (start)", 10, target_ewes, min(20, target_ewes), help="Kolik ovcí nakoupíte do začátku.")
    meat_price = st.slider("Cena masa (Kč/kg)", 60.0, 120.0, 85.0)
    start_hay = st.number_input("Počáteční zásoba sena (balíky)", 0, 500, 25)
    cap = st.number_input("Počáteční kapitál (CZK)", value=200000)
    
    st.header("3. Pokročilé")
    labor_on = st.checkbox("🕐 Započítat náklady na vlastní práci", False, help="6h/rok na bahnici @ 200 Kč/h")
    climate = st.selectbox("🌤️ Klimatický profil", ["Normal", "Dry", "Mountain"])
    machinery = st.radio("🚜 Seč a lisování", ["Services", "Own"], help="Services = pronájem; Own = vlastní stroj")
    use_forecast = st.toggle("Cashflow Planner", value=True)
    
    st.markdown("---")
    st.header("Detailní nastavení parametrů")
    
    with st.expander("🧬 Biologie a Produkce"):
        p_fertility = st.number_input("Plodnost (ks/bahnici)", 1.0, 3.0, 1.5, 0.1)
        p_mortality_lamb = st.number_input("Úhyn jehňat (%)", 0.0, 50.0, 10.0, 1.0) / 100.0
        p_mortality_ewe = st.number_input("Úhyn bahnic (%)", 0.0, 20.0, 4.0, 0.5) / 100.0
        p_feed_ewe = st.number_input("Spotřeba bahnice (kg sušiny/den)", 1.0, 4.0, 2.2, 0.1)
        p_hay_yield = st.number_input("Výnos sena (balíků/ha)", 5.0, 30.0, 12.0, 1.0)
        
    with st.expander("💰 Provozní Náklady a Ceny"):
        c_feed_own = st.number_input("Cena vl. krmiva (Kč/kg)", 0.5, 10.0, 2.5, 0.1)
        c_feed_market = st.number_input("Cena kup. krmiva (Kč/kg)", 2.0, 20.0, 8.0, 0.5)
        c_vet = st.number_input("Veterina (Kč/ks/rok)", 100.0, 2000.0, 350.0, 50.0)
        c_shearing = st.number_input("Stříhání (Kč/ks)", 20.0, 200.0, 50.0, 10.0)
        c_ram = st.number_input("Cena berana (Kč)", 5000.0, 30000.0, 10000.0, 1000.0)
        c_bale_sell_winter = st.number_input("Cena sena Zima (Kč/balík)", 200.0, 2000.0, 800.0, 50.0)
        c_bale_sell_summer = st.number_input("Cena sena Léto (Kč/balík)", 100.0, 1000.0, 400.0, 50.0)
        
    with st.expander("🚜 Stroje a Služby"):
        s_mow_ha = st.number_input("Služba: Seč (Kč/ha)", 500.0, 5000.0, 1500.0, 100.0)
        s_bale = st.number_input("Služba: Lisování (Kč/ks)", 50.0, 500.0, 200.0, 10.0)
        o_capex = st.number_input("Vlastní: Cena stroje (Kč)", 100000.0, 5000000.0, 600000.0, 50000.0)
        o_fuel = st.number_input("Vlastní: Nafta seč (Kč/ha)", 100.0, 1000.0, 400.0, 50.0)
        o_repair = st.number_input("Vlastní: Opravy ročně (Kč)", 0.0, 100000.0, 15000.0, 1000.0)

    with st.expander("🏛️ Dotace a Daně"):
        sub_ha = st.number_input("SAPS (Kč/ha)", 0.0, 20000.0, 8500.0, 100.0)
        sub_sheep = st.number_input("VDJ (Kč/ks)", 0.0, 5000.0, 603.0, 10.0)
        tax_land = st.number_input("Daň z nemovitosti (Kč/ha)", 0.0, 2000.0, 500.0, 50.0)
        tax_build = st.number_input("Daň ze staveb (Kč/m²)", 0.0, 100.0, 15.0, 1.0)

    with st.expander("🏢 Režie a Škálování"):
        ov_base = st.number_input("Základní režie (Kč/rok)", 0.0, 200000.0, 40000.0, 1000.0)
        adm_base = st.number_input("Admin základ (Kč/rok)", 0.0, 50000.0, 5000.0, 500.0)
        adm_factor = st.number_input("Admin faktor (Diseconomy)", 1.0, 3.0, 1.5, 0.1, help="Exponent růstu administrativy. 1.0 = lineární, 1.5 = progresivní zátěž.")
        wage = st.number_input("Hodinová mzda (Kč/h)", 100.0, 1000.0, 200.0, 10.0)
        labor_h = st.number_input("Pracnost (h/ks/rok)", 1.0, 20.0, 6.0, 0.5)
        labor_ha = st.number_input("Pracnost půda (h/ha/rok)", 0.0, 50.0, 10.0, 1.0, help="Údržba ohradníků, pastvin, sečení nedopasků.")
        labor_fix = st.number_input("Fixní pracnost (h/rok)", 0.0, 1000.0, 200.0, 50.0, help="Údržba budov, administrativa, cesty.")
        labor_barn_m2 = st.number_input("Pracnost budovy (h/m²/rok)", 0.0, 10.0, 0.5, 0.1, help="Úklid, údržba, manipulace v ovčíně.")
        maint_barn_m2 = st.number_input("Údržba budovy (Kč/m²/rok)", 0.0, 1000.0, 60.0, 10.0, help="Opravy střechy, nátěry, dezinfekce.")
        shock_p = st.number_input("Pravděpodobnost šoku (denní %)", 0.0, 5.0, 0.5, 0.1) / 100.0

# --- RUN SIMULATION ---
cfg = FarmConfig(
    sim_years=5, 
    land_area=area, 
    meadow_share=meadow_pct/100.0, 
    barn_capacity=target_ewes,
    initial_ewes=start_ewes,
    barn_area_m2=barn_m2,
    hay_barn_area_m2=hay_barn_m2,
    capital=cap,
    price_meat_avg=meat_price, 
    initial_hay_bales=start_hay,
    enable_forecasting=use_forecast, 
    safety_margin=0.2,
    include_labor_cost=labor_on,
    climate_profile=climate,
    machinery_mode=machinery,
    
    # Overrides from advanced settings
    fertility_mean=p_fertility,
    mortality_lamb_mean=p_mortality_lamb,
    mortality_ewe_mean=p_mortality_ewe,
    feed_intake_ewe=p_feed_ewe,
    hay_yield_ha_mean=p_hay_yield,
    
    cost_feed_own_mean=c_feed_own,
    cost_feed_market_mean=c_feed_market,
    cost_vet_base=c_vet,
    cost_shearing=c_shearing,
    price_ram_purchase=c_ram,
    price_bale_sell_winter=c_bale_sell_winter,
    price_bale_sell_summer=c_bale_sell_summer,
    
    service_mow_ha=s_mow_ha,
    service_bale_pcs=s_bale,
    own_machine_capex=o_capex,
    own_mow_fuel_ha=o_fuel,
    machinery_repair_mean=o_repair,
    
    subsidy_ha_mean=sub_ha,
    subsidy_sheep_mean=sub_sheep,
    tax_land_ha=tax_land,
    tax_building_m2=tax_build,
    
    overhead_base_year=ov_base,
    barn_maintenance_m2_year=maint_barn_m2,
    admin_base_cost=adm_base,
    admin_complexity_factor=adm_factor,
    wage_hourly=wage,
    labor_hours_per_ewe_year=labor_h,
    labor_hours_per_ha_year=labor_ha,
    labor_hours_fix_year=labor_fix,
    labor_hours_barn_m2_year=labor_barn_m2,
    shock_prob_daily=shock_p
)

model = FarmModel(cfg)
df = model.run()

# --- SIDEBAR EXPORT ---
with st.sidebar:
    st.markdown("---")
    st.header("💾 Export")
    csv = df.to_csv().encode('utf-8')
    st.download_button(
        label="📥 Stáhnout data (CSV)",
        data=csv,
        file_name='farm_11_simulation.csv',
        mime='text/csv',
    )
    if st.checkbox("📋 Ukázat surová data"):
        st.dataframe(df.head(50), use_container_width=True)

# --- MAIN DASHBOARD ---
st.title("Přehled farmy ovčího hospodářství") 

# --- 1. KPI ROW ---
col_kpi1, col_kpi2, col_kpi3, col_kpi4, col_kpi5 = st.columns(5)

final_cash = df["Cash"].iloc[-1]
final_animals = df["Total Animals"].iloc[-1]
final_hay = df["Hay Stock"].iloc[-1]
total_profit = final_cash - cap

col_kpi1.metric(
    "Hotovost (konec)", 
    f"{final_cash:,.0f} Kč", 
    delta=f"{total_profit:,.0f}",
    delta_color="off"
)
col_kpi2.metric(
    "Stav stáda", 
    int(final_animals),
    delta=int(final_animals - start_ewes),
    delta_color="off"
)
col_kpi3.metric(
    "Seno (konec sezóny)", 
    f"{final_hay:.0f} balíků"
)
col_kpi4.metric(
    "ROI", 
    f"{(total_profit/cap*100):.1f}%",
    delta_color="off"
)

avg_md_year = (df["Labor Hours"].sum() / cfg.sim_years) / 8.0
col_kpi5.metric(
    "Pracnost (MD/rok)", 
    f"{avg_md_year:.1f} MD",
    help="Průměrný počet Man-Days (8h) ročně nutný k obsluze farmy."
)

# --- 2. HERD STRUCTURE ---
st.subheader("Struktura stáda (detailně)")

fig_herd, ax_herd = plt.subplots(figsize=(14, 5))

ax_herd.plot(df.index, df["Ewes"], label="Bahnice", color="#3498db", linewidth=3, marker="o", markersize=1)
ax_herd.plot(df.index, df["Lambs Male"], label="Beránci (výkrm)", color="#e74c3c", linewidth=2.5, alpha=0.8)
ax_herd.plot(df.index, df["Lambs Female"], label="Jehničky (obnova)", color="#2ecc71", linewidth=2.5, alpha=0.8)
ax_herd.plot(df.index, df["Total Animals"], label="Celkem", color="#f39c12", linewidth=2, linestyle="--", alpha=0.7)

ax_herd.set_ylabel("Počet Zvířat", fontsize=11, fontweight="bold")
ax_herd.set_xlabel("Datum", fontsize=11)
ax_herd.grid(True, alpha=0.2)
ax_herd.legend(loc="upper left", fontsize=10, framealpha=0.95)

st.pyplot(fig_herd)

# --- 3. HAY MANAGEMENT ---
st.subheader("Seno - výroba a spotřeba")

fig_hay, ax_hay = plt.subplots(figsize=(14, 5))

ax_hay.fill_between(df.index, 0, df["Hay Stock"], color="#f39c12", alpha=0.3, label="Zásoby")
ax_hay.plot(df.index, df["Hay Stock"], color="#f39c12", linewidth=3, marker="o", markersize=1)
ax_hay.axhline(y=0, color="red", linestyle="--", linewidth=2, alpha=0.5, label="Kritický stav (prázdno)")

ax_hay.set_ylabel("Balíky Sena", fontsize=11, fontweight="bold")
ax_hay.set_xlabel("Datum", fontsize=11)
ax_hay.grid(True, alpha=0.2)
ax_hay.legend(loc="upper right", fontsize=10)
ax_hay.set_ylim(bottom=-10)

st.pyplot(fig_hay)
st.caption("Červen = senoseč, září = otava. Zimní úbytek = spotřeba. Červená čára = kritický stav zásob.")

# --- 4. FINANCIAL OVERVIEW (STACKED CASHFLOW) ---
st.subheader("Cashflow a ziskovost")
col_chart, col_pie = st.columns([3, 1])

with col_chart:
    # Monthly aggregation for nicer chart
    df_monthly = df.resample("ME").sum()
    
    fig, ax = plt.subplots(figsize=(12, 5))
    # Income (Positive)
    ax.bar(df_monthly.index, df_monthly["Income"], label="Celkový Příjem", color="#2ecc71", width=20)
    
    # Expenses (Negative)
    total_exp_monthly = df_monthly["Exp_Feed"] + df_monthly["Exp_Variable"] + df_monthly["Exp_Admin"] + df_monthly["Exp_Overhead"] + df_monthly["Exp_Labor"] + df_monthly["Exp_Shock"]
    ax.bar(df_monthly.index, -df_monthly["Exp_Feed"], label="Krmivo", color="#e67e22", width=20)
    ax.bar(df_monthly.index, -df_monthly["Exp_Variable"], bottom=-df_monthly["Exp_Feed"], label="Veterina+Seč", color="#9b59b6", width=20)
    ax.bar(df_monthly.index, -df_monthly["Exp_Admin"], bottom=-(df_monthly["Exp_Feed"]+df_monthly["Exp_Variable"]), label="Administrativa", color="#7f8c8d", width=20)
    ax.bar(df_monthly.index, -df_monthly["Exp_Overhead"], bottom=-(df_monthly["Exp_Feed"]+df_monthly["Exp_Variable"]+df_monthly["Exp_Admin"]), label="Režie", color="#3498db", width=20)
    ax.bar(df_monthly.index, -df_monthly["Exp_Labor"], bottom=-(df_monthly["Exp_Feed"]+df_monthly["Exp_Variable"]+df_monthly["Exp_Admin"]+df_monthly["Exp_Overhead"]), label="Práce", color="#1abc9c", width=20)
    ax.bar(df_monthly.index, -df_monthly["Exp_Shock"], bottom=-(df_monthly["Exp_Feed"]+df_monthly["Exp_Variable"]+df_monthly["Exp_Admin"]+df_monthly["Exp_Overhead"]+df_monthly["Exp_Labor"]), label="Šoky", color="#e74c3c", width=20)
    
    ax.axhline(0, color="white", linewidth=1)
    ax.set_ylabel("CZK")
    ax.set_xlabel("Měsíc")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.2)
    st.pyplot(fig)

with col_pie:
    st.markdown("**Struktura nákladů**")
    total_exp = {
        "Krmivo": df["Exp_Feed"].sum(),
        "Veterina+Seč": df["Exp_Variable"].sum(),
        "Administrativa": df["Exp_Admin"].sum(),
        "Režie": df["Exp_Overhead"].sum(),
        "Práce": df["Exp_Labor"].sum(),
        "Šoky": df["Exp_Shock"].sum()
    }
    fig_pie, ax_pie = plt.subplots(figsize=(6, 5))
    colors_pie = ["#e67e22", "#9b59b6", "#3498db", "#1abc9c", "#e74c3c"]
    ax_pie.pie(list(total_exp.values()), labels=list(total_exp.keys()), autopct='%1.1f%%', colors=colors_pie, startangle=90)
    st.pyplot(fig_pie)

# --- 5. SEASONAL ANALYSIS ---
st.subheader("Sezónní analýza")

col_season, col_price = st.columns(2)

with col_season:
    st.markdown("**Průměrný Denní Cashflow po Měsících**")
    
    df_month = df.copy()
    df_month["Month"] = df_month.index.month
    df_month["Daily_Flow"] = df_month["Income"] - (df_month["Exp_Feed"] + df_month["Exp_Variable"] + df_month["Exp_Admin"] + df_month["Exp_Overhead"] + df_month["Exp_Labor"] + df_month["Exp_Shock"])
    
    seasonal = df_month.groupby("Month")["Daily_Flow"].mean()
    
    fig_seas, ax_seas = plt.subplots(figsize=(10, 4))
    colors_seas = ["🟢" if x > 0 else "🔴" for x in seasonal]
    color_list = ["#2ecc71" if x > 0 else "#e74c3c" for x in seasonal]
    
    ax_seas.bar(seasonal.index, seasonal.values, color=color_list, edgecolor="white", linewidth=1)
    ax_seas.axhline(y=0, color="white", linestyle="-", linewidth=1)
    ax_seas.set_xticks(range(1, 13))
    ax_seas.set_ylabel("Denní Tok (CZK)", fontsize=10)
    ax_seas.set_xlabel("Měsíc", fontsize=10)
    ax_seas.grid(True, alpha=0.2, axis='y')
    
    st.pyplot(fig_seas)

with col_price:
    st.markdown("**Volatilita ceny masa**")
    
    fig_price, ax_price = plt.subplots(figsize=(10, 4))
    sns.histplot(df["Meat_Price"], kde=True, ax=ax_price, color="#9b59b6", bins=30)
    ax_price.axvline(cfg.price_meat_avg, color="white", linestyle="--", linewidth=2, label="Průměr")
    ax_price.set_xlabel("Cena Masa (Kč/kg)", fontsize=10)
    ax_price.set_ylabel("Výskyt", fontsize=10)
    ax_price.legend()
    
    st.pyplot(fig_price)

# --- 6. FEEDING EFFICIENCY ---
st.subheader("Účinnost krmení")

col_feed_chart, col_feed_info = st.columns([2, 1])

with col_feed_chart:
    total_days = model.feed_log["Grazing"] + model.feed_log["Stored"] + model.feed_log["Market"]
    
    grazing_pct = (model.feed_log["Grazing"] / total_days * 100) if total_days > 0 else 0
    stored_pct = (model.feed_log["Stored"] / total_days * 100) if total_days > 0 else 0
    market_pct = (model.feed_log["Market"] / total_days * 100) if total_days > 0 else 0
    
    fig_feed, ax_feed = plt.subplots(figsize=(10, 5))
    
    feed_data = [model.feed_log["Grazing"], model.feed_log["Stored"], model.feed_log["Market"]]
    feed_labels = [f"Pastva\n({grazing_pct:.0f}%)", f"Seno\n({stored_pct:.0f}%)", f"Nákup\n({market_pct:.0f}%)"]
    feed_colors = ["#2ecc71", "#f39c12", "#e74c3c"]
    
    bars = ax_feed.barh(feed_labels, feed_data, color=feed_colors, edgecolor="white", linewidth=2)
    
    for bar, value in zip(bars, feed_data):
        ax_feed.text(value/2, bar.get_y() + bar.get_height()/2, f"{int(value)}d", 
                    ha="center", va="center", fontsize=12, fontweight="bold", color="white")
    
    ax_feed.set_xlabel("Dny v Roce", fontsize=11, fontweight="bold")
    ax_feed.grid(axis='x', alpha=0.2)
    
    st.pyplot(fig_feed)

with col_feed_info:
    st.markdown("**Interpretace**")
    st.markdown(f"""
     **Pastva**: {grazing_pct:.0f}%
    - Nejlevnější (~0.2 Kč/kg)
    - Ideální pro léto
    
     **Vlastní Seno**: {stored_pct:.0f}%
    - Zásoba z jara
    - Cena: 50 Kč/balík
    
    **Tržní Nákup**: {market_pct:.0f}%
    - Pokud % > 20% 
    - Nákup v zimě: 800 Kč/balík
    - Nákup v létě: 400 Kč/balík
    """)
    
    if market_pct > 30:
        st.error(f" Vysoký podíl nákupu ({market_pct:.0f}%)! Zvětšete louky nebo zmenšete stádo.")
    elif market_pct > 15:
        st.warning(f" Nákup ({market_pct:.0f}%). Zvažte optimalizaci.")
    else:
        st.success(f" Excelentní ({market_pct:.0f}%). Autosuficience!")


# --- 6.b BCS EVOLUTION ---
st.subheader("📉 Vývoj Kondice (BCS)")

fig_bcs, ax_bcs = plt.subplots(figsize=(12, 4))
ax_bcs.plot(df.index, df["BCS"], color="#e67e22", linewidth=2, label="BCS Průměr")

# Add optimal zone (2.5 - 3.5)
ax_bcs.axhspan(2.5, 3.5, color='#2ecc71', alpha=0.2, label="Optimální zóna")
ax_bcs.axhline(2.0, color="red", linestyle="--", alpha=0.5, label="Podvýživa")

ax_bcs.set_ylabel("Body Condition Score (1-5)", color="white")
ax_bcs.set_ylim(1.5, 4.5)
ax_bcs.grid(True, alpha=0.2)
ax_bcs.legend(loc="upper right")
ax_bcs.tick_params(axis='y', labelcolor="white")
ax_bcs.tick_params(axis='x', labelcolor="white")

st.pyplot(fig_bcs)
st.caption("BCS ovlivňuje plodnost a mortalitu. Cíl je držet se v zelené zóně (2.5 - 3.5). Pod 2.0 hrozí úhyn a neplodnost.")

# --- 7. EVENT LOG ---
st.markdown("---")

with st.expander("📜 Deník Farmáře (Events)", expanded=False):
    st.markdown("**Posledních 30 záznamů:**")
    for event in model.event_log[-30:]:
        st.text(event)

# --- 8. BENCHMARK COMPARISON (Detailed Validation) ---
st.markdown("---")
st.subheader("✅ Komplexní Validace (Model vs. Realita ČR)")

# 1. Benchmark Data (Zdroje: SCHOK, ÚZEI, FADN)
benchmark_data = {
    "1. Krmivo (Kč/ks)": 1750,
    "2. Veterina a Služby (Kč/ks)": 500,
    "3. Režie a Opravy (Kč/ks)": 1500,
    "4. Tržby Maso (Kč/ks)": 2900,
    "5. Zisk bez dotací (Kč/ks)": -1150,
    "6. Odchov (ks jehňat/matku)": 1.35,
    "7. Závislost na dotacích (%)": 65.0
}

# 2. Calculate model metrics
avg_ewes = df["Ewes"].mean()
if avg_ewes == 0: avg_ewes = 1
years = cfg.sim_years

# Economics per ewe
# OPRAVA 1: Použití df.index.month místo get_level_values
oct_income = df[df.index.month == 10]["Income"].sum()

model_feed = df["Exp_Feed"].sum() / (avg_ewes * years)

# Rozdělení nákladů
model_vet_services = (df["Exp_Vet"].sum() + df["Exp_Shearing"].sum() + df["Exp_RamPurchase"].sum()) / (avg_ewes * years)
model_overhead_real = (df["Exp_Mow"].sum() + df["Exp_Overhead"].sum() + df["Exp_Shock"].sum() + df["Exp_Labor"].sum()) / (avg_ewes * years)

# Meat Income
model_meat = (oct_income / (avg_ewes * years)) if oct_income > 0 else (df["Income"].sum() / (avg_ewes * years))
model_profit_no_sub = model_meat - (model_feed + model_vet_services + model_overhead_real)

# OPRAVA 2: Sečtení jehňat pro validaci (sloupec "Lambs" neexistuje)
lambs_total_series = df["Lambs Male"] + df["Lambs Female"]
avg_lamb_peak = lambs_total_series[df.index.month == 6].mean()
model_rearing = avg_lamb_peak / avg_ewes if avg_ewes > 0 else 0

# Subsidy dependence
total_income = df["Income"].sum()
subsidy_income = df[(df.index.month == 4) | (df.index.month == 11)]["Income"].sum()
model_subsidy_dep = (subsidy_income / total_income * 100) if total_income > 0 else 0

# 3. Create comparison dataframe
validation_df = pd.DataFrame({
    "Metrika": list(benchmark_data.keys()),
    "Průměr ČR (Realita)": list(benchmark_data.values()),
    "Tvůj Model": [model_feed, model_vet_services, model_overhead_real, model_meat, model_profit_no_sub, model_rearing, model_subsidy_dep]
})

# Calculate difference
validation_df["Odchylka"] = validation_df["Tvůj Model"] - validation_df["Průměr ČR (Realita)"]

# Display table
st.markdown("###  Detailní Srovnání")
st.dataframe(validation_df.style.format("{:,.0f}", subset=["Průměr ČR (Realita)", "Tvůj Model", "Odchylka"]), use_container_width=True, height=300)
