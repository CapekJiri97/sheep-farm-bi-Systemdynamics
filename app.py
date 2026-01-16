import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
import time

# --- CONFIGURATION ---
st.set_page_config(page_title="Sheep Farm - System Dynamics", layout="wide", page_icon="🚜")
plt.style.use('dark_background')

# --- SESSION STATE INIT ---
if 'custom_scenarios' not in st.session_state:
    st.session_state['custom_scenarios'] = {}

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
    
    # Market Loop (Tiered Pricing)
    market_local_limit: int = 40        # Kolik ovcí prodám "ze dvora" za plnou cenu
    price_meat_wholesale: float = 55.0  # Výkupní cena pro nadprodukci (Kč/kg)
    
    # Ecological Loop
    pasture_degradation_rate: float = 0.01 # Kolik % zdraví zmizí denně při přetížení
    pasture_recovery_rate: float = 0.002   # Kolik % zdraví se vrátí denně při odpočinku
    
    # Delays (System Dynamics)
    delay_bcs_perception: int = 10   # Dny: Jak dlouho trvá, než si všimnu změny kondice (Informační zpoždění)
    delay_feed_delivery: int = 3     # Dny: Jak dlouho trvá dodání nakoupeného krmiva (Materiálové zpoždění)
    max_ewe_age: float = 8.0         # Roky: Kdy jde ovce do "důchodu" (Pipeline Delay exit)
    
    # 3. STRATEGY (New!)
    machinery_mode: str = "Services" # "Services" or "Own"
    climate_profile: str = "Normal"  # "Normal", "Dry", "Mountain"
    include_labor_cost: bool = False
    
    # Weather Scenarios (Overrides)
    rain_growth_global_mod: float = 1.0  # 1.0 = default dle profilu
    drought_prob_add: float = 0.0        # +0.0 = default
    winter_len_global_mod: float = 1.0   # 1.0 = default
    
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
        # OPTIMALIZACE: Věk jako numpy array pro vektorizaci
        self.ewe_ages = np.full(cfg.initial_ewes, 3.0, dtype=np.float32)
        self.rams_breeding = max(1, int(cfg.initial_ewes / 30))
        self.ram_age = 3.0
        self.lambs_male = 0
        self.lambs_female = 0
        self.lamb_age = 0.0
        
        # --- ASSETS ---
        self.cash = cfg.capital
        self.hay_stock_bales = cfg.initial_hay_bales
        self.bcs = 3.0
        self.pasture_health = 1.0 # 1.0 = 100% zdravá pastvina, 0.0 = poušť
        self.perceived_bcs = 3.0  # To, co si farmář myslí, že je BCS (Informační zpoždění)
        self.feed_orders = []     # Fronta objednávek: list of tuples (delivery_date, amount)
        self.pregnant_ewes = 0    # Počet březích bahnic (Gestační zpoždění)
        self.yearly_age_snapshots = {} # Pro UI graf věkové struktury v čase
        
        # --- WEATHER STATE (Autocorrelation) ---
        self.weather_regime = 1.0  # 1.0 = normál, <1 sucho, >1 vlhko (Týdenní trend)
        self.weather_timer = 0     # Kolik dní ještě trvá tento režim
        
        # --- CLIMATE SETUP ---
        if cfg.climate_profile == "Dry":
            base_grass = 0.7
            base_drought = 0.02
            base_winter = 0.8
        elif cfg.climate_profile == "Mountain":
            base_grass = 1.2
            base_drought = 0.001
            base_winter = 1.3
        else:
            base_grass = 1.0
            base_drought = 0.005
            base_winter = 1.0
            
        # Aplikace uživatelských scénářů (Overrides)
        self.grass_mod = base_grass * cfg.rain_growth_global_mod
        self.drought_chance = max(0.0, min(1.0, base_drought + cfg.drought_prob_add))
        self.winter_len_mod = base_winter * cfg.winter_len_global_mod

        self.is_winter = True
        self.winter_end_day = int(80 * self.winter_len_mod)
        self.winter_start_day = int(365 - (60 * self.winter_len_mod))
        
        # --- LAND ---
        self.area_meadow = cfg.land_area * cfg.meadow_share
        self.area_pasture = cfg.land_area * (1 - cfg.meadow_share)
        
        self.event_log = []
        self.feed_log = {"Grazing": 0, "Stored": 0, "Market": 0}
        
        # OPTIMALIZACE: Před-alokace historie (místo appendování dictů)
        self.total_steps = self.cfg.sim_years * 365
        self.dates = pd.date_range(start="2025-01-01", periods=self.total_steps, freq="D")
        # Předvypočítané kalendářní údaje pro rychlý přístup
        self.months = self.dates.month.values
        self.days = self.dates.day.values
        self.day_of_years = self.dates.dayofyear.values
        
        # Numpy pole pro historii (mnohem rychlejší zápis)
        self.h_cols = ["Cash", "Ewes", "Lambs", "Lambs Male", "Lambs Female", "Total Animals", 
                       "Hay Stock", "Income", "Exp_Feed", "Exp_Vet", "Exp_Machinery", "Exp_Mow", 
                       "Exp_Shearing", "Exp_RamPurchase", "Exp_Admin", "Exp_Labor", "Labor Hours", 
                       "Exp_Overhead", "Exp_Shock", "Exp_Variable", "BCS", "Meat_Price", 
                       "Pasture_Health", "Perceived_BCS", "Weather_Regime", "Is_Winter", "Is_Drought",
                       "Inc_Meat", "Inc_Hay", "Inc_Subsidy", "Sold_Animals", "Sold_Hay"]
        self.history_store = {col: np.zeros(self.total_steps, dtype=np.float32) for col in self.h_cols}
        self.feed_source_store = [None] * self.total_steps
        
        # Grass curve (Month 1-12)
        self.grass_curve = {1:0, 2:0, 3:0.1, 4:0.5, 5:1.2, 6:1.1, 7:0.8, 8:0.6, 9:0.8, 10:0.4, 11:0.1, 12:0}

    def _check_barn_capacity(self):
        max_volume = self.cfg.hay_barn_area_m2 * 3.0
        max_bales = int(max_volume / self.cfg.bale_volume_m3)
        if self.hay_stock_bales > max_bales:
            excess = self.hay_stock_bales - max_bales
            income = excess * self.cfg.price_bale_sell_summer
            # self.cash += income # Cash update moved to step() via income aggregation
            self.hay_stock_bales = max_bales
            self.event_log.append(f"{self.date.date()}: ⚠️ Seník plný! Prodáno {int(excess)} balíků.")
            return income, excess
        return 0.0, 0.0

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

    def step(self, t):
        # OPTIMALIZACE: Použití předvypočítaných hodnot z numpy polí
        self.date = self.dates[t] # Pro logování a kompatibilitu
        month = self.months[t]
        day = self.day_of_years[t]
        
        # --- AGING (COHORT DELAY) ---
        # OPTIMALIZACE: Vektorové stárnutí
        self.ewe_ages += (1/365.0)
        self.ram_age += (1/365.0)
        
        # --- 0. SYSTEM DYNAMICS: DELAYS ---
        # A) Informační zpoždění (Perception Delay)
        # Farmář nevidí aktuální BCS, ale "klouzavý průměr" za posledních X dní.
        # Exponential smoothing: New = Old + alpha * (Target - Old)
        alpha = 1.0 / max(1, self.cfg.delay_bcs_perception)
        self.perceived_bcs = (self.perceived_bcs * (1 - alpha)) + (self.bcs * alpha)
        
        # --- WEATHER REGIME UPDATE (Autocorrelation) ---
        if self.weather_timer <= 0:
            # Losujeme nové počasí na 5-10 dní (týdenní trend)
            # Generujeme číslo kolem 1.0 (např. 0.7 = suchý týden, 1.3 = mokrý týden)
            self.weather_regime = np.random.normal(1.0, 0.25) 
            self.weather_regime = max(0.5, min(1.5, self.weather_regime)) # Omezení extrémů
            self.weather_timer = np.random.randint(5, 10)
        
        self.weather_timer -= 1
        
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

        # Process Feed Arrivals (Material Delay Resolution)
        # Orders structure: list of {"date": Timestamp, "amount": float}
        arrived_orders = [o for o in self.feed_orders if o["date"] <= self.date]
        self.feed_orders = [o for o in self.feed_orders if o["date"] > self.date]
        for order in arrived_orders:
            self.hay_stock_bales += order["amount"]
            self.event_log.append(f"{self.date.date()}: 🚚 Dorazilo krmivo ({int(order['amount'])} balíků).")

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
            # Riziko sucha je mnohem vyšší, pokud je "suchý týden" (regime < 0.8)
            current_drought_prob = self.drought_chance
            if self.weather_regime < 0.8:
                current_drought_prob *= 5.0 # 5x vyšší šance v suchém týdnu
            elif self.weather_regime > 1.2:
                current_drought_prob *= 0.1 # V deštivém týdnu sucho nehrozí

            if np.random.random() < current_drought_prob:
                is_drought = True
                self.event_log.append(f"{self.date.date()}: ☀️ Sucho! Tráva neroste.")

        # --- FEEDING LOGIC WITH DELAYS ---
        # Automatické objednávání (Reorder Point)
        # Pokud zásoby klesnou pod 3 dny spotřeby, objednáme na týden dopředu.
        daily_bales_needed = (demand_kg * 1.2) / self.cfg.bale_weight_kg
        pending_bales = sum(o["amount"] for o in self.feed_orders)
        
        if (self.hay_stock_bales + pending_bales) < (daily_bales_needed * 3):
            # Objednáváme
            order_amount = daily_bales_needed * 7
            delivery_date = self.date + pd.Timedelta(days=self.cfg.delay_feed_delivery)
            cost = order_amount * get_stochastic_value(self.cfg.price_bale_sell_winter, 100)
            
            if self.cash > cost:
                self.cash -= cost
                self.feed_orders.append({"date": delivery_date, "amount": order_amount})
                # self.event_log.append(f"{self.date.date()}: 🛒 Objednáno krmivo (příjezd za {self.cfg.delay_feed_delivery} dny).")
            else:
                # Nemáme na to -> krizový management (prodej zvířat?) - zatím nic, prostě hlad
                pass

        if self.is_winter or is_drought:
            # Feeding Hay
            needed_bales = (demand_kg * 1.2) / self.cfg.bale_weight_kg # 20% waste
            fed = min(self.hay_stock_bales, needed_bales)
            self.hay_stock_bales -= fed
            feed_cost = fed * 50 # Handling
            
            if fed < needed_bales:
                # DOŠLO KRMIVO A NOVÉ JEŠTĚ NEDORAZILO (Materiálové zpoždění)
                # Ovce hladoví výrazněji
                self.bcs = max(1.5, self.bcs - 0.005) # Hunger penalty
                feed_source = "Starvation (Wait for Delivery)"
            else:
                self.bcs = max(2.5, self.bcs - 0.001) # Maintenance
                feed_source = "Stored"
        else:
            # Grazing
            growth = self.grass_curve[month] * self.grass_mod
            # EKOLOGICKÁ SMYČKA: Růst závisí na zdraví pastviny
            growth *= self.pasture_health
            # AUTOKORELACE: Aplikace týdenního režimu + menší denní šum
            growth *= self.weather_regime * np.random.normal(1.0, 0.1)
            
            avail = self.area_pasture * 35.0 * growth * np.random.normal(1.0, 0.2)
            
            # --- PASTURE PROTECTION (Ochrana pastviny) ---
            # Pokud je zdraví kritické (< 50%) a máme seno, nepustíme ovce na pastvu (regenerace).
            force_hay = False
            if self.pasture_health < 0.5 and self.hay_stock_bales > 5:
                force_hay = True
            
            # Výpočet tlaku na pastvinu (Grazing Pressure)
            if force_hay:
                pressure = 0.0 # Pastvina odpočívá, nikdo ji nežere
            elif avail > 0:
                pressure = demand_kg / avail
            else:
                pressure = 10.0 # Extrémní tlak, když nic neroste

            # Update Pasture Health (Ecological Loop)
            if pressure > 0.95: # Ovce sežerou > 95% trávy -> degradace
                damage = (pressure - 0.95) * self.cfg.pasture_degradation_rate
                self.pasture_health = max(0.1, self.pasture_health - damage)
            elif pressure < 0.5: # Pastvina odpočívá -> regenerace
                self.pasture_health = min(1.0, self.pasture_health + self.cfg.pasture_recovery_rate)

            # Rozhodování o přikrmování na pastvě (Informační zpoždění)
            # Farmář se rozhoduje podle PERCEIVED BCS, ne podle skutečného.
            # Pokud si myslí, že jsou tlusté (Perceived > 3.2), nepřikrmuje, i když tráva dochází.
            wants_to_supplement = (self.perceived_bcs < 3.2)

            # Efektivní dostupná tráva (pokud chráníme pastvinu, je pro ovce 0)
            eff_avail = 0.0 if force_hay else avail

            if eff_avail >= demand_kg:
                self.bcs = min(4.0, self.bcs + 0.004)
                feed_cost = demand_kg * 0.2 # Salt/Water
                feed_source = "Grazing"
            elif wants_to_supplement or force_hay:
                # Supplement (Pokud farmář vidí potřebu NEBO je pastvina zavřená)
                deficit = demand_kg - eff_avail
                needed_bales = (deficit * 1.4) / self.cfg.bale_weight_kg
                fed = min(self.hay_stock_bales, needed_bales)
                self.hay_stock_bales -= fed
                
                feed_cost = (eff_avail * 0.2) + (fed * 50)
                
                if fed < needed_bales: 
                    # Došlo i seno na přikrmení
                    self.bcs -= 0.003
                    feed_source = "Grazing+Starvation" if not force_hay else "Starvation (No Hay)"
                else:
                    feed_source = "Grazing+Stored" if not force_hay else "Stored (Pasture Rest)"
            else:
                # Farmář si myslí, že jsou OK, tak nepřikrmuje, i když je málo trávy
                # "Necháme je vyžrat nedopasky"
                self.bcs -= 0.002 # Skutečná kondice klesá
                feed_source = "Grazing (No Supplement)"
        
        self.feed_log[feed_source] = self.feed_log.get(feed_source, 0) + 1

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
        inc_meat = 0.0
        inc_hay = 0.0
        inc_subsidy = 0.0
        sold_animals = 0
        sold_hay = 0.0
        var_cost = 0.0
        
        # --- REPRODUCTION PIPELINE (Gestation Delay) ---
        
        # 1. MATING (Říjen) - Začátek zpoždění
        # Rozhoduje se o potenciálu. Pokud jsou hubené TEĎ, v březnu jehňata nebudou.
        if month == 10 and self.date.day == 1:
            conception_rate = 0.95 if self.bcs > 3.0 else (0.7 if self.bcs > 2.5 else 0.3)
            self.pregnant_ewes = int(self.ewes * conception_rate)
            self.event_log.append(f"{self.date.date()}: 🐏 Připouštění. Březích: {self.pregnant_ewes} ks (BCS {self.bcs:.2f}).")

        # 2. GESTATION RISK (Zima) - Průběh zpoždění
        # Pokud v zimě hladoví, potratí.
        if self.pregnant_ewes > 0 and self.bcs < 2.0:
            abortions = int(self.pregnant_ewes * 0.05) # 5% denně při extrémním hladu
            self.pregnant_ewes = max(0, self.pregnant_ewes - abortions)

        # 3. LAMBING (Březen) - Konec zpoždění
        # Rodí se to, co bylo počato v říjnu a přežilo zimu.
        if month == 3:
            # Denně rodí cca 1/30 ze zbývajících březích
            born_today_mothers = np.random.binomial(self.pregnant_ewes, 0.1) 
            if born_today_mothers > 0:
                self.pregnant_ewes -= born_today_mothers
                f = get_stochastic_value(self.cfg.fertility_mean, self.cfg.fertility_std)
                new_lambs = int(born_today_mothers * f)
                self.lambs_male += int(new_lambs / 2)
                self.lambs_female += (new_lambs - int(new_lambs / 2))
                self.bcs -= (born_today_mothers / max(1, self.ewes)) * 0.4 # Kojení vyčerpává
                # self.event_log.append(f"{self.date.date()}: 🍼 Narozeno {new_lambs} jehňat.")

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
            h_inc, h_qty = self._check_barn_capacity()
            inc_hay += h_inc
            sold_hay += h_qty
            self.event_log.append(f"{self.date.date()}: 🚜 Seč ({self.cfg.machinery_mode}). {int(bales)} balíků.")

        # Sales (October)
        if month == 10 and self.date.day == 15:
            # Planner logic
            if self.cfg.enable_forecasting:
                forecast = self._perform_forecast()
                projected = self.cash - forecast
                if projected < 0 and self.hay_stock_bales > 0:
                    needed = int(abs(projected) / self.cfg.price_bale_sell_summer) + 1
                    sold_hay_planner = min(self.hay_stock_bales * 0.4, needed)
                    inc_hay += sold_hay_planner * self.cfg.price_bale_sell_summer
                    sold_hay += sold_hay_planner
                    self.hay_stock_bales -= sold_hay_planner
            
            # TRŽNÍ SMYČKA: Tiered Pricing
            # Máme limit pro lokální prodej (vysoká cena). Zbytek jde do výkupu (nízká cena).
            quota_remaining = self.cfg.market_local_limit
            
            def get_blended_price(count, retail_price):
                nonlocal quota_remaining
                if count <= 0: return 0.0
                local_part = min(count, quota_remaining)
                wholesale_part = count - local_part
                quota_remaining = max(0, quota_remaining - local_part)
                
                total_rev = (local_part * retail_price) + (wholesale_part * self.cfg.price_meat_wholesale)
                return total_rev / count # Průměrná cena za kg pro tuto várku
            
            # 1. Sell Males
            p_male = get_blended_price(self.lambs_male, base_meat_price)
            inc_meat += self.lambs_male * 40 * p_male
            sold_animals += self.lambs_male
            self.lambs_male = 0
            
            # 2. Renew Females (S respektem ke kapacitě ovčína)
            # CULLING LOGIC (Aging Chain Exit)
            # Vyřadíme ty, co jsou starší než limit (Pipeline exit)
            # OPTIMALIZACE: Numpy maskování
            old_mask = self.ewe_ages > self.cfg.max_ewe_age
            old_indices = np.where(old_mask)[0]
            
            # Pokud je starých málo, vyřadíme i nějaké náhodné (nemoc, úraz), abychom drželi 15% obnovu
            target_cull = int(self.ewes * 0.15)
            needed_random_cull = max(0, target_cull - len(old_indices))
            
            # Vytvoříme masku pro zachování (True = nechat)
            keep_mask = ~old_mask
            
            # Náhodné vyřazení do počtu
            candidates = np.where(keep_mask)[0]
            if len(candidates) > needed_random_cull:
                random_culls = np.random.choice(candidates, needed_random_cull, replace=False)
                keep_mask[random_culls] = False
            
            # Aplikace vyřazení
            self.ewe_ages = self.ewe_ages[keep_mask]
            cull_count = self.ewes - len(self.ewe_ages)
            self.ewes = len(self.ewe_ages) # Sync
            
            future_ewes = self.ewes
            
            # Kolik můžeme maximálně doplnit?
            max_new_capacity = self.cfg.barn_capacity - future_ewes
            
            # Chceme doplnit max. 80% jehniček, ale ne víc, než se vejde
            potential_keep = int(self.lambs_female * 0.8)
            keep = max(0, min(potential_keep, max_new_capacity))
            
            # Zbytek prodáme
            sell_females = self.lambs_female - keep
            p_female = get_blended_price(sell_females, base_meat_price)
            inc_meat += sell_females * 35.0 * p_female
            sold_animals += sell_females
            
            # 3. Add kept lambs to herd (Pipeline Entry)
            # Jehničky vstupují do stáda ve věku cca 0.6 roku (7 měsíců)
            self.ewe_ages = np.concatenate([self.ewe_ages, np.full(keep, 0.6)])
            self.ewes = len(self.ewe_ages)
            
            self.lambs_female = 0
            # Staré ovce jdou rovnou za sníženou cenu (maso na klobásy), nečerpají kvótu na jehněčí
            inc_meat += cull_count * 60 * self.cfg.price_meat_wholesale 
            sold_animals += cull_count
            
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
                 self.ram_age = 2.0 # Omlazení beranů nákupem nových

            # 5. Fall Vet
            day_vet += (self.ewes + self.rams_breeding) * (self.cfg.cost_vet_base * vet_multiplier / 2)
            
            self.event_log.append(f"{self.date.date()}: 💰 Prodej. Příjem {int(inc_meat + inc_hay)}.")

        # Subsidies
        if month == 11 and self.date.day == 20:
            inc_subsidy += ((self.cfg.land_area * get_stochastic_value(self.cfg.subsidy_ha_mean, 200)) + (self.ewes * self.cfg.subsidy_sheep_mean)) * 0.7
        if month == 4 and self.date.day == 20:
             inc_subsidy += ((self.cfg.land_area * get_stochastic_value(self.cfg.subsidy_ha_mean, 200)) + (self.ewes * self.cfg.subsidy_sheep_mean)) * 0.3
        
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
        income = inc_meat + inc_hay + inc_subsidy
        
        total_out = feed_cost + var_cost + daily_overhead + labor_val + shock_val
        self.cash += income - total_out
        
        # Mortality (Simple daily check)
        mort_prob_ewe = (self.cfg.mortality_ewe_mean / 365)
        if self.bcs < 2.0: mort_prob_ewe *= 5
        elif self.bcs < 2.5: mort_prob_ewe *= 2
        
        deaths_ewes = np.random.binomial(self.ewes, mort_prob_ewe)
        
        # Uložení věkové struktury (každý měsíc)
        if self.date.day == 1:
            snapshot = []
            # 1. Bahnice
            for age in self.ewe_ages: # Iterace přes numpy array je ok pro snapshot jednou měsíčně
                snapshot.append({"Age": age, "Category": "Bahnice"})
            # 2. Berani
            for _ in range(self.rams_breeding):
                snapshot.append({"Age": self.ram_age, "Category": "Berani"})
            # 3. Jehňata (cca věk podle data v roce)
            l_age = max(0.0, (self.date.dayofyear - 75) / 365.0) if (self.lambs_male + self.lambs_female) > 0 else 0.0
            for _ in range(self.lambs_female):
                snapshot.append({"Age": l_age, "Category": "Jehničky"})
            for _ in range(self.lambs_male):
                snapshot.append({"Age": l_age, "Category": "Beránci"})
            
            self.yearly_age_snapshots[self.date] = snapshot
        
        if deaths_ewes > 0 and self.ewes > 0:
            # Náhodně odstraníme z věkového seznamu (úhyn není závislý na věku v tomto zjednodušení)
            death_indices = np.random.choice(len(self.ewe_ages), min(deaths_ewes, len(self.ewe_ages)), replace=False)
            self.ewe_ages = np.delete(self.ewe_ages, death_indices)
            self.ewes = len(self.ewe_ages)
        
        if self.lambs_male + self.lambs_female > 0:
            mort_prob_lamb = (self.cfg.mortality_lamb_mean / 365) * (2 if self.bcs < 2.5 else 1)
            self.lambs_male = max(0, self.lambs_male - np.random.binomial(self.lambs_male, mort_prob_lamb))
            self.lambs_female = max(0, self.lambs_female - np.random.binomial(self.lambs_female, mort_prob_lamb))

        # OPTIMALIZACE: Zápis do numpy polí místo appendování dictu
        self.history_store["Cash"][t] = self.cash
        self.history_store["Ewes"][t] = self.ewes
        self.history_store["Lambs"][t] = self.lambs_male + self.lambs_female
        self.history_store["Lambs Male"][t] = self.lambs_male
        self.history_store["Lambs Female"][t] = self.lambs_female
        self.history_store["Total Animals"][t] = self.ewes + self.rams_breeding + self.lambs_male + self.lambs_female
        self.history_store["Hay Stock"][t] = self.hay_stock_bales
        self.history_store["Income"][t] = income
        self.history_store["Inc_Meat"][t] = inc_meat
        self.history_store["Inc_Hay"][t] = inc_hay
        self.history_store["Inc_Subsidy"][t] = inc_subsidy
        self.history_store["Sold_Animals"][t] = sold_animals
        self.history_store["Sold_Hay"][t] = sold_hay
        self.history_store["Exp_Feed"][t] = feed_cost
        self.history_store["Exp_Vet"][t] = day_vet
        self.history_store["Exp_Machinery"][t] = day_machinery
        self.history_store["Exp_Mow"][t] = day_mow
        self.history_store["Exp_Shearing"][t] = day_shearing
        self.history_store["Exp_RamPurchase"][t] = day_ram_purchase
        self.history_store["Exp_Admin"][t] = day_admin
        self.history_store["Exp_Labor"][t] = labor_val
        self.history_store["Labor Hours"][t] = daily_hours
        self.history_store["Exp_Overhead"][t] = daily_overhead
        self.history_store["Exp_Shock"][t] = shock_val
        self.history_store["Exp_Variable"][t] = day_vet + day_mow + day_shearing + day_ram_purchase + day_machinery
        self.history_store["BCS"][t] = self.bcs
        self.history_store["Meat_Price"][t] = base_meat_price
        self.history_store["Pasture_Health"][t] = self.pasture_health
        self.history_store["Perceived_BCS"][t] = self.perceived_bcs
        self.history_store["Weather_Regime"][t] = self.weather_regime
        self.feed_source_store[t] = feed_source
        self.history_store["Is_Winter"][t] = 1 if self.is_winter else 0
        self.history_store["Is_Drought"][t] = 1 if is_drought else 0

    def run(self):
        # OPTIMALIZACE: Cyklus přes indexy
        for t in range(self.total_steps): 
            self.step(t)
        
        # Vytvoření DataFrame až na konci z numpy polí
        df = pd.DataFrame(self.history_store, index=self.dates)
        df["Feed_Source"] = self.feed_source_store
        df.index.name = "Date"
        return df

# --- MONTE CARLO DEFINITIONS ---
# 1. BASELINE (Výchozí hodnoty pro všechny scénáře - "Průměrná farma")
BASE_SCENARIO = {
    # Škála
    "sim_years": 5, "land_area": 30.0, "meadow_share": 0.25, "barn_capacity": 200, "initial_ewes": 150,
    "barn_area_m2": 400.0, "hay_barn_area_m2": 200.0,
    # Ekonomika
    "capital": 250000.0, "price_meat_avg": 85.0, "meat_price_std": 10.0,
    "market_local_limit": 40, "price_meat_wholesale": 55.0,
    "initial_hay_bales": 50.0, "price_bale_sell_winter": 1200.0, "price_bale_sell_summer": 600.0,
    "price_ram_purchase": 10000.0,
    # Strategie
    "machinery_mode": "Services", "climate_profile": "Normal", "include_labor_cost": True,
    "rain_growth_global_mod": 1.0, "drought_prob_add": 0.0, "winter_len_global_mod": 1.0,
    "safety_margin": 0.2, "enable_forecasting": True,
    # Dynamika
    "pasture_degradation_rate": 0.01, "pasture_recovery_rate": 0.002,
    "delay_bcs_perception": 10, "delay_feed_delivery": 3, "max_ewe_age": 8.0,
    # Náklady
    "cost_feed_own_mean": 2.5, "cost_feed_market_mean": 8.0,
    "cost_vet_base": 350.0, "cost_shearing": 50.0,
    "admin_base_cost": 5000.0, "admin_complexity_factor": 1.5,
    # Stroje (Own)
    "own_machine_capex": 1000000.0, "own_machine_life": 10.0,
    "own_mow_fuel_ha": 400.0, "own_bale_material": 50.0,
    "machinery_repair_mean": 20000.0, "machinery_failure_prob_daily": 0.0001,
    # Služby
    "service_mow_ha": 3000.0, "service_bale_pcs": 200.0,
    # Biologie
    "fertility_mean": 1.5, "mortality_lamb_mean": 0.10, "mortality_ewe_mean": 0.04,
    "feed_intake_ewe": 2.2, "hay_yield_ha_mean": 12.0, "bale_weight_kg": 250.0,
    # Dotace
    "subsidy_ha_mean": 8500.0, "subsidy_sheep_mean": 603.0,
    "tax_land_ha": 500.0, "tax_building_m2": 15.0,
    # Práce
    "wage_hourly": 200.0, "labor_hours_per_ewe_year": 8.0,
    "labor_hours_per_ha_year": 10.0, "labor_hours_fix_year": 200.0
}

# --- DEFINICE 5 STRATEGICKÝCH SCÉNÁŘŮ ---

SCENARIOS = {
    # 1. REFERENČNÍ SCÉNÁŘ
    "1. Rodinný Ideál (Baseline)": {
        **BASE_SCENARIO,
        "land_area": 40.0, 
        "initial_ewes": 200, 
        "barn_capacity": 250, 
        "capital": 800000,
        # Strategie
        "machinery_mode": "Own",           
        "include_labor_cost": True,        
        "climate_profile": "Normal",
        # Parametry
        "admin_complexity_factor": 1.2,    
        "labor_hours_per_ewe_year": 7.0    
    },

    # 2. SCÉNÁŘ MALÉ ŠKÁLY
    "2. Hobby Zahrada (Small Scale)": {
        **BASE_SCENARIO,
        "land_area": 5.0, 
        "initial_ewes": 25, 
        "barn_capacity": 30, 
        "capital": 100000,
        # Strategie
        "machinery_mode": "Services",      
        "include_labor_cost": True,        
        "climate_profile": "Normal",
        # Parametry
        "admin_base_cost": 500,            
        "labor_hours_per_ewe_year": 12.0   
    },

    # 3. SCÉNÁŘ DISECONOMIES OF SCALE
    "3. Agro Moloch (Diseconomy)": {
        **BASE_SCENARIO,
        "land_area": 300.0, 
        "initial_ewes": 1500, 
        "barn_capacity": 1600, 
        "barn_area_m2": 4000, # Větší budova = vyšší údržba
        "capital": 10000000,
        # Strategie
        "machinery_mode": "Own",
        "own_machine_capex": 6000000,      
        "include_labor_cost": True,
        # Parametry
        "admin_base_cost": 150000,         
        "admin_complexity_factor": 1.8,    
        "labor_hours_fix_year": 2500,      
        "wage_hourly": 250                 
    },

    # 4. SCÉNÁŘ ENVIRONMENTÁLNÍHO STRESU
    "4. Klimatická Krize (Stress Test)": {
        **BASE_SCENARIO,
        "land_area": 50.0, 
        "initial_ewes": 200, 
        "barn_capacity": 250,
        "capital": 500000,
        # Strategie
        "climate_profile": "Dry",          
        # Parametry krize
        "rain_growth_global_mod": 0.6,     
        "drought_prob_add": 0.15,          
        "hay_yield_ha_mean": 6.0,          
        "price_bale_sell_winter": 1500.0   
    },

    # 5. SCÉNÁŘ RIZIKA A STRATEGIE
    "5. Vrakoviště (High Risk)": {
        **BASE_SCENARIO,
        "land_area": 60.0, 
        "initial_ewes": 300, 
        "barn_capacity": 350,
        "capital": 100000,                 
        # Strategie
        "machinery_mode": "Own",
        # Parametry
        "own_machine_capex": 250000,       
        "own_machine_life": 5.0,           
        "machinery_repair_mean": 120000,   
        "machinery_failure_prob_daily": 0.005, 
        "safety_margin": 0.05              
    }
}


# --- SIDEBAR UI ---
with st.sidebar:
    st.title("Ovčí farma")
    
    st.markdown("### Režim aplikace")
    mode_switch = st.radio("Režim aplikace", ["Jednotlivá simulace", "Monte Carlo Lab"], horizontal=True, help="Přepne na hromadné testování scénářů.", label_visibility="collapsed")
    st.markdown("---")
    
    # Placeholder for Save Scenario UI (to be rendered after inputs are defined)
    save_sc_container = st.container()
    
    # --- TABS FOR BETTER UI ORGANIZATION ---
    tab_main, tab_strat, tab_details = st.tabs(["Zaklad", "Strategie", "Detaily"])
    
    with tab_main:
        st.header("1. Kapacita a Infrastruktura")
        target_ewes = st.slider("Cílová kapacita (ovčín)", 10, 500, 60, help="Maximální počet bahnic. Určuje velikost potřebné budovy.")
        
        req_m2 = int(target_ewes * 2.5) # 2.5 m2 per ewe
        barn_m2 = st.number_input("Velikost ovčína (m²)", 20, 2000, max(20, req_m2), help=f"Pro zvířata. Doporučeno: {req_m2} m² (2.5 m²/ks vč. jehňat a uliček)")
        hay_barn_m2 = st.number_input("Velikost seníku (m²)", 50, 2000, 100, help="Pro uskladnění sena. 100 m² pojme cca 200 balíků (při stohování 3m).")
        
        area = st.number_input("Celková plocha (ha)", 5.0, 100.0, 15.0)
        meadow_pct = st.slider("Podíl luk na seno (%)", 0, 100, 40, help="Část plochy jen na výrobu sena (pastva zakázana)")
        
        st.header("2. Stádo a ekonomika")
        start_ewes = st.slider("Počet bahnic (start)", 1, target_ewes, min(20, target_ewes), help="Kolik ovcí nakoupíte do začátku.")
        meat_price = st.slider("Maloobchodní cena (Retail) Kč/kg", 60.0, 150.0, 85.0, help="Cena pro lokální prodej (ze dvora).")
        start_hay = st.number_input("Počáteční zásoba sena (balíky)", 0, 500, 25)
        cap = st.number_input("Počáteční kapitál (CZK)", value=200000)
        labor_on = st.checkbox("Zapocitat mzdy", True, help="6h/rok na bahnici @ 200 Kč/h")

    with tab_strat:
        st.header("3. Pokročilé")
        climate = st.selectbox("Klimaticky profil", ["Normal", "Dry", "Mountain"])
        machinery = st.radio("Sec a lisovani", ["Services", "Own"], help="Services = pronájem; Own = vlastní stroj")
        use_forecast = st.toggle("Cashflow Planner", value=True)
        
        with st.expander("Ovladani Pocasi (Scenare)"):
            rain_mod = st.slider("Intenzita srážek/růstu (%)", 50, 150, 100, help="100% = normál dle profilu. Ovlivňuje rychlost růstu trávy.") / 100.0
            drought_add = st.slider("Riziko sucha (+%)", 0.0, 10.0, 0.0, 0.1, help="Zvyšuje pravděpodobnost sucha v létě.") / 100.0
            winter_mod = st.slider("Délka zimy (%)", 50, 150, 100, help="Prodlouží/zkrátí zimní období.") / 100.0
        
        with st.expander("Trzni Strategie (Velkoobchod)"):
            m_quota = st.number_input("Kapacita lokálního trhu (ks/rok)", 0, 500, 40, help="Kolik zvířat prodáte sousedům za plnou cenu.")
            m_wholesale = st.number_input("Velkoobchodní cena (Kč/kg)", 30.0, 80.0, 55.0, help="Cena pro výkup (jatka), když zahltíte lokální trh.")

        with st.expander("Systemova Dynamika (Zpozdeni)"):
            delay_bcs = st.slider("Zpoždění vnímání kondice (dny)", 1, 30, 10, help="Jak dlouho trvá, než si všimnete, že ovce hubnou.")
            delay_mat = st.slider("Zpoždění dodávky krmiva (dny)", 0, 14, 3, help="Za jak dlouho přijede kamion s krmivem po objednání.")

    with tab_details:
        st.header("Detailní nastavení parametrů")
        
        with st.expander("Biologie a Produkce"):
            p_fertility = st.number_input("Plodnost (ks/bahnici)", 1.0, 3.0, 1.5, 0.1)
            p_mortality_lamb = st.number_input("Úhyn jehňat (%)", 0.0, 50.0, 10.0, 1.0) / 100.0
            p_mortality_ewe = st.number_input("Úhyn bahnic (%)", 0.0, 20.0, 4.0, 0.5) / 100.0
            p_feed_ewe = st.number_input("Spotřeba bahnice (kg sušiny/den)", 1.0, 4.0, 2.2, 0.1)
            p_hay_yield = st.number_input("Výnos sena (balíků/ha)", 5.0, 30.0, 12.0, 1.0)
            
        with st.expander("Provozni Naklady a Ceny"):
            c_feed_own = st.number_input("Cena vl. krmiva (Kč/kg)", 0.5, 10.0, 2.5, 0.1)
            c_feed_market = st.number_input("Cena kup. krmiva (Kč/kg)", 2.0, 20.0, 8.0, 0.5)
            c_vet = st.number_input("Veterina (Kč/ks/rok)", 100.0, 2000.0, 350.0, 50.0)
            c_shearing = st.number_input("Stříhání (Kč/ks)", 20.0, 200.0, 50.0, 10.0)
            c_ram = st.number_input("Cena berana (Kč)", 5000.0, 30000.0, 10000.0, 1000.0)
            c_bale_sell_winter = st.number_input("Cena sena Zima (Kč/balík)", 200.0, 2000.0, 800.0, 50.0)
            c_bale_sell_summer = st.number_input("Cena sena Léto (Kč/balík)", 100.0, 1000.0, 400.0, 50.0)
            
        with st.expander("Stroje a Sluzby"):
            s_mow_ha = st.number_input("Služba: Seč (Kč/ha)", 500.0, 5000.0, 1500.0, 100.0)
            s_bale = st.number_input("Služba: Lisování (Kč/ks)", 50.0, 500.0, 200.0, 10.0)
            o_capex = st.number_input("Vlastní: Cena stroje (Kč)", 100000.0, 5000000.0, 600000.0, 50000.0)
            o_fuel = st.number_input("Vlastní: Nafta seč (Kč/ha)", 100.0, 1000.0, 400.0, 50.0)
            o_repair = st.number_input("Vlastní: Opravy ročně (Kč)", 0.0, 100000.0, 15000.0, 1000.0)

        with st.expander("Dotace a Dane"):
            sub_ha = st.number_input("SAPS (Kč/ha)", 0.0, 20000.0, 8500.0, 100.0)
            sub_sheep = st.number_input("VDJ (Kč/ks)", 0.0, 5000.0, 603.0, 10.0)
            tax_land = st.number_input("Daň z nemovitosti (Kč/ha)", 0.0, 2000.0, 500.0, 50.0)
            tax_build = st.number_input("Daň ze staveb (Kč/m²)", 0.0, 100.0, 15.0, 1.0)

        with st.expander("Rezie a Skalovani"):
            ov_base = st.number_input("Základní režie (Kč/rok)", 0.0, 200000.0, 40000.0, 1000.0)
            adm_base = st.number_input("Admin základ (Kč/rok)", 0.0, 50000.0, 5000.0, 500.0)
            adm_factor = st.number_input("Admin faktor (Diseconomy)", 1.0, 3.5, 2.0, 0.1, help="Exponent růstu administrativy. 1.0 = lineární, 1.5 = progresivní zátěž.")
            wage = st.number_input("Hodinová mzda (Kč/h)", 100.0, 1000.0, 200.0, 10.0)
            labor_h = st.number_input("Pracnost (h/ks/rok)", 1.0, 20.0, 6.0, 0.5)
            labor_ha = st.number_input("Pracnost půda (h/ha/rok)", 0.0, 50.0, 10.0, 1.0, help="Údržba ohradníků, pastvin, sečení nedopasků.")
            labor_fix = st.number_input("Fixní pracnost (h/rok)", 0.0, 1000.0, 200.0, 50.0, help="Údržba budov, administrativa, cesty.")
            labor_barn_m2 = st.number_input("Pracnost budovy (h/m²/rok)", 0.0, 10.0, 0.5, 0.1, help="Úklid, údržba, manipulace v ovčíně.")
            maint_barn_m2 = st.number_input("Údržba budovy (Kč/m²/rok)", 0.0, 1000.0, 60.0, 10.0, help="Opravy střechy, nátěry, dezinfekce.")
            shock_p = st.number_input("Pravděpodobnost šoku (denní %)", 0.0, 5.0, 0.5, 0.1) / 100.0

    # --- SAVE SCENARIO UI ---
    with save_sc_container:
        with st.expander("Ulozit aktualni nastaveni (pro Monte Carlo)"):
            st.info("Tento scénář bude uložen pod kategorii **C (Custom)**.")
            new_sc_name = st.text_input("Název scénáře", placeholder="Např. Můj optimalizovaný chov")
            if st.button("Uložit scénář"):
                if new_sc_name:
                    # Vytvoříme konfiguraci na základě BASE_SCENARIO a přepíšeme ji aktuálními vstupy
                    custom_sc = BASE_SCENARIO.copy()
                    custom_sc.update({
                        "sim_years": 5, "land_area": area, "meadow_share": meadow_pct/100.0, "barn_capacity": target_ewes,
                        "initial_ewes": start_ewes, "barn_area_m2": barn_m2, "hay_barn_area_m2": hay_barn_m2, "capital": cap,
                        "price_meat_avg": meat_price, "market_local_limit": m_quota, "price_meat_wholesale": m_wholesale,
                        "delay_bcs_perception": delay_bcs, "delay_feed_delivery": delay_mat, "initial_hay_bales": start_hay,
                        "enable_forecasting": use_forecast, "safety_margin": 0.2, "include_labor_cost": labor_on,
                        "climate_profile": climate, "machinery_mode": machinery, "rain_growth_global_mod": rain_mod,
                        "drought_prob_add": drought_add, "winter_len_global_mod": winter_mod,
                        "fertility_mean": p_fertility, "mortality_lamb_mean": p_mortality_lamb, "mortality_ewe_mean": p_mortality_ewe,
                        "feed_intake_ewe": p_feed_ewe, "hay_yield_ha_mean": p_hay_yield, "cost_feed_own_mean": c_feed_own,
                        "cost_feed_market_mean": c_feed_market, "cost_vet_base": c_vet, "cost_shearing": c_shearing,
                        "price_ram_purchase": c_ram, "price_bale_sell_winter": c_bale_sell_winter, "price_bale_sell_summer": c_bale_sell_summer,
                        "service_mow_ha": s_mow_ha, "service_bale_pcs": s_bale, "own_machine_capex": o_capex, "own_mow_fuel_ha": o_fuel,
                        "machinery_repair_mean": o_repair, "subsidy_ha_mean": sub_ha, "subsidy_sheep_mean": sub_sheep,
                        "tax_land_ha": tax_land, "tax_building_m2": tax_build, "overhead_base_year": ov_base,
                        "barn_maintenance_m2_year": maint_barn_m2, "admin_base_cost": adm_base, "admin_complexity_factor": adm_factor,
                        "wage_hourly": wage, "labor_hours_per_ewe_year": labor_h, "labor_hours_per_ha_year": labor_ha,
                        "labor_hours_fix_year": labor_fix, "labor_hours_barn_m2_year": labor_barn_m2, "shock_prob_daily": shock_p
                    })
                    
                    # Uložíme do session state s prefixem "C." (Custom)
                    st.session_state['custom_scenarios'][f"C. {new_sc_name}"] = custom_sc
                    st.success(f"Scénář '{new_sc_name}' byl uložen! Najdete ho v Monte Carlo Lab pod skupinou 'C'.")
                else:
                    st.warning("Zadejte prosím název scénáře.")
    
    st.markdown("---")
    with st.expander("Seed (Opakovatelnost)", expanded=False):
        sim_seed = st.number_input("Seed simulace", value=1337420, min_value=0, max_value=9999999999, help="Fixní seed zajistí, že náhoda (počasí, ceny) bude stejná pro porovnání scénářů.")

if mode_switch == "Monte Carlo Lab":
    st.title("Monte Carlo Laboratoř")
    st.markdown("Simulace tisíců běhů pro ověření robustnosti scénářů.")
    
    mc_cols = st.columns(3)
    n_runs = mc_cols[0].number_input("Počet běhů na scénář", 10, 2000, 50, help="Pro rychlý test dej 50. Pro finální data 1000.")
    
    with mc_cols[0]:
        sensitivity_on = st.checkbox("Citlivostni analyza", help="Náhodně mění vybrané parametry v každém běhu.")
        sens_map = {
            "Cena Masa": "price_meat_avg",
            "Cena Nafty": "own_mow_fuel_ha",
            "Počasí (Růst)": "rain_growth_global_mod",
            "Lokální Trh": "market_local_limit",
            "Cena Balíků": "price_bale_sell_winter",
            "Plodnost": "fertility_mean"
        }
        if sensitivity_on:
            sens_range_pct = st.slider("Rozptyl (+/- %)", 5, 50, 20) / 100.0
            sens_selection = st.multiselect("Parametry", list(sens_map.keys()), default=["Cena Masa", "Cena Nafty"])
        else:
            sens_range_pct = 0.0
            sens_selection = []

    labor_override = mc_cols[1].radio("Náklady na práci (Labor)", ["Dle scénáře", "Vše ZAPNUTO", "Vše VYPNUTO"], help="Přepíše nastavení ve scénářích.")
    
    # 1. Merge custom scenarios from session state
    if st.session_state['custom_scenarios']:
        SCENARIOS.update(st.session_state['custom_scenarios'])
    
    # 2. Get all available groups dynamically (including "C")
    all_groups = sorted(list(set([k[0] for k in SCENARIOS.keys()])))
    
    selected_groups = mc_cols[2].multiselect("Vyber skupiny scénářů", all_groups, default=["1", "5"])
    
    # Filter scenarios based on selection
    active_scenarios = {k: v for k, v in SCENARIOS.items() if k[0] in selected_groups}
    
    if st.button(f"🚀 Spustit simulaci ({len(active_scenarios) * n_runs} běhů)"):
        run_summaries = []
        quarterly_data = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        total_sims = len(active_scenarios) * n_runs
        counter = 0
        
        start_time = time.time()
        
        # Base config from sidebar inputs (as a baseline)
        base_kwargs = {
            "sim_years": 5, "land_area": area, "meadow_share": meadow_pct/100.0, "barn_capacity": target_ewes,
            "initial_ewes": start_ewes, "barn_area_m2": barn_m2, "hay_barn_area_m2": hay_barn_m2, "capital": cap,
            "price_meat_avg": meat_price, "market_local_limit": m_quota, "price_meat_wholesale": m_wholesale,
            "delay_bcs_perception": delay_bcs, "delay_feed_delivery": delay_mat, "initial_hay_bales": start_hay,
            "enable_forecasting": use_forecast, "safety_margin": 0.2, "include_labor_cost": labor_on,
            "climate_profile": climate, "machinery_mode": machinery, "rain_growth_global_mod": rain_mod,
            "drought_prob_add": drought_add, "winter_len_global_mod": winter_mod,
            "fertility_mean": p_fertility, "mortality_lamb_mean": p_mortality_lamb, "mortality_ewe_mean": p_mortality_ewe,
            "feed_intake_ewe": p_feed_ewe, "hay_yield_ha_mean": p_hay_yield, "cost_feed_own_mean": c_feed_own,
            "cost_feed_market_mean": c_feed_market, "cost_vet_base": c_vet, "cost_shearing": c_shearing,
            "price_ram_purchase": c_ram, "price_bale_sell_winter": c_bale_sell_winter, "price_bale_sell_summer": c_bale_sell_summer,
            "service_mow_ha": s_mow_ha, "service_bale_pcs": s_bale, "own_machine_capex": o_capex, "own_mow_fuel_ha": o_fuel,
            "machinery_repair_mean": o_repair, "subsidy_ha_mean": sub_ha, "subsidy_sheep_mean": sub_sheep,
            "tax_land_ha": tax_land, "tax_building_m2": tax_build, "overhead_base_year": ov_base,
            "barn_maintenance_m2_year": maint_barn_m2, "admin_base_cost": adm_base, "admin_complexity_factor": adm_factor,
            "wage_hourly": wage, "labor_hours_per_ewe_year": labor_h, "labor_hours_per_ha_year": labor_ha,
            "labor_hours_fix_year": labor_fix, "labor_hours_barn_m2_year": labor_barn_m2, "shock_prob_daily": shock_p
        }

        for sc_name, sc_params in active_scenarios.items():
            # Merge base config with scenario overrides
            run_kwargs = base_kwargs.copy()
            run_kwargs.update(sc_params)
            
            # Apply Labor Override
            if labor_override == "Vše ZAPNUTO":
                run_kwargs["include_labor_cost"] = True
            elif labor_override == "Vše VYPNUTO":
                run_kwargs["include_labor_cost"] = False
            
            for i in range(n_runs):
                # Random seed for each run
                # FIX: Consistent seeds across scenarios (Seed 0 is always Seed 0)
                current_seed = sim_seed + i
                np.random.seed(current_seed) 
                
                # Sensitivity Perturbation (Per Run)
                current_run_kwargs = run_kwargs.copy()
                sens_log = {}
                
                if sensitivity_on and sens_selection:
                    for label in sens_selection:
                        key = sens_map[label]
                        factor = np.random.uniform(1.0 - sens_range_pct, 1.0 + sens_range_pct)
                        
                        if key == "price_bale_sell_winter":
                            current_run_kwargs["price_bale_sell_winter"] *= factor
                            current_run_kwargs["price_bale_sell_summer"] *= factor
                            sens_log[label] = current_run_kwargs["price_bale_sell_winter"]
                        elif key == "market_local_limit":
                            current_run_kwargs[key] = int(current_run_kwargs[key] * factor)
                            sens_log[label] = current_run_kwargs[key]
                        else:
                            current_run_kwargs[key] *= factor
                            sens_log[label] = current_run_kwargs[key]
                
                # RE-SEED: Zajistíme, že stochastika modelu (počasí, ceny) bude identická
                # pro daný Seed, bez ohledu na to, zda jsme "spotřebovali" náhodu pro citlivostní analýzu.
                np.random.seed(current_seed)
                
                # Create config object
                mc_cfg = FarmConfig(**current_run_kwargs)
                
                mc_model = FarmModel(mc_cfg)
                mc_df = mc_model.run()
                
                # --- 1. RUN SUMMARY (Agregace za celý běh) ---
                profit = mc_df["Cash"].iloc[-1] - mc_cfg.capital
                is_bankrupt = 1 if mc_df["Cash"].iloc[-1] < 0 else 0
                
                total_labor = mc_df["Labor Hours"].sum()
                efficiency = profit / max(1.0, total_labor)
                
                summary_row = {
                    "Scénář": sc_name,
                    "Skupina": sc_name.split(":")[0],
                    "Seed": current_seed,
                    "Počet Ovcí (Start)": mc_cfg.initial_ewes,
                    "Plocha (ha)": mc_cfg.land_area,
                    "Zisk (Kč)": profit,
                    "Efektivita (Kč/h)": efficiency,
                    "Konečný Cash": mc_df["Cash"].iloc[-1],
                    "Bankrot": is_bankrupt,
                    "Min BCS": mc_df["BCS"].min(),
                    "Max BCS": mc_df["BCS"].max(),
                    "Průměr BCS": mc_df["BCS"].mean(),
                    "Konečné Ovce": mc_df["Total Animals"].iloc[-1],
                    "Pasture Health (End)": mc_df["Pasture_Health"].iloc[-1],
                    "Pracnost (h)": mc_df["Labor Hours"].sum(),
                    "Dny Sucha": mc_df["Is_Drought"].sum(),
                    "Dny Zimy": mc_df["Is_Winter"].sum(),
                    "Seno (Konec)": mc_df["Hay Stock"].iloc[-1]
                }
                # Add sensitivity inputs
                summary_row.update(sens_log)
                run_summaries.append(summary_row)
                
                # --- 2. QUARTERLY DATA (Pro časovou analýzu) ---
                # Resample na kvartály (používáme 'M' a filtrujeme, pro kompatibilitu)
                # Vezmeme poslední den v měsíci
                monthly = mc_df.resample('M').last()
                # Filtrujeme jen březen, červen, září, prosinec
                quarterly = monthly[monthly.index.month.isin([3, 6, 9, 12])].copy()
                
                for date, row in quarterly.iterrows():
                    q_label = f"{date.year} Q{(date.month-1)//3 + 1}"
                    quarterly_data.append({
                        "Scénář": sc_name,
                        "Seed": current_seed,
                        "Datum": date,
                        "Kvartál": q_label,
                        "Cash": row["Cash"],
                        "Animals": row["Total Animals"],
                        "BCS": row["BCS"],
                        "Hay Stock": row["Hay Stock"],
                        "Pasture Health": row["Pasture_Health"]
                    })
                
                counter += 1
                if counter % 10 == 0:
                    progress_bar.progress(counter / total_sims)
                    status_text.text(f"Simuluji: {sc_name} (Běh {i+1}/{n_runs})")
        
        progress_bar.empty()
        status_text.success(f"Hotovo! Simulováno {total_sims} běhů za {time.time()-start_time:.1f}s.")
        
        # Uložení výsledků do session state pro persistenci při interakci s grafy
        st.session_state['mc_results'] = {
            'summary': pd.DataFrame(run_summaries),
            'quarterly': pd.DataFrame(quarterly_data)
        }
        
    # Pokud máme výsledky v paměti, zobrazíme je (i po restartu stránky)
    if 'mc_results' in st.session_state:
        # --- VISUALIZATION ---
        df_summary = st.session_state['mc_results']['summary']
        df_quarterly = st.session_state['mc_results']['quarterly']
        
        # 1. SCENARIO DEFINITIONS TABLE
        st.subheader("Definice Scenaru")
        st.dataframe(pd.DataFrame(SCENARIOS).T)

        # 2. TIME SLICER & BOXPLOTS
        st.subheader("Porovnani v case (Slicer)")
        
        # Get unique quarters sorted
        available_quarters = sorted(df_quarterly["Kvartál"].unique())
        selected_q = st.select_slider("Vyberte období pro srovnání:", options=available_quarters, value=available_quarters[-1])
        
        # Filter data for chart
        df_slice = df_quarterly[df_quarterly["Kvartál"] == selected_q]
        
        chart_profit = alt.Chart(df_slice).mark_boxplot().encode(
            x=alt.X("Scénář:N", title=None),
            y=alt.Y("Cash:Q", title=f"Hotovost v {selected_q} (Kč)"),
            color="Scénář:N",
            tooltip=["Scénář", "Cash", "BCS", "Animals"]
        ).properties(height=400, title=f"Rozdělení hotovosti ({selected_q})")
        st.altair_chart(chart_profit, use_container_width=True)
        
        # 2b. EFFICIENCY CHART
        st.subheader("Pracovni Efektivita (Zisk na hodinu)")
        chart_eff = alt.Chart(df_summary).mark_boxplot().encode(
            x=alt.X("Scénář:N", title=None),
            y=alt.Y("Efektivita (Kč/h):Q", title="Zisk na hodinu práce (Kč/h)"),
            color="Skupina:N",
            tooltip=["Scénář", "Efektivita (Kč/h)", "Zisk (Kč)", "Pracnost (h)"]
        ).properties(height=300)
        st.altair_chart(chart_eff, use_container_width=True)
        
        # 3. RISK CHART (X = Sheep Count)
        st.subheader("Risk vs Reward (Riziko vs Zisk)")
        risk_agg = df_summary.groupby("Scénář").agg(
            Riziko_Bankrotu=("Bankrot", "mean"),
            Průměr_Min_BCS=("Min BCS", "mean"),
            Průměr_Zisk=("Zisk (Kč)", "mean"),
            Počet_Ovcí_Start=("Počet Ovcí (Start)", "first"), # Constant per scenario
            Plocha=("Plocha (ha)", "first")
        ).reset_index()
        
        risk_chart = alt.Chart(risk_agg).mark_circle(opacity=0.8).encode(
            x=alt.X("Průměr_Zisk:Q", title="Průměrný Zisk (Kč)"),
            y=alt.Y("Riziko_Bankrotu:Q", title="Pravděpodobnost Bankrotu (0-1)", axis=alt.Axis(format='%')),
            size=alt.Size("Počet_Ovcí_Start:Q", title="Velikost Stáda", scale=alt.Scale(range=[200, 1000])),
            color=alt.Color("Průměr_Min_BCS:Q", scale=alt.Scale(scheme="redyellowgreen", domain=[1.5, 3.5]), title="Avg Min BCS"),
            tooltip=["Scénář", "Riziko_Bankrotu", "Průměr_Zisk", "Průměr_Min_BCS", "Počet_Ovcí_Start"]
        ).properties(height=400)
        
        st.altair_chart(risk_chart, use_container_width=True)
        st.caption("Osa X: Průměrný Zisk. Osa Y: Riziko bankrotu. Barva: Zdraví zvířat (Červená = Hlad). Velikost bubliny: Počet ovcí.")
        
        # 3b. TIME SERIES EVOLUTION
        st.subheader("Vyvoj v case")
        
        ts_view_mode = st.radio("Rezim zobrazeni", ["Vsechny behy (Detail)", "Pasmo spolehlivosti (Agregace)"], horizontal=True)
        
        if ts_view_mode == "Vsechny behy (Detail)":
            # Calculate opacity based on number of runs to avoid overplotting
            opacity_val = max(0.05, min(0.8, 20.0 / n_runs))
            selection = alt.selection_point(fields=['Scénář'], bind='legend')
            
            chart_cf = alt.Chart(df_quarterly).mark_line().encode(
                x=alt.X("Datum:T", title="Čas"),
                y=alt.Y("Cash:Q", title="Hotovost (Kč)"),
                color="Scénář:N",
                detail="Seed:N",
                opacity=alt.condition(selection, alt.value(opacity_val), alt.value(0.005)),
                tooltip=["Scénář", "Seed", "Datum", "Cash"]
            ).add_params(selection).properties(title="Vývoj Cashflow (Všechny simulace)", height=300)
            
            chart_bcs = alt.Chart(df_quarterly).mark_line().encode(
                x=alt.X("Datum:T", title="Čas"),
                y=alt.Y("BCS:Q", title="BCS", scale=alt.Scale(domain=[1.5, 4.0])),
                color="Scénář:N",
                detail="Seed:N",
                opacity=alt.condition(selection, alt.value(opacity_val), alt.value(0.005)),
                tooltip=["Scénář", "Seed", "Datum", "BCS"]
            ).add_params(selection).properties(title="Vývoj Kondice (BCS)", height=300)
            
            chart_pas = alt.Chart(df_quarterly).mark_line().encode(
                x=alt.X("Datum:T", title="Čas"),
                y=alt.Y("Pasture Health:Q", title="Zdraví Pastviny (0-1)"),
                color="Scénář:N",
                detail="Seed:N",
                opacity=alt.condition(selection, alt.value(opacity_val), alt.value(0.005)),
                tooltip=["Scénář", "Seed", "Datum", "Pasture Health"]
            ).add_params(selection).properties(title="Degradace Pastviny", height=300)
            
        else:
            # Confidence Interval Aggregation
            ci_agg = df_quarterly.groupby(["Scénář", "Datum"]).agg(
                Mean_Cash=("Cash", "mean"),
                Min_Cash=("Cash", lambda x: x.quantile(0.05)),
                Max_Cash=("Cash", lambda x: x.quantile(0.95)),
                Mean_BCS=("BCS", "mean"),
                Min_BCS=("BCS", lambda x: x.quantile(0.05)),
                Max_BCS=("BCS", lambda x: x.quantile(0.95)),
                Mean_Pas=("Pasture Health", "mean"),
                Min_Pas=("Pasture Health", lambda x: x.quantile(0.05)),
                Max_Pas=("Pasture Health", lambda x: x.quantile(0.95))
            ).reset_index()
            
            def create_ci_chart(y_mean, y_min, y_max, title, y_title):
                base = alt.Chart(ci_agg).encode(x=alt.X("Datum:T", title="Čas"), color="Scénář:N")
                band = base.mark_area(opacity=0.3).encode(y=alt.Y(f"{y_min}:Q", title=y_title), y2=f"{y_max}:Q")
                line = base.mark_line(size=3).encode(y=f"{y_mean}:Q")
                return (band + line).properties(title=title, height=300)

            chart_cf = create_ci_chart("Mean_Cash", "Min_Cash", "Max_Cash", "Vývoj Cashflow (Průměr + 90% Interval)", "Hotovost (Kč)")
            chart_bcs = create_ci_chart("Mean_BCS", "Min_BCS", "Max_BCS", "Vývoj Kondice (BCS)", "BCS")
            chart_pas = create_ci_chart("Mean_Pas", "Min_Pas", "Max_Pas", "Degradace Pastviny", "Zdraví Pastviny (0-1)")
        
        st.altair_chart(chart_cf, use_container_width=True)
        
        col_ts1, col_ts2 = st.columns(2)
        with col_ts1:
            st.altair_chart(chart_bcs, use_container_width=True)
            
        with col_ts2:
            st.altair_chart(chart_pas, use_container_width=True)
        
        # 4. SENSITIVITY ANALYSIS (Scatter)
        if sensitivity_on and sens_selection:
            st.subheader("Citlivostni Analyza (Korelace)")
            
            # Create dynamic columns based on selection
            cols = st.columns(min(len(sens_selection), 3))
            
            for i, label in enumerate(sens_selection):
                with cols[i % 3]:
                    chart_sens = alt.Chart(df_summary).mark_circle(size=60, opacity=0.5).encode(
                        x=alt.X(f"{label}:Q", title=label, scale=alt.Scale(zero=False)),
                        y=alt.Y("Zisk (Kč):Q", title="Zisk"),
                        color="Skupina:N",
                        tooltip=["Scénář", label, "Zisk (Kč)"]
                    ).properties(title=f"Zisk vs. {label}")
                    st.altair_chart(chart_sens, use_container_width=True)
        
        # 4. DATA TABLES
        st.subheader("Souhrnne Vysledky (Prumery)")
        st.dataframe(risk_agg.style.format({
            "Riziko_Bankrotu": "{:.1%}", 
            "Průměr_Zisk": "{:,.0f}", 
            "Průměr_Min_BCS": "{:.2f}"
        }), use_container_width=True)
        
        with st.expander("Surova Data (Kvartalni export)"):
            st.markdown("Data obsahují záznam pro každý Seed a každý Kvartál.")
            st.dataframe(df_quarterly)
            st.download_button("Stáhnout CSV (Quarterly)", df_quarterly.to_csv(index=False).encode('utf-8'), "monte_carlo_quarterly.csv")
            
        with st.expander("Surova Data (Souhrn behu)"):
            st.markdown("Data obsahují jeden řádek pro každý Seed (finální výsledky).")
            st.dataframe(df_summary)
            st.download_button("Stáhnout CSV (Summary)", df_summary.to_csv(index=False).encode('utf-8'), "monte_carlo_summary.csv")
            
    st.stop() # Stop execution here so standard dashboard doesn't render below

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
    market_local_limit=m_quota,
    price_meat_wholesale=m_wholesale,
    delay_bcs_perception=delay_bcs,
    delay_feed_delivery=delay_mat,
    initial_hay_bales=start_hay,
    enable_forecasting=use_forecast, 
    safety_margin=0.2,
    include_labor_cost=labor_on,
    climate_profile=climate,
    machinery_mode=machinery,
    rain_growth_global_mod=rain_mod,
    drought_prob_add=drought_add,
    winter_len_global_mod=winter_mod,
    
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

np.random.seed(sim_seed)
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
final_cash = df["Cash"].iloc[-1]
final_animals = df["Total Animals"].iloc[-1]
final_hay = df["Hay Stock"].iloc[-1]
total_profit = final_cash - cap
avg_md_year = (df["Labor Hours"].sum() / cfg.sim_years) / 8.0

# --- STICKY KPI ROW (FIXED POSITION) ---
st.markdown(f"""
    <div style="position: fixed; top: 3.5rem; left: 21rem; right: 0; z-index: 999; background-color: #0e1117; padding: 0.5rem 2rem; border-bottom: 1px solid #262730; display: flex; justify-content: space-around; align-items: center; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); border-bottom-left-radius: 8px;">
        <div style="text-align: center;">
            <div style="font-size: 0.8rem; color: #fafafa; opacity: 0.8;">Hotovost</div>
            <div style="font-size: 1.1rem; font-weight: bold; color: #2ecc71;">{final_cash:,.0f} Kč</div>
            <div style="font-size: 0.7rem; color: {'#2ecc71' if total_profit >= 0 else '#e74c3c'};">{total_profit:+,.0f}</div>
        </div>
        <div style="text-align: center;">
            <div style="font-size: 0.8rem; color: #fafafa; opacity: 0.8;">Stav stáda</div>
            <div style="font-size: 1.1rem; font-weight: bold; color: #fafafa;">{int(final_animals)}</div>
            <div style="font-size: 0.7rem; color: #fafafa;">{int(final_animals - start_ewes):+d}</div>
        </div>
        <div style="text-align: center;">
            <div style="font-size: 0.8rem; color: #fafafa; opacity: 0.8;">Seno</div>
            <div style="font-size: 1.1rem; font-weight: bold; color: #f39c12;">{final_hay:.0f}</div>
        </div>
        <div style="text-align: center;">
            <div style="font-size: 0.8rem; color: #fafafa; opacity: 0.8;">ROI</div>
            <div style="font-size: 1.1rem; font-weight: bold; color: #fafafa;">{(total_profit/cap*100):.1f}%</div>
        </div>
        <div style="text-align: center;">
            <div style="font-size: 0.8rem; color: #fafafa; opacity: 0.8;">Pracnost</div>
            <div style="font-size: 1.1rem; font-weight: bold; color: #fafafa;">{avg_md_year:.1f} MD</div>
        </div>
    </div>
    <div style="height: 4rem;"></div> <!-- Spacer to prevent content overlap -->
""", unsafe_allow_html=True)

# --- 2. HERD STRUCTURE ---
st.subheader("Struktura stáda (detailně)")

df_herd_melt = df.reset_index().melt(id_vars='Date', value_vars=['Ewes', 'Lambs Male', 'Lambs Female'], var_name='Kategorie', value_name='Počet')

herd_chart = alt.Chart(df_herd_melt).mark_area(opacity=0.7).encode(
    x=alt.X('Date:T', title='Datum'),
    y=alt.Y('Počet:Q', title='Počet zvířat', stack='zero'),
    color=alt.Color('Kategorie:N', title='Kategorie'),
    tooltip=['Date:T', 'Kategorie:N', 'Počet:Q']
).properties(
    height=350
)
st.altair_chart(herd_chart, use_container_width=True)

# --- 3. HAY MANAGEMENT ---
st.subheader("Seno - výroba a spotřeba")

hay_chart = alt.Chart(df.reset_index()).mark_area(
    line={'color':'#f39c12'},
    color=alt.Gradient(
        gradient='linear',
        stops=[alt.GradientStop(color='black', offset=0), alt.GradientStop(color='#f39c12', offset=1)],
        x1=1, x2=1, y1=1, y2=0
    )
).encode(
    x=alt.X('Date:T', title='Datum'),
    y=alt.Y('Hay Stock:Q', title='Balíky sena'),
    tooltip=['Date:T', alt.Tooltip('Hay Stock:Q', format='.0f')]
)
st.altair_chart(hay_chart, use_container_width=True)
st.caption("Červen = senoseč, září = otava. Zimní úbytek = spotřeba. Červená čára = kritický stav zásob.")

# --- 4. FINANCIAL OVERVIEW (STACKED CASHFLOW) ---
st.subheader("Cashflow a ziskovost")
col_chart, col_pie = st.columns([3, 1])

with col_chart:
    # Monthly aggregation for nicer chart
    df_monthly = df.resample("M").sum()
    df_monthly['Net Flow'] = df_monthly['Income'] - (df_monthly['Exp_Feed'] + df_monthly['Exp_Variable'] + df_monthly['Exp_Admin'] + df_monthly['Exp_Overhead'] + df_monthly['Exp_Labor'] + df_monthly['Exp_Shock'])
    df_monthly['Cumulative Cash'] = df_monthly['Net Flow'].cumsum() + cfg.capital
    
    # Výpočet průměrných cen pro tooltip (ošetření dělení nulou)
    df_monthly['Avg_Meat_Price'] = df_monthly.apply(lambda x: x['Inc_Meat'] / x['Sold_Animals'] if x['Sold_Animals'] > 0 else 0, axis=1)
    df_monthly['Avg_Hay_Price'] = df_monthly.apply(lambda x: x['Inc_Hay'] / x['Sold_Hay'] if x['Sold_Hay'] > 0 else 0, axis=1)

    cash_flow_chart = alt.Chart(df_monthly.reset_index()).mark_bar().encode(
        x=alt.X('Date:T', title='Měsíc'),
        y=alt.Y('Net Flow:Q', title='Měsíční Cashflow (Kč)'),
        color=alt.condition(
            alt.datum['Net Flow'] > 0,
            alt.value('#2ecc71'),  # Zelená pro zisk
            alt.value('#e74c3c')   # Červená pro ztrátu
        ),
        tooltip=[
            alt.Tooltip('Date:T', title='Měsíc', format='%B %Y'),
            alt.Tooltip('Net Flow:Q', title='Čistý tok', format=',.0f'),
            alt.Tooltip('Inc_Meat:Q', title='Příjem Maso', format=',.0f'),
            alt.Tooltip('Sold_Animals:Q', title='Prodané kusy', format=',.0f'),
            alt.Tooltip('Avg_Meat_Price:Q', title='Ø Cena Maso', format=',.0f'),
            alt.Tooltip('Inc_Hay:Q', title='Příjem Seno', format=',.0f'),
            alt.Tooltip('Sold_Hay:Q', title='Prodané balíky', format=',.0f'),
            alt.Tooltip('Inc_Subsidy:Q', title='Dotace', format=',.0f'),
            alt.Tooltip('Exp_Feed:Q', title='Náklady Krmivo', format=',.0f'),
            alt.Tooltip('Exp_Variable:Q', title='Náklady Var.', format=',.0f'),
            alt.Tooltip('Exp_Admin:Q', title='Náklady Admin', format=',.0f'),
            alt.Tooltip('Exp_Overhead:Q', title='Náklady Režie', format=',.0f'),
            alt.Tooltip('Exp_Labor:Q', title='Náklady Práce', format=',.0f'),
            alt.Tooltip('Exp_Shock:Q', title='Náklady Šoky', format=',.0f'),
        ]
    ).properties(
        title='Měsíční čistý peněžní tok'
    )

    cumulative_line = alt.Chart(df_monthly.reset_index()).mark_line(color='#3498db', size=3).encode(
        x=alt.X('Date:T'),
        y=alt.Y('Cumulative Cash:Q', title='Kumulativní hotovost (Kč)', axis=alt.Axis(orient='right')),
        tooltip=[alt.Tooltip('Date:T', title='Měsíc'), alt.Tooltip('Cumulative Cash:Q', title='Hotovost', format=',.0f')]
    )

    final_cashflow_chart = alt.layer(cash_flow_chart, cumulative_line).resolve_scale(
        y='independent'
    )

    st.altair_chart(final_cashflow_chart, use_container_width=True)

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
    
    df_price = df.copy()
    df_price["Month"] = df_price.index.month
    
    # Agregace pro boxplot tooltips
    price_stats = df_price.groupby("Month")["Meat_Price"].describe().reset_index()
    
    base_price = alt.Chart(price_stats).encode(x=alt.X('Month:O', title='Měsíc'))
    
    rule = base_price.mark_rule().encode(
        y=alt.Y('min:Q', title='Cena masa (Kč/kg)', scale=alt.Scale(zero=False)),
        y2='max:Q'
    )
    
    bar = base_price.mark_bar(size=15).encode(
        y='25%:Q',
        y2='75%:Q',
        tooltip=[
            alt.Tooltip('Month:O', title='Měsíc'),
            alt.Tooltip('mean:Q', title='Průměr', format='.1f'),
            alt.Tooltip('50%:Q', title='Medián', format='.1f'),
            alt.Tooltip('min:Q', title='Min', format='.1f'),
            alt.Tooltip('max:Q', title='Max', format='.1f')
        ]
    )
    
    tick = base_price.mark_tick(color='white', size=15).encode(y='50%:Q')
    
    price_boxplot = (rule + bar + tick).properties(title="Měsíční distribuce cen masa (vč. Velikonoc)")
    
    st.altair_chart(price_boxplot, use_container_width=True)

# --- 6. FEEDING EFFICIENCY ---
st.subheader("Účinnost krmení")
col_feed_chart, col_feed_info = st.columns([2, 1])

with col_feed_chart:
    # Agregace klíčů pro zjednodušení grafu
    days_grazing = model.feed_log.get("Grazing", 0) + model.feed_log.get("Grazing (No Supplement)", 0) + model.feed_log.get("Grazing+Starvation", 0)
    days_stored = model.feed_log.get("Stored", 0) + model.feed_log.get("Grazing+Stored", 0)
    days_market = model.feed_log.get("Market", 0) + model.feed_log.get("Starvation (Wait for Delivery)", 0) + model.feed_log.get("Starvation (No Hay)", 0)
    days_stored += model.feed_log.get("Stored (Pasture Rest)", 0)
    
    total_days = sum(model.feed_log.values())
    
    grazing_pct = (days_grazing / total_days * 100) if total_days > 0 else 0
    stored_pct = (days_stored / total_days * 100) if total_days > 0 else 0
    market_pct = (days_market / total_days * 100) if total_days > 0 else 0
    
    fig_feed, ax_feed = plt.subplots(figsize=(10, 5))
    
    feed_data = [days_grazing, days_stored, days_market]
    feed_labels = [f"Pastva\n({grazing_pct:.0f}%)", f"Seno\n({stored_pct:.0f}%)", f"Nákup\n({market_pct:.0f}%)"]
    feed_colors = ["#2ecc71", "#f39c12", "#e74c3c"]
    
    bars = ax_feed.barh(feed_labels, feed_data, color=feed_colors, edgecolor="white", linewidth=2)
    
    for bar, value in zip(bars, feed_data):
        ax_feed.text(value/2, bar.get_y() + bar.get_height()/2, f"{int(value)}d", 
                    ha="center", va="center", fontsize=12, fontweight="bold", color="white")
    
    ax_feed.set_xlabel("Dny v Roce", fontsize=11, fontweight="bold")
    ax_feed.grid(axis='x', alpha=0.2)
    
    st.pyplot(fig_feed)
    
    # --- FEEDING TIMELINE (New!) ---
    st.markdown("**Historie krmení**")
    feed_timeline = alt.Chart(df.reset_index()).mark_bar().encode(
        x=alt.X('Date:T', title='Datum'),
        y=alt.Y('Feed_Source:N', title='Zdroj krmiva'),
        color=alt.Color('Feed_Source:N', legend=None),
        tooltip=['Date:T', 'Feed_Source:N', alt.Tooltip('Exp_Feed:Q', title='Náklady (Kč)', format='.0f')]
    ).properties(
        height=150
    )
    st.altair_chart(feed_timeline, use_container_width=True)

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

bcs_melt = df.reset_index().melt(id_vars='Date', value_vars=['BCS', 'Perceived_BCS'], var_name='Typ', value_name='Hodnota')

bcs_chart = alt.Chart(bcs_melt).mark_line().encode(
    x=alt.X('Date:T', title='Datum'),
    y=alt.Y('Hodnota:Q', title='BCS (1-5)', scale=alt.Scale(domain=[1.5, 4.5])),
    color='Typ:N',
    strokeDash=alt.condition(alt.datum.Typ == 'Perceived_BCS', alt.value([5,5]), alt.value([0])),
    tooltip=['Date:T', 'Typ:N', alt.Tooltip('Hodnota:Q', format='.2f')]
)

st.altair_chart(bcs_chart, use_container_width=True)
st.caption("BCS ovlivňuje plodnost a mortalitu. Cíl je držet se v zelené zóně (2.5 - 3.5). Pod 2.0 hrozí úhyn a neplodnost.")

# --- 6.c PASTURE HEALTH ---
st.subheader("🌱 Zdraví Pastviny (Ecological Loop)")

pasture_chart = alt.Chart(df.reset_index()).mark_area(
    line={'color':'#27ae60'},
    color=alt.Gradient(
        gradient='linear',
        stops=[alt.GradientStop(color='#0e1117', offset=0), alt.GradientStop(color='#27ae60', offset=1)],
        x1=1, x2=1, y1=1, y2=0
    )
).encode(
    x=alt.X('Date:T', title='Datum'),
    y=alt.Y('Pasture_Health:Q', title='Zdraví pastviny (%)', axis=alt.Axis(format='%')),
    tooltip=['Date:T', alt.Tooltip('Pasture_Health:Q', format='.1%')]
)
st.altair_chart(pasture_chart, use_container_width=True)
st.caption("Pokud zdraví klesá, máte příliš mnoho ovcí na málo hektarů (Overgrazing). Tráva přestane růst.")

# --- 6.d ADMIN DISECONOMY ---
st.subheader("📉 Administrativní Zátěž (Diseconomy of Scale)")

col_sim, col_theory = st.columns(2)

with col_sim:
    st.markdown("**Vývoj v simulaci**")
    base = alt.Chart(df.reset_index()).encode(x='Date:T')
    admin_line = base.mark_line(color='#e74c3c').encode(
        y=alt.Y('Exp_Admin:Q', title='Admin Náklady (Kč/den)', axis=alt.Axis(titleColor='#e74c3c')),
        tooltip=['Date:T', alt.Tooltip('Exp_Admin:Q', format=',.0f')]
    )
    animal_line = base.mark_line(color='#3498db', strokeDash=[5,5]).encode(
        y=alt.Y('Total Animals:Q', title='Počet zvířat', axis=alt.Axis(titleColor='#3498db')),
        tooltip=['Date:T', alt.Tooltip('Total Animals:Q', format='.0f')]
    )
    admin_chart = alt.layer(admin_line, animal_line).resolve_scale(y='independent')
    st.altair_chart(admin_chart, use_container_width=True)

with col_theory:
    st.markdown("**Teoretické křivky**")
    animals = np.arange(0, 501, 10)
    factors = [1.0, 1.5, 2.0, 2.5, 3.0]
    data = []
    for factor in factors:
        for n in animals:
            cost = (cfg.admin_base_cost * (max(1, n) / 50.0) ** factor)
            data.append({'Počet zvířat': n, 'Faktor': str(factor), 'Roční náklady': cost})
    df_admin_comp = pd.DataFrame(data)

    theory_chart = alt.Chart(df_admin_comp).mark_line().encode(
        x=alt.X('Počet zvířat:Q'),
        y=alt.Y('Roční náklady:Q', title='Roční náklady (Kč)'),
        color='Faktor:N',
        tooltip=['Počet zvířat', 'Faktor', alt.Tooltip('Roční náklady:Q', format=',.0f')]
    )
    st.altair_chart(theory_chart, use_container_width=True)

st.caption("Sledujte, jak náklady na administrativu (červená) rostou rychleji než počet zvířat (modrá). To je 'Diseconomy of Scale'.")

# --- 6.e WEATHER ANALYSIS ---
st.subheader("🌤️ Analýza Počasí a Klimatu")

weather_base = alt.Chart(df.reset_index()).encode(x=alt.X('Date:T', title='Datum'))

# 1. Regime (Line)
regime_line = weather_base.mark_line(color='#f1c40f').encode(
    y=alt.Y('Weather_Regime:Q', title='Vlhkostní Režim', scale=alt.Scale(domain=[0.4, 1.6])),
    tooltip=['Date:T', alt.Tooltip('Weather_Regime:Q', format='.2f')]
)

# 2. Background bands (Winter, Drought)
# Používáme transform_filter, aby se vykreslily jen relevantní dny (oprava tooltipu)
winter_band = weather_base.transform_filter(
    alt.datum.Is_Winter == 1
).mark_bar(opacity=0.3, color='#3498db').encode(
    y=alt.value(0), y2=alt.value(200)
)

drought_band = weather_base.transform_filter(
    alt.datum.Is_Drought == 1
).mark_bar(opacity=0.6, color='#e74c3c').encode(
    y=alt.value(0), y2=alt.value(200),
    tooltip=[alt.Tooltip('Date:T'), alt.Tooltip('Is_Drought:Q', title='Sucho')]
)

weather_chart = (winter_band + drought_band + regime_line).properties(height=200, title="Klima a Počasí")
st.altair_chart(weather_chart, use_container_width=True)
st.caption("Barevné pásy ukazují převládající počasí. Tmavě modrá = zima, oranžová = suchý trend, červená = extrémní sucho.")

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
    "3a. Režie, Admin, Práce (Kč/ks)": 1000,
    "3b. Stroje, Seč, Opravy (Kč/ks)": 500,
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
model_overhead_admin = (df["Exp_Overhead"].sum() + df["Exp_Admin"].sum() + df["Exp_Labor"].sum()) / (avg_ewes * years)
model_machinery_ops = (df["Exp_Mow"].sum() + df["Exp_Machinery"].sum() + df["Exp_Shock"].sum()) / (avg_ewes * years)

# Meat Income
model_meat = (oct_income / (avg_ewes * years)) if oct_income > 0 else (df["Income"].sum() / (avg_ewes * years))
model_profit_no_sub = model_meat - (model_feed + model_vet_services + model_overhead_admin + model_machinery_ops)

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
    "Tvůj Model": [model_feed, model_vet_services, model_overhead_admin, model_machinery_ops, model_meat, model_profit_no_sub, model_rearing, model_subsidy_dep]
})

# Calculate difference
validation_df["Odchylka"] = validation_df["Tvůj Model"] - validation_df["Průměr ČR (Realita)"]

# Display table
st.markdown("###  Detailní Srovnání")
st.dataframe(validation_df.style.format("{:,.0f}", subset=["Průměr ČR (Realita)", "Tvůj Model", "Odchylka"]), use_container_width=True, height=300)

# --- 9. AGE STRUCTURE (Validation) ---
st.markdown("---")
st.subheader("🎂 Věková struktura stáda (na konci simulace)")

snapshot_dates = sorted(list(model.yearly_age_snapshots.keys()))
if snapshot_dates:
    selected_date = st.select_slider(
        "Vyberte datum pro zobrazení struktury",
        options=snapshot_dates,
        format_func=lambda x: x.strftime("%b %Y"),
        value=snapshot_dates[-1]
    )
    
    snapshot_data = model.yearly_age_snapshots[selected_date]
    df_age_snap = pd.DataFrame(snapshot_data)
    
    age_chart = alt.Chart(df_age_snap).mark_bar().encode(
        x=alt.X("Age:Q", bin=alt.Bin(step=1), title="Věk (roky)"),
        y=alt.Y("count()", title="Počet ovcí"),
        color=alt.Color("Category:N", title="Kategorie"),
        tooltip=[alt.Tooltip("Category:N", title="Kategorie"), alt.Tooltip("count()", title="Počet")]
    )
    
    limit_line = alt.Chart(pd.DataFrame({'x': [cfg.max_ewe_age]})).mark_rule(color='red', strokeDash=[5, 5]).encode(
        x='x:Q'
    )

    final_age_chart = (age_chart + limit_line).properties(
        title=f"Struktura stáda: {selected_date.strftime('%B %Y')}"
    )
    
    st.altair_chart(final_age_chart, use_container_width=True)

st.caption("Histogram ukazuje rozložení věku bahnic. Měli byste vidět 'schody' (kohorty) a propad po dosažení věku vyřazení.")
