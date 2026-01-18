import numpy as np
import pandas as pd
from dataclasses import dataclass

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
        elif cfg.climate_profile == "UI_Custom":
            base_grass = 1.0
            base_drought = 0.0
            base_winter = 1.0
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
        self.feed_log = {"Pastva": 0, "Seno": 0, "Nákup": 0}
        
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
                feed_source = "Hladovění (Čekání)"
            else:
                self.bcs = max(2.5, self.bcs - 0.001) # Maintenance
                feed_source = "Seno"
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
                feed_source = "Pastva"
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
                    feed_source = "Pastva + Hlad" if not force_hay else "Hladovění (Bez sena)"
                else:
                    feed_source = "Pastva + Seno" if not force_hay else "Seno (Ochrana)"
            else:
                # Farmář si myslí, že jsou OK, tak nepřikrmuje, i když je málo trávy
                # "Necháme je vyžrat nedopasky"
                self.bcs -= 0.002 # Skutečná kondice klesá
                feed_source = "Pastva (Bez příkrmu)"
        
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
    "1. Rodinný Ideál (Základ)": {
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
    "2. Hobby Zahrada (Malá škála)": {
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
    "3. Agro Moloch (Neefektivita)": {
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
    "4. Klimatická Krize (Zátěžový test)": {
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
    "5. Vrakoviště (Vysoké riziko)": {
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
