import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass

# --- CONFIGURATION AND STYLE ---
st.set_page_config(page_title="Ovčí farma", layout="wide")
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
    meadow_share: float        # % of land reserved only for hay production
    barn_capacity: int
    barn_area_m2: float        # Storage capacity in m²
    capital: float
    price_meat_avg: float
    enable_forecasting: bool
    safety_margin: float
    include_labor_cost: bool = False  # Toggle for labor accounting

    # --- BIOLOGY (Hard Data) ---
    carrying_capacity_mean: float = 7.0
    carrying_capacity_std: float = 1.0
    fertility_mean: float = 1.5
    fertility_std: float = 0.2
    mortality_lamb_mean: float = 0.10
    mortality_lamb_std: float = 0.03
    mortality_ewe_mean: float = 0.04
    mortality_ewe_std: float = 0.01
    feed_intake_ewe: float = 2.2      # kg dry matter/day (ewes)
    feed_intake_lamb: float = 1.2     # kg dry matter/day (lambs)
    feed_intake_std: float = 0.3
    hay_yield_ha_mean: float = 12.0   # Bales per hectare (updated)
    hay_yield_ha_std: float = 3.0
    bale_weight_kg: float = 250.0     # Weight of one bale
    bale_volume_m3: float = 1.4       # Volume for barn storage
    
    # --- ECONOMICS ---
    price_meat_std: float = 10.0 
    price_bale_sell_winter: float = 800.0   # Selling hay in winter
    price_bale_sell_summer: float = 400.0   # Selling hay surplus in summer
    price_ram_purchase: float = 10000.0     # Breeding ram cost
    
    # Feed costs
    cost_feed_own_mean: float = 2.5
    cost_feed_own_std: float = 0.5
    cost_feed_market_mean: float = 8.0
    cost_feed_market_std: float = 2.0
    
    # Detailed costs
    cost_vet_per_sheep: float = 350.0       # Medicines, deworming/year
    cost_shearing: float = 50.0             # Shearing per animal
    cost_mowing_ha: float = 1500.0          # Mowing + fuel per ha
    cost_bale_production: float = 200.0     # Baling + netting per bale
    
    # Labor
    wage_hourly: float = 200.0              # Hourly wage
    labor_hours_per_ewe_year: float = 6.0   # Hours per ewe per year
    
    # Subsidies (Updated Czech rates)
    subsidy_ha_mean: float = 8500.0
    subsidy_ha_std: float = 200.0
    subsidy_sheep_mean: float = 603.0
    subsidy_sheep_std: float = 50.0
    
    # Land tax
    tax_land_ha: float = 500.0
    
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
        
        # --- DETAILED HERD STRUCTURE ---
        self.ewes = cfg.barn_capacity              # Bahnice (matky)
        self.ewe_ages = [3.0] * cfg.barn_capacity  # Age of each ewe (years) - NEW
        self.rams_breeding = max(1, cfg.barn_capacity // 25)  # Breeding rams (1 per 25-30 ewes)
        self.ram_age = 3.0                         # Age of breeding ram - NEW
        self.lambs_male = 0                        # Beránci na výkrm
        self.lambs_female = 0                      # Jehničky na obnovu
        self.lamb_age = 0.0                        # Age of lambs (years) - NEW
        
        # --- STATES ---
        self.cash = cfg.capital
        self.hay_stock_bales = 25.0                # Starting hay stock
        self.bcs = 3.0                             # Body condition score
        
        # --- LAND SPLIT ---
        self.area_meadow = cfg.land_area * cfg.meadow_share     # Only for hay
        self.area_pasture = cfg.land_area * (1 - cfg.meadow_share) # Only for grazing
        
        # --- SEASONAL LOGIC ---
        self.is_winter_mode = True                 # Start in winter
        self.winter_end_day = np.random.randint(70, 90)  # Stochastic spring start (mid-March)
        self.winter_start_day = 280                # Around early October
        
        self.history = []
        self.event_log = []  # Event logging
        self.feed_log = {"Grazing": 0, "Stored": 0, "Market": 0}

        # Grass growth curve by month
        self.grass_curve = {
            1:0, 2:0, 3:0.1, 4:0.5, 5:1.2, 6:1.1, 
            7:0.7, 8:0.5, 9:0.7, 10:0.4, 11:0.1, 12:0
        }

    def _check_barn_capacity(self):
        """Check if hay fits in barn. Sell excess."""
        max_volume = self.cfg.barn_area_m2 * 3.0  # Assume 3m stacking height
        max_bales = int(max_volume / self.cfg.bale_volume_m3)
        
        if self.hay_stock_bales > max_bales:
            excess = self.hay_stock_bales - max_bales
            income = excess * self.cfg.price_bale_sell_summer
            self.cash += income
            self.hay_stock_bales = max_bales
            self.event_log.append(f"{self.date.date()}: ⚠️ Stodola plná! Prodáno {int(excess)} balíků za {int(income)} Kč.")

    def _get_seasonal_overhead(self, month):
        base_daily = self.cfg.overhead_base_year / 365
        if month in [6, 7, 8]: return base_daily * 1.5
        elif month in [1, 2, 12]: return base_daily * 1.3
        else: return base_daily * 0.8

    def _perform_forecast(self):
        """Estimate winter feeding costs."""
        total_adults = self.ewes + self.rams_breeding
        winter_feed_cost = 180 * self.cfg.feed_intake_ewe * self.cfg.cost_feed_market_mean * total_adults
        winter_overhead = (self.cfg.overhead_base_year / 2) * 1.2
        return (winter_feed_cost + winter_overhead) * (1.0 + self.cfg.safety_margin)

    def step(self):
        month = self.date.month
        day = self.date.dayofyear
        
        # --- 1. MEAT PRICE (With Easter effect) ---
        base_meat_price = get_stochastic_value(self.cfg.price_meat_avg, self.cfg.price_meat_std)
        # Easter price premium (March-April, months 3-4)
        if month in [3, 4]:
            base_meat_price *= np.random.uniform(1.2, 1.4)  # 20-40% premium
        
        # --- 2. SEASONAL MODE LOGIC (Stochastic winter/summer transition) ---
        if self.is_winter_mode and day > self.winter_end_day:
            if np.random.random() < 0.15:  # 15% chance each day after threshold
                self.is_winter_mode = False
                self.event_log.append(f"{self.date.date()}: 🌱 Začalo jaro! Ovce jdou na pastvu.")
        
        if not self.is_winter_mode and day > self.winter_start_day:
            if np.random.random() < 0.15:
                self.is_winter_mode = True
                self.winter_end_day = int(get_stochastic_value(80, 10))  # Set next year's spring end
                self.event_log.append(f"{self.date.date()}: ❄️ Začala zima! Ovce se stahují do ovčína.")

        # --- 3. BIOLOGY & FEEDING ---
        total_adults = self.ewes + self.rams_breeding
        total_lambs = self.lambs_male + self.lambs_female
        
        # Feed demand (kg dry matter)
        demand_kg = (total_adults * self.cfg.feed_intake_ewe) + (total_lambs * self.cfg.feed_intake_lamb)
        
        feed_cost = 0.0
        feed_source = ""
        # daily trackers for specific expenses (for validation breakdown)
        day_vet = 0.0
        day_mow = 0.0
        day_shearing = 0.0
        day_ram_purchase = 0.0
        
        if self.is_winter_mode:
            # WINTER: Feed only from hay storage
            # Assume 20% waste (trampling, spillage in feeder)
            needed_bales = (demand_kg * 1.2) / self.cfg.bale_weight_kg
            
            # Take what we have
            real_fed_bales = min(self.hay_stock_bales, needed_bales)
            self.hay_stock_bales -= real_fed_bales
            
            # Cost of own hay (just labor/handling in winter)
            feed_cost = real_fed_bales * 50  # 50 CZK per bale for handling
            
            # If short on hay, buy from market
            if real_fed_bales < needed_bales:
                missing = needed_bales - real_fed_bales
                cost_buy = missing * get_stochastic_value(self.cfg.price_bale_sell_winter, 100)
                feed_cost += cost_buy
                self.bcs = max(2.0, self.bcs - 0.001)  # Stress penalty
                feed_source = "Market"
            else:
                self.bcs = max(2.5, self.bcs - 0.0005)  # Maintenance loss
                feed_source = "Stored"
                
        else:
            # SUMMER: Grazing on pastures
            growth_factor = self.grass_curve[month]
            available_grass = self.area_pasture * 35.0 * growth_factor * np.random.normal(1.0, 0.2)
            
            if available_grass >= demand_kg:
                # Enough grass
                self.bcs = min(4.0, self.bcs + 0.003)
                feed_cost = demand_kg * 0.2  # Minimal cost (minerals, water)
                feed_source = "Grazing"
            else:
                # Drought: supplement with hay
                missing_kg = demand_kg - available_grass
                # Summer losses higher than winter (trampling on dry ground): apply 35% waste
                needed_bales = (missing_kg * 1.35) / self.cfg.bale_weight_kg
                
                taken = min(self.hay_stock_bales, needed_bales)
                self.hay_stock_bales -= taken
                
                # Cost of hay feeding in summer
                feed_cost = available_grass * 0.2
                feed_cost += taken * 50  # Handling cost
                
                # If we run out of hay in summer, buy expensive supplement
                if taken < needed_bales:
                    missing_bales = needed_bales - taken
                    feed_cost += missing_bales * self.cfg.price_bale_sell_summer
                    self.bcs -= 0.005
                    feed_source = "Market"
                else:
                    feed_source = "Stored"

        self.feed_log[feed_source] += 1

        # --- 4. DAILY MORTALITY (All groups) ---
        # Ewes: Age-dependent mortality (older ewes die more often)
        if self.ewes > 0:
            avg_ewe_age = np.mean(self.ewe_ages)
            age_factor = 1.0 + (max(0, avg_ewe_age - 4.0) * 0.2)  # Mortality increases after 4 years
            ewe_mort_prob = (self.cfg.mortality_ewe_mean / 365) * age_factor * (2 if self.bcs < 2.5 else 1)
            ewe_deaths = np.random.binomial(self.ewes, ewe_mort_prob)
            
            if ewe_deaths > 0:
                # Remove oldest ewes first (realistic)
                self.ewe_ages = sorted(self.ewe_ages)
                self.ewe_ages = self.ewe_ages[ewe_deaths:]
                self.ewes = max(0, self.ewes - ewe_deaths)
        
        # Rams: Similar to ewes but slightly higher (more aggressive)
        if self.rams_breeding > 0:
            ram_mort_prob = (self.cfg.mortality_ewe_mean / 365) * 1.3 * (2 if self.bcs < 2.5 else 1)  # 30% higher
            ram_deaths = np.random.binomial(self.rams_breeding, ram_mort_prob)
            # Remove dead rams
            self.rams_breeding = max(0, self.rams_breeding - ram_deaths)
            if ram_deaths > 0:
                # Record replacement purchase as daily ram purchase (avoid double cash-debit)
                replace_cost = ram_deaths * self.cfg.price_ram_purchase
                day_ram_purchase += replace_cost
                # restore breeding count to keep flock functional
                self.rams_breeding += ram_deaths
                # new ram(s) are young
                self.ram_age = 0.5
                self.event_log.append(f"{self.date.date()}: 🐑 Beran zemřel (x{ram_deaths}). Koupě náhrad za {int(replace_cost)} Kč.")
        
        # Lambs: MUCH higher mortality (10% annual = 0.027% daily)
        # Newborns are most vulnerable
        if (self.lambs_male + self.lambs_female) > 0:
            total_lambs = self.lambs_male + self.lambs_female
            # Lambs born in March, so in spring they are most vulnerable
            lamb_vulnerability = 2.5 if month in [3, 4, 5] else 2.0  # Higher mortality in spring
            lamb_mort_prob = (self.cfg.mortality_lamb_mean / 365) * lamb_vulnerability * (2 if self.bcs < 2.5 else 1)
            lamb_deaths = np.random.binomial(total_lambs, lamb_mort_prob)
            
            if lamb_deaths > 0:
                # Distribute deaths proportionally to male and female
                male_share = self.lambs_male / total_lambs if total_lambs > 0 else 0.5
                male_deaths = int(lamb_deaths * male_share)
                female_deaths = lamb_deaths - male_deaths
                
                self.lambs_male = max(0, self.lambs_male - male_deaths)
                self.lambs_female = max(0, self.lambs_female - female_deaths)

        # --- 5. CASHFLOW PLANNING & CALENDAR EVENTS ---
        # Crisis planner: estimate winter cost and perform emergency hay sale if enabled
        income = 0.0
        variable_costs = 0.0

        if self.cfg.enable_forecasting and month == 10 and self.date.day == 1:
            forecast_cost = self._perform_forecast()
            # Estimate October sales from current lambs (males sold, ~20% of females sold)
            est_sell_females = int(self.lambs_female * 0.2)
            oct_sales_estimate = (self.lambs_male * 40.0 * base_meat_price) + (est_sell_females * 35.0 * base_meat_price)
            projected_cash = self.cash + oct_sales_estimate - forecast_cost

            # If projected cash is negative, sell part of hay stocks to stabilise cashflow
            if projected_cash < 0 and self.hay_stock_bales > 10:
                # Sell up to 30% of current hay, or enough to cover projected shortfall
                needed = int(min(self.hay_stock_bales * 0.3, max(0, (abs(projected_cash) // max(1, self.cfg.price_bale_sell_summer)) + 1)))
                if needed > 0:
                    emergency_income = needed * self.cfg.price_bale_sell_summer
                    self.cash += emergency_income
                    self.hay_stock_bales -= needed
                    self.event_log.append(f"{self.date.date()}: 🚨 Krizový prodej sena! Prodáno {int(needed)} balíků za {int(emergency_income)} Kč pro cashflow.")
        
        # A) MARCH 15: LAMBING & SPRING VET (50/50 M/F split)
        if month == 3 and self.date.day == 15:
            # Fertility depends on BCS
            fert = get_stochastic_value(self.cfg.fertility_mean, self.cfg.fertility_std)
            if self.bcs < 2.5:
                fert *= 0.7  # Penalty for poor condition
            
            total_born = int(self.ewes * fert)
            males = int(total_born * 0.5)
            females = total_born - males
            
            self.lambs_male += males
            self.lambs_female += females
            self.lamb_age = 0.0  # Newborns
            self.bcs -= 0.5  # Energy depletion from pregnancy
            
            # Spring vet (deworming)
            total_heads = self.ewes + self.rams_breeding + self.lambs_male + self.lambs_female
            vet_cost = total_heads * (self.cfg.cost_vet_per_sheep / 2)
            variable_costs += vet_cost
            day_vet += vet_cost
            
            self.event_log.append(f"{self.date.date()}: 🍼 Narozeno {total_born} jehňat ({males}M / {females}F). Vet: {int(vet_cost)} Kč.")

        # B) MAY 15: SHEARING ONLY (veterinary removed)
        if month == 5 and self.date.day == 15:
            # Shearing all adults
            shearing_cost = (self.ewes + self.rams_breeding) * self.cfg.cost_shearing
            variable_costs += shearing_cost
            day_shearing += shearing_cost
            self.event_log.append(f"{self.date.date()}: ✂️ Stříhání. Náklad: {int(shearing_cost)} Kč.")

        # C) JUNE 10 & SEPT 10: MOWING & BALING
        if (month == 6 and self.date.day == 10) or (month == 9 and self.date.day == 10):
            is_second_cut = (month == 9)
            yield_factor = 0.6 if is_second_cut else 1.0  # Second cut is weaker
            
            current_yield = get_stochastic_value(self.cfg.hay_yield_ha_mean, self.cfg.hay_yield_ha_std) * yield_factor
            
            # Hay made ONLY from meadows
            new_bales = self.area_meadow * current_yield
            self.hay_stock_bales += new_bales
            
            # Costs: mowing (fuel, machine) + baling (twine, labor)
            mow_cost = (self.area_meadow * self.cfg.cost_mowing_ha) + (new_bales * self.cfg.cost_bale_production)
            variable_costs += mow_cost
            day_mow += mow_cost
            
            cut_name = "Otava" if is_second_cut else "Senoseč"
            self.event_log.append(f"{self.date.date()}: 🚜 {cut_name}. Vyrobeno {int(new_bales)} balíků. Náklad: {int(mow_cost)} Kč.")
            
            # Check barn capacity
            self._check_barn_capacity()

        # D) OCTOBER 15: BIG MANAGEMENT DAY
        if month == 10 and self.date.day == 15:
            # 1) Sell all male lambs
            if self.lambs_male > 0:
                meat_income = self.lambs_male * 40.0 * base_meat_price  # 40 kg average
                income += meat_income
                sold_males = self.lambs_male
                self.lambs_male = 0
            else:
                sold_males = 0
                meat_income = 0
            
            # 2) Sort female lambs (80% keep for breeding, 20% sell)
            keep_ratio = 0.8
            keep_females = int(self.lambs_female * keep_ratio)
            sell_females = self.lambs_female - keep_females
            
            if sell_females > 0:
                income += sell_females * 35.0 * base_meat_price  # Lighter than males
            
            # 3) Cull old ewes (prefer older ones - culling age 5+ years)
            # Prioritize removing the oldest ewes
            ewes_to_cull = []
            if len(self.ewe_ages) > 0:
                # Sort indices by age (oldest first) and choose culls preferring 5+ year olds
                sorted_indices = sorted(range(len(self.ewe_ages)), key=lambda i: self.ewe_ages[i], reverse=True)
                target_cull = int(self.ewes * 0.15)  # 15% annual turnover

                # First pass: select oldest >4y for culling
                cull_set = set()
                for idx in sorted_indices:
                    if len(cull_set) < target_cull and self.ewe_ages[idx] > 4.0:
                        cull_set.add(idx)

                # If still need to cull, take next-oldest regardless of age
                i = 0
                while len(cull_set) < target_cull and i < len(sorted_indices):
                    idx = sorted_indices[i]
                    if idx not in cull_set:
                        cull_set.add(idx)
                    i += 1

                # Build new ewe ages excluding culled indices
                new_ewe_ages = [age for i, age in enumerate(self.ewe_ages) if i not in cull_set]
                cull_ewes = len(cull_set)
                self.ewe_ages = new_ewe_ages
                self.ewes = len(self.ewe_ages)
            else:
                cull_ewes = 0
            
            if cull_ewes > 0:
                income += cull_ewes * 60.0 * (base_meat_price * 0.7)  # Cull meat is cheaper
            
            # Herd renewal - add young ewes from female lambs
            self.ewes = max(1, self.ewes + keep_females)
            self.ewe_ages.extend([0.5] * keep_females)  # New ewes are 6 months old
            self.lambs_female = 0
            
            # 4) Ram replacement (every 2 years)
            if self.date.year % 2 == 0:
                # Replace a share of rams proportional to flock size (approx. 50% of rams every 2 years)
                replace_count = max(1, int(round(self.rams_breeding * 0.5)))
                replace_cost = replace_count * self.cfg.price_ram_purchase
                # record purchase as daily ram purchase expense (accounting will add it below)
                day_ram_purchase += replace_cost
                # Sell old rams for meat
                income += replace_count * 80.0 * (base_meat_price * 0.6)
                self.ram_age = 0.5  # New rams are young
                self.event_log.append(f"{self.date.date()}: 🐏 Výměna plemenných beranů x{replace_count}. Cena: {int(replace_cost)} Kč.")
            
            # 5) Fall deworming
            vet_cost = (self.ewes + self.rams_breeding + self.lambs_male + self.lambs_female) * (self.cfg.cost_vet_per_sheep / 2)
            variable_costs += vet_cost
            # Ensure October vet shows up in daily vet expense (Exp_Vet)
            day_vet += vet_cost
            self.event_log.append(f"{self.date.date()}: 💰 Říjnový prodej. Tržba: {int(income)} Kč. Vet: {int(vet_cost)} Kč. Prodáno: {sold_males}M, {sell_females}F, {cull_ewes}E (avg age: {np.mean(self.ewe_ages):.1f} let).")

        # E) NOV 20 & APR 20: SUBSIDIES (stochastic)
        if month == 11 and self.date.day == 20:
            # 70% advance payment
            sub_ha = get_stochastic_value(self.cfg.subsidy_ha_mean, self.cfg.subsidy_ha_std)
            sub_sheep = get_stochastic_value(self.cfg.subsidy_sheep_mean, self.cfg.subsidy_sheep_std)
            sub = (self.cfg.land_area * sub_ha) + (self.ewes * sub_sheep)
            income += sub * 0.7
        if month == 4 and self.date.day == 20:
            # 30% final payment
            sub_ha = get_stochastic_value(self.cfg.subsidy_ha_mean, self.cfg.subsidy_ha_std)
            sub_sheep = get_stochastic_value(self.cfg.subsidy_sheep_mean, self.cfg.subsidy_sheep_std)
            sub = (self.cfg.land_area * sub_ha) + (self.ewes * sub_sheep)
            income += sub * 0.3

        # F) DEC 31: LAND TAX
        if month == 12 and self.date.day == 31:
            land_tax = self.cfg.land_area * self.cfg.tax_land_ha
            variable_costs += land_tax
            self.event_log.append(f"{self.date.date()}: 🏛️ Daň z pozemku: {int(land_tax)} Kč.")

        # --- 6. DAILY LABOR COST ---
        labor_cost = 0.0
        if self.cfg.include_labor_cost:
            # Count labour per adult animals only (ewes + breeding rams).
            total_adult_heads = self.ewes + self.rams_breeding
            daily_hours = (total_adult_heads * self.cfg.labor_hours_per_ewe_year) / 365
            labor_cost = daily_hours * self.cfg.wage_hourly

        # --- 7. RANDOM SHOCKS (Illness, accident) ---
        shock_cost = 0.0
        if np.random.random() < self.cfg.shock_prob_daily:
            shock_cost = get_stochastic_value(self.cfg.shock_cost_mean, self.cfg.shock_cost_std)
            if shock_cost > 0:
                self.event_log.append(f"{self.date.date()}: ⚡ Neočekávaná nehoda. Náklad: {int(shock_cost)} Kč.")

        # --- 8. FINALIZE DAY ---
        daily_overhead = self._get_seasonal_overhead(month)
        
        # --- AGE ALL ANIMALS ---
        # Every year, increase age by 1
        if month == 1 and self.date.day == 1:  # January 1st = birthday
            self.ewe_ages = [age + 1.0 for age in self.ewe_ages]  # Ewes age
            self.ram_age += 1.0  # Ram ages
            self.lamb_age += 1.0  # Lambs age (until they become ewes or are sold)
        
        # Add any ram purchases recorded during the day to variable costs
        variable_costs += day_ram_purchase

        total_income = income
        total_expense = feed_cost + variable_costs + daily_overhead + labor_cost + shock_cost
        
        self.cash += total_income - total_expense
        
        # --- LOGGING ---
        self.history.append({
            "Date": self.date,
            "Cash": self.cash,
            "Total Animals": self.ewes + self.rams_breeding + self.lambs_male + self.lambs_female,
            "Ewes": self.ewes,
            "Rams": self.rams_breeding,
            "Lambs Male": self.lambs_male,
            "Lambs Female": self.lambs_female,
            "Hay Stock": self.hay_stock_bales,
            "BCS": self.bcs,
            "Income": total_income,
            "Exp_Feed": feed_cost,
            "Exp_Vet": day_vet,
            "Exp_Mow": day_mow,
            "Exp_Shearing": day_shearing,
            "Exp_RamPurchase": day_ram_purchase,
            "Exp_Variable": variable_costs,
            "Exp_Overhead": daily_overhead,
            "Exp_Labor": labor_cost,
            "Exp_Shock": shock_cost,
            "Is_Winter": int(self.is_winter_mode),
            "Meat_Price": base_meat_price
        })
        
        self.date += pd.Timedelta(days=1)

    def run(self):
        days = self.cfg.sim_years * 365
        for _ in range(days):
            self.step()
        return pd.DataFrame(self.history).set_index("Date")

# --- SIDEBAR UI ---
with st.sidebar:
    st.title("Ovčí farma")
    st.markdown("**Proměnné modelu**")
    
    st.header("1. Půda a budovy")
    area = st.number_input("Celková plocha (ha)", 5.0, 100.0, 15.0)
    meadow_pct = st.slider("Podíl luk na seno (%)", 0, 100, 40, help="Část plochy jen na výrobu sena (pastva zakázana)")
    barn_m2 = st.number_input("Velikost stodoly (m²)", 50, 1000, 150, help="Stacking height: 3m")
    
    st.header("2. Stádo a ekonomika")
    start_ewes = st.slider("Počet bahnic (start)", 10, 200, 20)
    meat_price = st.slider("Cena masa (Kč/kg)", 60.0, 120.0, 85.0)
    cap = st.number_input("Počáteční kapitál (CZK)", value=200000)
    
    st.header("3. Pokročilé")
    labor_on = st.checkbox("🕐 Započítat náklady na vlastní práci", False, help="6h/rok na bahnici @ 200 Kč/h")
    use_forecast = st.toggle("Cashflow Planner", value=True)
    
    st.markdown("---")
    st.header("Parametry modelu")
    
    # Instance for defaults
    d = HardDataConfig(0,0,0,0,0,0,0,0,0)
    
    with st.expander("Biologie", expanded=False):
        st.code(f"""
Plodnost:       {d.fertility_mean} ± {d.fertility_std}
Úhyn jehňat:    {d.mortality_lamb_mean*100:.0f}%
Úhyn bahnic:    {d.mortality_ewe_mean*100:.0f}%
Spotřeba (bahnice):  {d.feed_intake_ewe} kg/den
Spotřeba (jehně):    {d.feed_intake_lamb} kg/den
Výnos sena:     {d.hay_yield_ha_mean} balíků/ha
Váha balíku:    {d.bale_weight_kg} kg
        """)
    
    with st.expander("Ceny a dotace", expanded=False):
        st.code(f"""
Krmivo (vlastní):    {d.cost_feed_own_mean} Kč/kg
Krmivo (nákup):      {d.cost_feed_market_mean} Kč/kg
Seno (zima):         {d.price_bale_sell_winter} Kč/balík
Seno (léto):         {d.price_bale_sell_summer} Kč/balík

Veterina/rok:        {d.cost_vet_per_sheep} Kč/ks
Stříhání:            {d.cost_shearing} Kč/ks
Seč:                 {d.cost_mowing_ha} Kč/ha
Lisování:            {d.cost_bale_production} Kč/balík

SAPS dotace:    {d.subsidy_ha_mean:,.0f} Kč/ha
VDJ dotace:     {d.subsidy_sheep_mean:,.0f} Kč/ks
Daň z pozemků:  {d.tax_land_ha} Kč/ha
        """)
    
    with st.expander("Ostatní", expanded=False):
        st.code(f"""
Režie/rok:           {d.overhead_base_year:,.0f} Kč
Práce:               {d.labor_hours_per_ewe_year}h/rok @ {d.wage_hourly} Kč/h
Beran:               {d.price_ram_purchase:,.0f} Kč (výměna)
Šok (riskní):        {d.shock_prob_daily*100:.1f}% / den
        """)

# --- RUN SIMULATION ---
cfg = HardDataConfig(
    sim_years=5, 
    land_area=area, 
    meadow_share=meadow_pct/100.0, 
    barn_capacity=start_ewes,
    barn_area_m2=barn_m2,
    capital=cap,
    price_meat_avg=meat_price, 
    enable_forecasting=use_forecast, 
    safety_margin=0.2,
    include_labor_cost=labor_on
)

model = FarmBIModel(cfg)
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
col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)

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
    total_exp_monthly = df_monthly["Exp_Feed"] + df_monthly["Exp_Variable"] + df_monthly["Exp_Overhead"] + df_monthly["Exp_Labor"] + df_monthly["Exp_Shock"]
    ax.bar(df_monthly.index, -df_monthly["Exp_Feed"], label="Krmivo", color="#e67e22", width=20)
    ax.bar(df_monthly.index, -df_monthly["Exp_Variable"], bottom=-df_monthly["Exp_Feed"], label="Veterina+Seč", color="#9b59b6", width=20)
    ax.bar(df_monthly.index, -df_monthly["Exp_Overhead"], bottom=-(df_monthly["Exp_Feed"]+df_monthly["Exp_Variable"]), label="Režie", color="#3498db", width=20)
    ax.bar(df_monthly.index, -df_monthly["Exp_Labor"], bottom=-(df_monthly["Exp_Feed"]+df_monthly["Exp_Variable"]+df_monthly["Exp_Overhead"]), label="Práce", color="#1abc9c", width=20)
    ax.bar(df_monthly.index, -df_monthly["Exp_Shock"], bottom=-(df_monthly["Exp_Feed"]+df_monthly["Exp_Variable"]+df_monthly["Exp_Overhead"]+df_monthly["Exp_Labor"]), label="Šoky", color="#e74c3c", width=20)
    
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
        "Režie": df["Exp_Overhead"].sum(),
        "Práce": df["Exp_Labor"].sum(),
        "Šoky": df["Exp_Shock"].sum()
    }
    fig_pie, ax_pie = plt.subplots(figsize=(6, 5))
    colors_pie = ["#e67e22", "#9b59b6", "#3498db", "#1abc9c", "#e74c3c"]
    ax_pie.pie(total_exp.values(), labels=total_exp.keys(), autopct='%1.1f%%', colors=colors_pie, startangle=90)
    st.pyplot(fig_pie)

# --- 5. SEASONAL ANALYSIS ---
st.subheader("Sezónní analýza")

col_season, col_price = st.columns(2)

with col_season:
    st.markdown("**Průměrný Denní Cashflow po Měsících**")
    
    df_month = df.copy()
    df_month["Month"] = df_month.index.month
    df_month["Daily_Flow"] = df_month["Income"] - (df_month["Exp_Feed"] + df_month["Exp_Variable"] + df_month["Exp_Overhead"] + df_month["Exp_Labor"] + df_month["Exp_Shock"])
    
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
    "2. Veterina a Služby (Kč/ks)": 500,       # Léky + Stříhání
    "3. Režie a Opravy (Kč/ks)": 1500,        # Nafta (seč), Energie, Pojištění
    "4. Tržby Maso (Kč/ks)": 2900,
    "5. Zisk bez dotací (Kč/ks)": -1150,
    "6. Odchov (ks jehňat/matku)": 1.35,
    "7. Závislost na dotacích (%)": 65.0
}

# 2. Calculate model metrics
avg_ewes = df["Ewes"].mean()
if avg_ewes == 0: avg_ewes = 1
years = cfg.sim_years

# Economics per ewe (Annualized)
model_feed = df["Exp_Feed"].sum() / (avg_ewes * years)

# Rozdělení "Overhead" na Veterinu/Služby a Skutečnou Režii
# Veterina a Služby = Vet + Shearing + Ram Purchase
model_vet_services = (df["Exp_Vet"].sum() + df["Exp_Shearing"].sum() + df["Exp_RamPurchase"].sum()) / (avg_ewes * years)

# Režie a Opravy = Mow (nafta) + Overhead (fixní) + Shock + Labor
model_overhead_real = (df["Exp_Mow"].sum() + df["Exp_Overhead"].sum() + df["Exp_Shock"].sum() + df["Exp_Labor"].sum()) / (avg_ewes * years)

# Meat Income
oct_income = df[df.index.month == 10]["Income"].sum()
model_meat = (oct_income / (avg_ewes * years)) if oct_income > 0 else (df["Income"].sum() / (avg_ewes * years))

# Profit
model_profit_no_sub = model_meat - (model_feed + model_vet_services + model_overhead_real)

# Biology (Reproduction)
avg_lamb_peak = df[df.index.month == 6]["Lambs Male"].mean() + df[df.index.month == 6]["Lambs Female"].mean()
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

# Display table and KPI checks
col_val1, col_val2 = st.columns([4, 3])

with col_val1:
    st.markdown("###  Detailní Srovnání")
    st.dataframe(validation_df.style.format("{:,.0f}", subset=["Průměr ČR (Realita)", "Tvůj Model", "Odchylka"]), use_container_width=True, height=300)

with col_val2:
    st.markdown("###  Interpretace")
    
    # Check overheads
    if model_overhead_real > 2000:
        st.warning(f" **Vysoká Režie ({model_overhead_real:.0f} Kč/ks)**. Důvodem je pravděpodobně započítaná cena práce nebo malý počet ovcí (fixní náklady se málo rozpočítají).")
    
    # Profitability check
    diff_profit = model_profit_no_sub - benchmark_data["5. Zisk bez dotací (Kč/ks)"]
    if model_profit_no_sub > 0:
        st.error(f" **PŘÍLIŠ ZISKOVÉ!** Zisk {model_profit_no_sub:.0f} Kč. V ČR je to nereálné bez dotací.")
    elif abs(diff_profit) < 500:
        st.success(" **EKONOMIKA SEDÍ:** Ztráta odpovídá realitě.")
    
    # Subsidy check
    st.metric("Závislost na dotacích", f"{model_subsidy_dep:.1f} %")

