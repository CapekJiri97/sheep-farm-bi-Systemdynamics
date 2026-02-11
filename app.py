# Importujeme knihovny pro UI, data a grafy
# streamlit (st): Framework pro tvorbu webové aplikace.
# altair (alt): Knihovna pro interaktivní grafy.
# model: Náš vlastní modul (soubor model.py), odkud bereme logiku farmy.
import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
import time

from model import FarmConfig, FarmModel, SCENARIOS, BASE_SCENARIO

# --- CONFIGURATION ---
# Nastavení stránky (titulek, ikona, rozložení na celou šířku).
st.set_page_config(page_title="Ovčí farma - Systémová dynamika", layout="wide", page_icon="🚜")

# --- SESSION STATE INIT ---
# Session State slouží k uchování dat mezi obnoveními stránky (reruns).
# Streamlit spouští celý skript znovu při každé interakci uživatele.
if 'custom_scenarios' not in st.session_state:
    st.session_state['custom_scenarios'] = {}

# --- SIDEBAR UI ---
# 'with st.sidebar:' definuje blok kódu, který vykreslí prvky do levého panelu.
with st.sidebar:
    st.title("Ovčí farma")
    
    st.markdown("### Režim aplikace")
    mode_switch = st.radio("Režim aplikace", ["Jednotlivá simulace", "Monte Carlo Laboratoř"], horizontal=True, help="Přepne na hromadné testování scénářů.", label_visibility="collapsed")
    st.markdown("---")
    
    # Placeholder for Save Scenario UI (to be rendered after inputs are defined)
    save_sc_container = st.container()
    
    # --- TABS FOR BETTER UI ORGANIZATION ---
    # Rozdělení nastavení do záložek pro přehlednost.
    tab_main, tab_strat, tab_details = st.tabs(["Základ", "Strategie", "Detaily"])
    
    with tab_main:
        # st.slider: Vytvoří posuvník. Vrací hodnotu, kterou uživatel vybral.
        # st.number_input: Vytvoří pole pro zadání čísla.
        st.header("1. Kapacita a Infrastruktura")
        target_ewes = st.slider("Cílová kapacita (ovčín)", 10, 500, 60, help="Maximální počet bahnic. Určuje velikost potřebné budovy.")
        
        req_m2 = int(target_ewes * 2.5) # 2.5 m2 per ewe
        barn_m2 = st.number_input("Velikost ovčína (m²)", 20, 2000, max(20, req_m2), help=f"Pro zvířata. Doporučeno: {req_m2} m² (2.5 m²/ks vč. jehňat a uliček)")
        hay_barn_m2 = st.number_input("Velikost seníku (m²)", 50, 2000, 100, help="Pro uskladnění sena. 100 m² pojme cca 200 balíků (při stohování 3m).")
        
        area = st.number_input("Celková plocha (ha)", 5.0, 100.0, 15.0)
        meadow_pct = st.slider("Podíl luk na seno (%)", 0, 100, 40, help="Část plochy jen na výrobu sena (pastva zakázana)")
        
        st.header("2. Stádo a ekonomika")
        start_ewes = st.slider("Počet bahnic (start)", 1, target_ewes, min(20, target_ewes), help="Kolik ovcí nakoupíte do začátku.")
        meat_price = st.slider("Maloobchodní cena (Ze dvora) Kč/kg", 60.0, 150.0, 85.0, help="Cena pro lokální prodej (ze dvora).")
        start_hay = st.number_input("Počáteční zásoba sena (balíky)", 0, 500, 25)
        cap = st.number_input("Počáteční kapitál (Kč)", value=200000)
        labor_on = st.checkbox("Započítat náklady na práci", True, help="Mzdy za odpracované hodiny (cca 6h/rok na bahnici).")

    with tab_strat:
        st.header("3. Pokročilé")
        
        # --- CLIMATE PRESETS LOGIC ---
        # Inicializace proměnných v session state pro slidery počasí, pokud neexistují.
        if 'rain_val' not in st.session_state: st.session_state['rain_val'] = 100
        if 'drought_val' not in st.session_state: st.session_state['drought_val'] = 0.5
        if 'winter_val' not in st.session_state: st.session_state['winter_val'] = 100

        def update_climate_preset():
            sel = st.session_state.climate_selector
            if sel == "Normální":
                st.session_state.rain_val, st.session_state.drought_val, st.session_state.winter_val = 100, 0.5, 100
            elif sel == "Suchý":
                st.session_state.rain_val, st.session_state.drought_val, st.session_state.winter_val = 70, 2.0, 80
            elif sel == "Horský":
                st.session_state.rain_val, st.session_state.drought_val, st.session_state.winter_val = 120, 0.1, 130

        # st.selectbox: Rozbalovací menu. on_change spustí funkci update_climate_preset při změně.
        st.selectbox("Klimatický profil (Přednastavení)", ["Normální", "Suchý", "Horský"], key="climate_selector", on_change=update_climate_preset, help="Nastaví posuvníky níže na typické hodnoty pro danou oblast.")
        climate = "UI_Custom" # Pro UI používáme tento speciální profil, který se řídí čistě posuvníky
        
        machinery_map = {"Služby": "Services", "Vlastní": "Own"}
        machinery_label = st.radio("Sklizeň sena (Seč a lisování)", list(machinery_map.keys()), help="Služby = pronájem; Vlastní = vlastní stroj")
        machinery = machinery_map[machinery_label]
        
        use_freezing = st.toggle("Aktivovat Mrazírny (Sektor 8)", value=True, help="Umožňuje mrazit maso a prodávat ho v průběhu roku za lepší ceny.")
        
        use_forecast = st.toggle("Plánovač Cashflow", value=True)
        
        # st.expander: Sbalitelná sekce pro pokročilá nastavení.
        with st.expander("Nastavení Počasí (Detail)", expanded=True):
            rain_mod = st.slider("Intenzita srážek (Růst trávy %)", 50, 150, key="rain_val", help="100% = Standardní růst.") / 100.0
            drought_add = st.slider("Riziko sucha (Denní %)", 0.0, 5.0, key="drought_val", step=0.1, help="Pravděpodobnost, že v letní den nastane sucho (tráva neroste).") / 100.0
            winter_mod = st.slider("Délka zimy (%)", 50, 150, key="winter_val", help="100% = Standardní délka zimy.") / 100.0
        
        with st.expander("Tržní Strategie (Velkoobchod)"):
            m_quota_kg = st.number_input("Limit prodeje ze dvora (kg masa/rok)", 0, 5000, 800, help="Kolik kg masa prodáte sousedům za plnou cenu.")
            m_wholesale = st.number_input("Výkupní cena (Nadprodukce) Kč/kg", 30.0, 80.0, 55.0, help="Cena pro výkup (jatka), když zahltíte lokální trh.")

        with st.expander("Systémová Dynamika (Zpoždění)"):
            delay_bcs = st.slider("Informační zpoždění (Vnímání kondice)", 1, 30, 10, help="Jak dlouho trvá, než si všimnete, že ovce hubnou.")
            delay_mat = st.slider("Materiálové zpoždění (Dodávka krmiva)", 0, 14, 3, help="Za jak dlouho přijede kamion s krmivem po objednání.")

    with tab_details:
        st.header("Detailní nastavení parametrů")
        
        with st.expander("Biologie a Produkce"):
            p_fertility = st.number_input("Plodnost (ks/bahnici)", 1.0, 3.0, 1.5, 0.1)
            p_mortality_lamb = st.number_input("Úhyn jehňat (%)", 0.0, 50.0, 10.0, 1.0) / 100.0
            p_mortality_ewe = st.number_input("Úhyn bahnic (%)", 0.0, 20.0, 4.0, 0.5) / 100.0
            p_feed_ewe = st.number_input("Spotřeba bahnice (kg sušiny/den)", 1.0, 4.0, 2.2, 0.1)
            p_hay_yield = st.number_input("Výnos sena (balíků/ha)", 5.0, 30.0, 12.0, 1.0)
            
        with st.expander("Provozní Náklady a Ceny"):
            c_feed_own = st.number_input("Cena vl. krmiva (Kč/kg)", 0.5, 10.0, 2.5, 0.1)
            c_feed_market = st.number_input("Cena kup. krmiva (Kč/kg)", 2.0, 20.0, 8.0, 0.5)
            c_vet = st.number_input("Veterina (Kč/ks/rok)", 100.0, 2000.0, 350.0, 50.0)
            c_shearing = st.number_input("Stříhání (Kč/ks)", 20.0, 200.0, 50.0, 10.0)
            c_ram = st.number_input("Cena berana (Kč)", 5000.0, 30000.0, 10000.0, 1000.0)
            c_bale_sell_winter = st.number_input("Cena sena Zima (Kč/balík)", 200.0, 2000.0, 800.0, 50.0)
            c_bale_sell_summer = st.number_input("Cena sena Léto (Kč/balík)", 100.0, 1000.0, 400.0, 50.0)
            
        with st.expander("Stroje a Služby"):
            s_mow_ha = st.number_input("Služba: Seč (Kč/ha)", 500.0, 5000.0, 1500.0, 100.0)
            s_bale = st.number_input("Služba: Lisování (Kč/ks)", 50.0, 500.0, 200.0, 10.0)
            o_capex = st.number_input("Vlastní: Cena stroje (Kč)", 100000.0, 5000000.0, 600000.0, 50000.0)
            o_fuel = st.number_input("Vlastní: Nafta seč (Kč/ha)", 100.0, 1000.0, 400.0, 50.0)
            o_repair = st.number_input("Vlastní: Opravy ročně (Kč)", 0.0, 100000.0, 15000.0, 1000.0)
            
        with st.expander("Logistika a Mrazírny (Sektor 8)"):
            p_freezer_cap = st.number_input("Kapacita mrazáku (kg)", 100.0, 5000.0, 500.0, 50.0)
            p_freezer_capex = st.number_input("Cena mrazáku (Kč)", 5000.0, 200000.0, 30000.0, 1000.0)
            p_elec_price = st.number_input("Cena elektřiny (Kč/kWh)", 1.0, 20.0, 6.0, 0.5)
            p_elec_usage = st.number_input("Spotřeba chlazení (kWh/kg/den)", 0.001, 0.5, 0.015, 0.001)

        with st.expander("Dotace a Daně"):
            sub_ha = st.number_input("SAPS (Kč/ha)", 0.0, 20000.0, 8500.0, 100.0)
            sub_sheep = st.number_input("VDJ (Kč/ks)", 0.0, 5000.0, 603.0, 10.0)
            tax_land = st.number_input("Daň z nemovitosti (Kč/ha)", 0.0, 2000.0, 500.0, 50.0)
            tax_build = st.number_input("Daň ze staveb (Kč/m²)", 0.0, 100.0, 15.0, 1.0)

        with st.expander("Režie a Škálování"):
            ov_base = st.number_input("Základní režie (Kč/rok)", 0.0, 200000.0, 40000.0, 1000.0)
            adm_base = st.number_input("Admin základ (Kč/rok)", 0.0, 50000.0, 5000.0, 500.0)
            adm_factor = st.number_input("Admin faktor (Diseconomy)", 1.0, 3.5, 2.0, 0.1, help="Exponent růstu administrativy. 1.0 = lineární, 1.5 = progresivní zátěž.")
            wage = st.number_input("Hodinová mzda (Kč/h)", 100.0, 1000.0, 200.0, 10.0)
            labor_h = st.number_input("Pracnost zvířata (h/ks/rok)", 1.0, 20.0, 6.0, 0.5)
            labor_ha = st.number_input("Pracnost půda (h/ha/rok)", 0.0, 50.0, 10.0, 1.0, help="Údržba ohradníků, pastvin, sečení nedopasků.")
            labor_fix = st.number_input("Fixní pracnost (h/rok)", 0.0, 1000.0, 200.0, 50.0, help="Údržba budov, administrativa, cesty.")
            labor_barn_m2 = st.number_input("Pracnost budovy (h/m²/rok)", 0.0, 10.0, 0.5, 0.1, help="Úklid, údržba, manipulace v ovčíně.")
            maint_barn_m2 = st.number_input("Údržba budovy (Kč/m²/rok)", 0.0, 1000.0, 60.0, 10.0, help="Opravy střechy, nátěry, dezinfekce.")
            shock_p = st.number_input("Pravděpodobnost šoku (denní %)", 0.0, 5.0, 0.5, 0.1) / 100.0

    # --- SAVE SCENARIO UI ---
    with save_sc_container:
        # Logika pro uložení vlastního scénáře do paměti (session state).
        with st.expander("Uložit aktuální nastavení (pro Monte Carlo)"):
            st.info("Tento scénář bude uložen pod kategorii **C (Vlastní)**.")
            new_sc_name = st.text_input("Název scénáře", placeholder="Např. Můj optimalizovaný chov")
            if st.button("Uložit scénář"):
                if new_sc_name:
                    # Vytvoříme konfiguraci na základě BASE_SCENARIO a přepíšeme ji aktuálními vstupy
                    custom_sc = BASE_SCENARIO.copy()
                    custom_sc.update({
                        "sim_years": 5, "land_area": area, "meadow_share": meadow_pct/100.0, "barn_capacity": target_ewes,
                        "initial_ewes": start_ewes, "barn_area_m2": barn_m2, "hay_barn_area_m2": hay_barn_m2, "capital": cap,
                        "price_meat_avg": meat_price, "market_quota_kg": m_quota_kg, "price_meat_wholesale": m_wholesale,
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
                        "labor_hours_fix_year": labor_fix, "labor_hours_barn_m2_year": labor_barn_m2, "shock_prob_daily": shock_p,
                        "enable_freezing": use_freezing, "freezer_capacity_kg": p_freezer_cap, "freezer_capex": p_freezer_capex,
                        "electricity_price": p_elec_price, "cooling_energy_per_kg": p_elec_usage
                    })
                    
                    # Uložíme do session state s prefixem "C." (Custom)
                    st.session_state['custom_scenarios'][f"C. {new_sc_name}"] = custom_sc
                    st.success(f"Scénář '{new_sc_name}' byl uložen! Najdete ho v Monte Carlo Lab pod skupinou 'C'.")
                else:
                    st.warning("Zadejte prosím název scénáře.")
    
    st.markdown("---")
    with st.expander("Seed (Opakovatelnost)", expanded=False):
        sim_seed = st.number_input("Seed simulace", value=1337420, min_value=0, max_value=9999999999, help="Fixní seed zajistí, že náhoda (počasí, ceny) bude stejná pro porovnání scénářů.")

if mode_switch == "Monte Carlo Laboratoř":
    # --- SEKCE MONTE CARLO ---
    st.title("Monte Carlo Laboratoř")
    st.markdown("Simulace tisíců běhů pro ověření robustnosti scénářů.")
    
    mc_cols = st.columns(3)
    n_runs = mc_cols[0].number_input("Počet běhů na scénář", 10, 2000, 50, help="Pro rychlý test dej 50. Pro finální data 1000.")
    
    with mc_cols[0]:
        sensitivity_on = st.checkbox("Citlivostní analýza", help="Náhodně mění vybrané parametry v každém běhu.")
        sens_map = {
            "Cena Masa": "price_meat_avg",
            "Cena Nafty": "own_mow_fuel_ha",
            "Počasí (Růst)": "rain_growth_global_mod",
            "Lokální Trh": "market_quota_kg",
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
    # Spojíme vestavěné scénáře s uživatelskými.
    # Work with a local copy to ensure clean state on every rerun
    active_scenarios_pool = SCENARIOS.copy()
    if st.session_state['custom_scenarios']:
        active_scenarios_pool.update(st.session_state['custom_scenarios'])
    
    # 2. Get all available groups dynamically (including "C")
    all_groups = sorted(list(set([k[0] for k in active_scenarios_pool.keys()])))
    
    selected_groups = mc_cols[2].multiselect("Vyber skupiny scénářů", all_groups, default=["1", "5"])
    
    # Filter scenarios based on selection
    active_scenarios = {k: v for k, v in active_scenarios_pool.items() if k[0] in selected_groups}
    
    # Tlačítko pro spuštění hromadné simulace.
    if st.button(f"Spustit simulaci ({len(active_scenarios) * n_runs} běhů)"):
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
            "price_meat_avg": meat_price, "market_quota_kg": m_quota_kg, "price_meat_wholesale": m_wholesale,
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
            "labor_hours_fix_year": labor_fix, "labor_hours_barn_m2_year": labor_barn_m2, "shock_prob_daily": shock_p,
            "enable_freezing": use_freezing, "freezer_capacity_kg": p_freezer_cap, "freezer_capex": p_freezer_capex,
            "electricity_price": p_elec_price, "cooling_energy_per_kg": p_elec_usage
        }
        config_fields = set(FarmConfig.__dataclass_fields__.keys())

        for sc_name, sc_params in active_scenarios.items():
            # Merge base config with scenario overrides
            run_kwargs = base_kwargs.copy()
            run_kwargs.update(sc_params)
            
            # Apply Labor Override
            if labor_override == "Vše ZAPNUTO":
                run_kwargs["include_labor_cost"] = True
            elif labor_override == "Vše VYPNUTO":
                run_kwargs["include_labor_cost"] = False
            
            # Normalize legacy scenario key (market_local_limit -> market_quota_kg)
            if "market_local_limit" in run_kwargs:
                run_kwargs["market_quota_kg"] = run_kwargs.get("market_quota_kg", run_kwargs["market_local_limit"])
                run_kwargs.pop("market_local_limit", None)
            
            # Remove any unexpected keys before FarmConfig(**kwargs)
            run_kwargs = {k: v for k, v in run_kwargs.items() if k in config_fields}
            
            for i in range(n_runs):
                # Random seed for each run
                # Pro každý běh nastavíme unikátní seed, ale konzistentní napříč scénáři.
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
                        elif key == "market_quota_kg":
                            current_run_kwargs[key] = current_run_kwargs[key] * factor
                            sens_log[label] = current_run_kwargs[key]
                        else:
                            current_run_kwargs[key] *= factor
                            sens_log[label] = current_run_kwargs[key]
                
                # RE-SEED: Zajistíme, že stochastika modelu (počasí, ceny) bude identická
                # pro daný Seed, bez ohledu na to, zda jsme "spotřebovali" náhodu pro citlivostní analýzu.
                np.random.seed(current_seed)
                
                # Create config object
                mc_cfg = FarmConfig(**current_run_kwargs)
                
                # Spuštění modelu
                mc_model = FarmModel(mc_cfg)
                mc_df = mc_model.run()
                
                # --- 1. RUN SUMMARY (Agregace za celý běh) ---
                profit = mc_df["Cash"].iloc[-1] - mc_cfg.capital
                is_bankrupt = 1 if mc_df["Cash"].iloc[-1] < 0 else 0
                
                total_labor = mc_df["Labor Hours"].sum()
                efficiency = profit / max(1.0, total_labor)
                
                summary_row = {
                    "Scénář": sc_name,
                    "Skupina": sc_name[0],
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
        # --- VIZUALIZACE VÝSLEDKŮ (ALTAIR) ---
        # --- VISUALIZATION ---
        df_summary = st.session_state['mc_results']['summary']
        df_quarterly = st.session_state['mc_results']['quarterly']
        
        # 1. SCENARIO DEFINITIONS TABLE
        st.subheader("Definice Scénářů")
        st.dataframe(pd.DataFrame(active_scenarios_pool).T)

        # 2. TIME SLICER & BOXPLOTS
        st.subheader("Porovnání v čase (Slicer)")
        
        # Get unique quarters sorted
        available_quarters = sorted(df_quarterly["Kvartál"].unique())
        selected_q = st.select_slider("Vyberte období pro srovnání:", options=available_quarters, value=available_quarters[-1])
        
        # Filter data for chart
        df_slice = df_quarterly[df_quarterly["Kvartál"] == selected_q]
        
        # Boxplot ukazuje rozdělení (medián, kvartily, extrémy).
        chart_profit = alt.Chart(df_slice).mark_boxplot().encode(
            x=alt.X("Scénář:N", title=None),
            y=alt.Y("Cash:Q", title=f"Hotovost v {selected_q} (Kč)"),
            color="Scénář:N",
            tooltip=["Scénář", "Cash", "BCS", "Animals"]
        ).properties(height=400, title=f"Rozdělení hotovosti ({selected_q})")
        st.altair_chart(chart_profit, use_container_width=True)
        
        # 2b. EFFICIENCY CHART
        st.subheader("Pracovní Efektivita (Zisk na hodinu)")
        chart_eff = alt.Chart(df_summary).mark_boxplot().encode(
            x=alt.X("Scénář:N", title=None),
            y=alt.Y("Efektivita (Kč/h):Q", title="Zisk na hodinu práce (Kč/h)"),
            color="Skupina:N",
            tooltip=["Scénář", "Efektivita (Kč/h)", "Zisk (Kč)", "Pracnost (h)"]
        ).properties(height=300)
        st.altair_chart(chart_eff, use_container_width=True)
        
        # 3. RISK CHART (X = Sheep Count)
        # Scatter plot (bublinový graf) pro porovnání rizika a zisku.
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
        st.subheader("Vývoj v čase")
        
        ts_view_mode = st.radio("Režim zobrazení", ["Všechny běhy (Detail)", "Pásmo spolehlivosti (Agregace)"], horizontal=True)
        
        if ts_view_mode == "Všechny běhy (Detail)":
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
            # Pásma spolehlivosti (Confidence Intervals)
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
            st.subheader("Citlivostní Analýza (Korelace)")
            
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
        st.subheader("Souhrnné Výsledky (Průměry)")
        st.dataframe(risk_agg.style.format({
            "Riziko_Bankrotu": "{:.1%}", 
            "Průměr_Zisk": "{:,.0f}", 
            "Průměr_Min_BCS": "{:.2f}"
        }), use_container_width=True)
        
        with st.expander("Surová Data (Kvartální export)"):
            st.markdown("Data obsahují záznam pro každý Seed a každý Kvartál.")
            st.dataframe(df_quarterly)
            st.download_button("Stáhnout CSV (Quarterly)", df_quarterly.to_csv(index=False).encode('utf-8'), "monte_carlo_quarterly.csv")
            
        with st.expander("Surová Data (Souhrn běhu)"):
            st.markdown("Data obsahují jeden řádek pro každý Seed (finální výsledky).")
            st.dataframe(df_summary)
            st.download_button("Stáhnout CSV (Summary)", df_summary.to_csv(index=False).encode('utf-8'), "monte_carlo_summary.csv")
            
    st.stop() # Stop execution here so standard dashboard doesn't render below

# --- SPUŠTĚNÍ JEDNOTLIVÉ SIMULACE (STANDARDNÍ REŽIM) ---
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
    market_quota_kg=m_quota_kg,
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
    shock_prob_daily=shock_p,
    
    # Sector 8
    enable_freezing=use_freezing,
    freezer_capacity_kg=p_freezer_cap,
    freezer_capex=p_freezer_capex,
    electricity_price=p_elec_price,
    cooling_energy_per_kg=p_elec_usage
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

# --- 3. SKLADOVÉ ZÁSOBY (Seno & Maso) ---
st.subheader("Skladové zásoby (Seno & Maso)")

col_hay, col_meat = st.columns(2)

with col_hay:
    st.markdown("**Seno (Balíky)**")
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
    ).properties(height=300)
    st.altair_chart(hay_chart, use_container_width=True)

with col_meat:
    st.markdown("**Prodeje Masa (kg)**")
    base = alt.Chart(df.reset_index()).encode(x=alt.X('Date:T', title='Datum'))
    
    fresh = base.mark_bar(color='#e74c3c').encode(
        y=alt.Y('Sold_Fresh_Kg:Q', title='Čerstvé (kg)', axis=alt.Axis(titleColor='#e74c3c')),
        tooltip=[alt.Tooltip('Date:T'), alt.Tooltip('Sold_Fresh_Kg:Q', title='Čerstvé', format='.1f')]
    )
    
    frozen = base.mark_line(color='#3498db', strokeWidth=2).encode(
        y=alt.Y('Sold_Frozen_Kg:Q', title='Mražené (kg)', axis=alt.Axis(titleColor='#3498db')),
        tooltip=[alt.Tooltip('Date:T'), alt.Tooltip('Sold_Frozen_Kg:Q', title='Mražené', format='.1f')]
    )
    
    meat_chart = alt.layer(fresh, frozen).resolve_scale(y='independent').properties(height=300)
    st.altair_chart(meat_chart, use_container_width=True)

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
    source = pd.DataFrame({
        'Náklad': ['Krmivo', 'Veterina+Seč', 'Administrativa', 'Režie', 'Práce', 'Šoky'],
        'Podíl': [df["Exp_Feed"].sum(), df["Exp_Variable"].sum(), df["Exp_Admin"].sum(), df["Exp_Overhead"].sum(), df["Exp_Labor"].sum(), df["Exp_Shock"].sum()]
    })

    pie_chart = alt.Chart(source).mark_arc(innerRadius=50).encode(
        theta=alt.Theta(field="Podíl", type="quantitative", stack=True),
        color=alt.Color(field="Náklad", type="nominal", scale=alt.Scale(
            domain=['Krmivo', 'Veterina+Seč', 'Administrativa', 'Režie', 'Práce', 'Šoky'],
            range=['#e67e22', '#9b59b6', '#7f8c8d', '#3498db', '#1abc9c', '#e74c3c']
        )),
        tooltip=['Náklad', alt.Tooltip('Podíl:Q', format=',.0f')]
    ).properties(title="Nákladová struktura")

    st.altair_chart(pie_chart, use_container_width=True)

# --- 5. SEASONAL ANALYSIS ---
st.subheader("Sezónní analýza")

col_season, col_price = st.columns(2)

with col_season:
    st.markdown("**Průměrný Denní Cashflow po Měsících**")
    
    df_month = df.copy()
    df_month["Month"] = df_month.index.month
    df_month["Daily_Flow"] = df_month["Income"] - (df_month["Exp_Feed"] + df_month["Exp_Variable"] + df_month["Exp_Admin"] + df_month["Exp_Overhead"] + df_month["Exp_Labor"] + df_month["Exp_Shock"])
    
    seasonal = df_month.groupby("Month")["Daily_Flow"].mean()
    seasonal_df = seasonal.reset_index()
    
    chart_seas = alt.Chart(seasonal_df).mark_bar().encode(
        x=alt.X("Month:O", title="Měsíc"),
        y=alt.Y("Daily_Flow:Q", title="Denní Tok (Kč)"),
        color=alt.condition(
            alt.datum.Daily_Flow > 0,
            alt.value("#2ecc71"),  # Zelená pro zisk
            alt.value("#e74c3c")   # Červená pro ztrátu
        ),
        tooltip=["Month", alt.Tooltip("Daily_Flow", format=",.0f")]
    ).properties(height=300)
    
    st.altair_chart(chart_seas, use_container_width=True)

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
    days_grazing = model.feed_log.get("Pastva", 0) + model.feed_log.get("Pastva (Bez příkrmu)", 0) + model.feed_log.get("Pastva + Hlad", 0)
    days_stored = model.feed_log.get("Seno", 0) + model.feed_log.get("Pastva + Seno", 0)
    days_market = model.feed_log.get("Nákup", 0) + model.feed_log.get("Hladovění (Čekání)", 0) + model.feed_log.get("Hladovění (Bez sena)", 0)
    days_stored += model.feed_log.get("Seno (Ochrana)", 0)
    
    total_days = sum(model.feed_log.values())
    
    grazing_pct = (days_grazing / total_days * 100) if total_days > 0 else 0
    stored_pct = (days_stored / total_days * 100) if total_days > 0 else 0
    market_pct = (days_market / total_days * 100) if total_days > 0 else 0
    
    feed_df = pd.DataFrame({
        "Zdroj": ["Pastva", "Seno", "Nákup"],
        "Dny": [days_grazing, days_stored, days_market],
        "Procento": [grazing_pct, stored_pct, market_pct],
        "Color": ["#2ecc71", "#f39c12", "#e74c3c"]
    })
    chart_feed = alt.Chart(feed_df).mark_bar().encode(
        x=alt.X("Dny:Q", title="Dny v roce"),
        y=alt.Y("Zdroj:N", sort=["Pastva", "Seno", "Nákup"], title=None),
        color=alt.Color("Color:N", scale=None),
        tooltip=["Zdroj", "Dny", alt.Tooltip("Procento", format=".1f")]
    ).properties(height=200)
    
    text_feed = chart_feed.mark_text(
        align='left',
        baseline='middle',
        dx=3,
        color='white'
    ).encode(
        text=alt.Text("Dny:Q", format=".0f")
    )
    
    st.altair_chart(chart_feed + text_feed, use_container_width=True)
    
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
st.subheader("🌱 Zdraví Pastviny (Ekologická smyčka)")

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
st.subheader("📉 Administrativní Zátěž (Neefektivita z rozsahu)")

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

with st.expander("📜 Deník Farmáře (Události)", expanded=False):
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
# POUŽITÍ PŘESNÝCH SLOUPCŮ Z MODELU (Inc_Meat, Inc_Subsidy, atd.)
total_meat_income = df["Inc_Meat"].sum()
total_hay_income = df["Inc_Hay"].sum()
total_subsidy_income = df["Inc_Subsidy"].sum()
total_expenses = df["Exp_Feed"].sum() + df["Exp_Variable"].sum() + df["Exp_Admin"].sum() + df["Exp_Overhead"].sum() + df["Exp_Labor"].sum() + df["Exp_Shock"].sum()

model_feed = df["Exp_Feed"].sum() / (avg_ewes * years)

# Rozdělení nákladů
model_vet_services = (df["Exp_Vet"].sum() + df["Exp_Shearing"].sum() + df["Exp_RamPurchase"].sum()) / (avg_ewes * years)
model_overhead_admin = (df["Exp_Overhead"].sum() + df["Exp_Admin"].sum() + df["Exp_Labor"].sum()) / (avg_ewes * years)
model_machinery_ops = (df["Exp_Mow"].sum() + df["Exp_Machinery"].sum() + df["Exp_Shock"].sum()) / (avg_ewes * years)

# Meat Income
model_meat = total_meat_income / (avg_ewes * years)

# Zisk bez dotací (Operational Profit)
# (Tržby za maso + seno - Náklady) / (ovce * roky)
model_profit_no_sub = (total_meat_income + total_hay_income - total_expenses) / (avg_ewes * years)

# Odchov (použití existujícího sloupce Lambs)
avg_lamb_peak = df[df.index.month == 6]["Lambs"].mean()
model_rearing = avg_lamb_peak / avg_ewes if avg_ewes > 0 else 0

# Subsidy dependence
total_income = df["Income"].sum()
model_subsidy_dep = (total_subsidy_income / total_income * 100) if total_income > 0 else 0

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