# Streamlit app wrapping your existing processing code.
# Keeps your original code structure and adds:
# - File upload / local path input
# - Interactive plotting of signals and peaks
# - Download of processed Excel result
#
# Usage:
#  - pip install -r requirements.txt
#  - streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import re
import os
import tempfile
from io import BytesIO, StringIO
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_prominences

st.set_page_config(layout="wide", page_title="Analyse Stimulation - GUI")

st.title("Analyse pre_suc")
st.markdown("Chargez un fichier Excel ou texte contenant les signaux. L'application gardera votre code tel quel et affichera les courbes, détectera les peaks, "
            "et proposera le téléchargement du fichier résultat.")

# --------------------------
# Input: upload or local path
# --------------------------
uploaded_file = st.file_uploader("Téléversez un fichier (Excel .xlsx/.xls)", type=["xlsx","xls","csv","txt"])
local_path = st.text_input("Ou indiquez un chemin local (laisser vide si vous avez téléversé un fichier) :", "")

# helper to persist a file-like to temp path when uploaded
def save_uploaded_to_temp(uploaded):
    if uploaded is None:
        return None
    suffix = os.path.splitext(uploaded.name)[1]
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded.getbuffer())
    tmp.flush()
    tmp.close()
    return tmp.name

# --------------------------
# Your original functions (kept as-is, small adaptions for file-like support)
# --------------------------
def is_data_line(line, min_numeric=4):
    parts = re.split(r'[\t,; ]+', line.strip())
    numeric_count = sum(bool(re.match(r'^-?\d+(\.\d+)?(E[+-]?\d+)?$', p, re.IGNORECASE)) for p in parts)
    return numeric_count >= min_numeric

def read_clean_text_dataframe(filepath, sep=None, min_numeric=4, header=None):
    """
    Lit un fichier texte, retire les lignes parasites, et retourne un DataFrame pandas.
    filepath may be a path string.
    """
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    # Garde seulement les lignes de données
    data_lines = [line for line in lines if is_data_line(line, min_numeric)]
    if not data_lines:
        return pd.DataFrame()
    # Auto-détection du séparateur si non spécifié
    if sep is None:
        sample = data_lines[0]
        if "\t" in sample:
            sep = "\t"
        elif ";" in sample:
            sep = ";"
        elif "," in sample:
            sep = ","
        else:
            sep = r"\s+"  # Espace
    data_str = "".join(data_lines)
    df = pd.read_csv(StringIO(data_str), sep=sep, header=header)
    return df

def read_clean_excel_dataframe(filepath, min_numeric=4, header=0):
    """
    Lit un fichier Excel, retire les lignes parasites, et retourne un DataFrame pandas.
    """
    df_raw = pd.read_excel(filepath, header=header)
    def is_row_data(row):
        numeric_count = sum(pd.to_numeric(row, errors="coerce").notnull())
        return numeric_count >= min_numeric
    mask = df_raw.apply(is_row_data, axis=1)
    return df_raw[mask].reset_index(drop=True)

def auto_clean_dataframe(filepath, min_numeric=4, header=None):
    """
    Nettoie automatiquement le fichier (texte/Excel) et retourne un DataFrame propre.
    """
    ext = os.path.splitext(filepath)[1].lower()
    if ext in [".xlsx", ".xls"]:
        # Header=0 pour Excel par défaut
        return read_clean_excel_dataframe(filepath, min_numeric=min_numeric, header=header if header is not None else 0)
    else:
        return read_clean_text_dataframe(filepath, min_numeric=min_numeric, header=header)

# --------------------------
# Processing block (kept logic similar to your provided script)
# --------------------------
def process_file(chemin_fichier):
    # Lecture
    df = auto_clean_dataframe(chemin_fichier)

    # If the read produced empty DF, try a raw read
    if df.empty:
        try:
            df = pd.read_excel(chemin_fichier)
        except Exception:
            pass

    # Try to set expected columns if possible (user original mapping)
    # If there are more/fewer columns, attempt to guess columns by name
    try:
        df.columns = ['seconde','NP/NS pressure','Finger Pressure','Mean Arterial','Heart Rate']
    except Exception:
        # best-effort mapping using substrings
        col_map = {}
        cols = list(df.columns)
        for c in cols:
            lower = str(c).lower()
            if 'sec' in lower or 'time' in lower:
                col_map[c] = 'seconde'
            elif 'np' in lower or 'ns' in lower:
                col_map[c] = 'NP/NS pressure'
            elif 'finger' in lower or 'finger pressure' in lower:
                col_map[c] = 'Finger Pressure'
            elif 'mean' in lower and ('arter' in lower or 'map' in lower):
                col_map[c] = 'Mean Arterial'
            elif 'heart' in lower or 'hr' in lower:
                col_map[c] = 'Heart Rate'
        df = df.rename(columns=col_map)
        # If required columns still missing, leave as read and proceed (errors will be handled)

    df_1 = df.copy()

    col= ['seconde','NP/NS pressure','Finger Pressure','Mean Arterial','Heart Rate']
    for i in col:
        if i in df_1.columns:
            df_1[i]=pd.to_numeric(df_1[i],errors='coerce')

    # Add marker column
    df_1['marquer'] = df_1.get('marquer','')

    # detect events (start when NP/NS pressure <= -5 or >= 5)
    if 'NP/NS pressure' in df_1.columns:
        mask = df_1['NP/NS pressure'] <= -5
        mask1 = df_1['NP/NS pressure'] >= 5
        start_of_sequence = mask & (~mask.shift(1, fill_value=False))
        start_of_sequence1 = mask1 & (~mask1.shift(1, fill_value=False))
        df_1.loc[start_of_sequence, 'marquer'] = 'succion'
        df_1.loc[start_of_sequence1, 'marquer'] = 'pression'

    # moyenne (5 secondes après le début)
    if 'moyenne' not in df_1.columns:
        df_1['moyenne'] = None

    evenements = ['pression', 'succion']
    rows_to_add = []
    for index, row in df_1[df_1['marquer'].isin(evenements)].iterrows():
        t0 = row.get('seconde', None)
        if pd.isna(t0):
            continue
        t1 = t0 + 5
        interval = df_1[(df_1['seconde'] >= t0) & (df_1['seconde'] <= t1)]
        moyenne = None
        if 'NP/NS pressure' in interval.columns:
            moyenne = interval['NP/NS pressure'].mean()
        df_1.at[index, 'moyenne'] = moyenne
        new_row = row.copy()
        new_row['seconde'] = t1
        new_row['marquer'] = 'fin'
        new_row['moyenne'] = None
        rows_to_add.append(new_row)
    if rows_to_add:
        df_1 = pd.concat([df_1, pd.DataFrame(rows_to_add)], ignore_index=True)
        df_1 = df_1.sort_values(by='seconde').reset_index(drop=True)

    # Compute min/max HR and MAP between event and next 'fin'
    for colname in ['min_heart_rate', 'min_mean_arterial', 'max_heart_rate', 'max_mean_arterial']:
        if colname not in df_1.columns:
            df_1[colname] = None

    for event in ['pression', 'succion']:
        event_indices = df_1[df_1['marquer'] == event].index
        for idx in event_indices:
            t_start = df_1.at[idx, 'seconde']
            fin_rows = df_1[(df_1['marquer'] == 'fin') & (df_1['seconde'] > t_start)]
            if not fin_rows.empty:
                fin_index = fin_rows.index[0]
                t_end = df_1.at[fin_index, 'seconde']
                interval = df_1[(df_1['seconde'] > t_start) & (df_1['seconde'] < t_end)]
                if not interval.empty:
                    if 'Heart Rate' in interval.columns:
                        min_hr = interval['Heart Rate'].min()
                        max_hr = interval['Heart Rate'].max()
                        df_1.at[fin_index, 'min_heart_rate'] = min_hr
                        df_1.at[fin_index, 'max_heart_rate'] = max_hr
                    if 'Mean Arterial' in interval.columns:
                        min_map = interval['Mean Arterial'].min()
                        max_map = interval['Mean Arterial'].max()
                        df_1.at[fin_index, 'min_mean_arterial'] = min_map
                        df_1.at[fin_index, 'max_mean_arterial'] = max_map

    # time to peaks (min/max) relative to event start
    for colname in ['time_to_min_hr', 'time_to_min_map', 'time_to_max_hr', 'time_to_max_map']:
        if colname not in df_1.columns:
            df_1[colname] = None

    for event in ['pression', 'succion']:
        event_indices = df_1[df_1['marquer'].astype(str).str.strip().str.lower() == event].index
        for i, event_index in enumerate(event_indices):
            t_start = df_1.at[event_index, 'seconde']
            if i + 1 < len(event_indices):
                t_end = df_1.at[event_indices[i + 1], 'seconde']
            else:
                t_end = df_1['seconde'].max()
            interval = df_1[(df_1['seconde'] >= t_start) & (df_1['seconde'] <= t_end)]
            if not interval.empty:
                if 'Heart Rate' in df_1.columns:
                    idx_min_hr = interval['Heart Rate'].idxmin()
                    t_min_hr = df_1.at[idx_min_hr, 'seconde']
                    delta_min_hr = t_min_hr - t_start
                    idx_max_hr = interval['Heart Rate'].idxmax()
                    t_max_hr = df_1.at[idx_max_hr, 'seconde']
                    delta_max_hr = t_max_hr - t_start
                    df_1.at[event_index, 'time_to_min_hr'] = delta_min_hr
                    df_1.at[event_index, 'time_to_max_hr'] = delta_max_hr
                if 'Mean Arterial' in df_1.columns:
                    idx_min_map = interval['Mean Arterial'].idxmin()
                    t_min_map = df_1.at[idx_min_map, 'seconde']
                    delta_min_map = t_min_map - t_start
                    idx_max_map = interval['Mean Arterial'].idxmax()
                    t_max_map = df_1.at[idx_max_map, 'seconde']
                    delta_max_map = t_max_map - t_start
                    df_1.at[event_index, 'time_to_min_map'] = delta_min_map
                    df_1.at[event_index, 'time_to_max_map'] = delta_max_map

    # Peak detection in 'Finger Pressure' before each event (kept from your script)
    col_pressure = "Finger Pressure"
    col_mark = "marquer"
    col_time = "seconde"

    window_seconds = 3
    max_distance_from_event = 5
    min_sep_seconds = 0.03
    max_peaks_per_zone = 3
    selection_metric = 'proximity_then_prominence'
    cluster_mode = True

    if col_mark not in df_1.columns:
        df_1[col_mark] = ""

    event_mask = df_1[col_mark].astype(str).str.contains("pression|succion", case=False, na=False)
    event_indices = df_1.index[event_mask].tolist()
    fin_indices = df_1.index[df_1[col_mark].astype(str).str.lower() == "fin"].tolist()

    zones = []
    if event_indices:
        zones.append((0, event_indices[0]))
    else:
        zones.append((0, len(df_1) - 1))
    for fin_idx in fin_indices:
        next_events = [idx for idx in event_indices if idx > fin_idx]
        if next_events:
            zones.append((fin_idx + 1, next_events[0]))
        else:
            zones.append((fin_idx + 1, len(df_1) - 1))

    # Ensure time numeric
    df_1[col_time] = pd.to_numeric(df_1[col_time], errors='coerce')

    peaks_info = []  # For UI plotting: store selections per zone

    for izone, (start_idx, end_idx) in enumerate(zones):
        if start_idx > end_idx:
            continue
        event_idx = end_idx
        try:
            t_event = float(df_1.at[event_idx, col_time])
        except Exception:
            continue

        if window_seconds is None:
            t_left = float(df_1.at[start_idx, col_time])
        else:
            t_left = t_event - float(window_seconds)
            t_left = max(t_left, float(df_1.at[start_idx, col_time]))

        mask_zone = (df_1[col_time] >= t_left) & (df_1[col_time] < t_event)
        mask_zone &= (df_1.index >= start_idx) & (df_1.index <= end_idx)
        df_zone = df_1.loc[mask_zone].copy()

        if df_zone.empty:
            continue

        vals = pd.to_numeric(df_zone.get(col_pressure, pd.Series(dtype=float)), errors='coerce').dropna()
        if vals.empty:
            continue

        peaks_rel, _ = find_peaks(vals.values)
        if len(peaks_rel) == 0:
            continue

        peaks_global = vals.index[peaks_rel]
        heights = vals.values[peaks_rel]
        proms = peak_prominences(vals.values, peaks_rel)[0]

        cand = pd.DataFrame({
            'idx': peaks_global,
            'time': df_1.loc[peaks_global, col_time].values,
            'height': heights,
            'prominence': proms
        })
        cand['distance'] = t_event - cand['time']
        cand = cand[cand['distance'] >= 0]

        cand_close = cand[cand['distance'] <= max_distance_from_event]
        if cand_close.empty:
            cand_close = cand.copy()

        if selection_metric == 'height':
            cand_sorted = cand_close.sort_values('height', ascending=False)
        elif selection_metric == 'prominence':
            cand_sorted = cand_close.sort_values('prominence', ascending=False)
        else:
            cand_sorted = cand_close.sort_values(['distance', 'prominence'], ascending=[True, False])

        selected = []
        for _, crow in cand_sorted.iterrows():
            t_cand = float(crow['time'])
            idx_cand = int(crow['idx'])
            if cluster_mode:
                selected.append({'idx': idx_cand, 'time': t_cand})
            else:
                if all(abs(t_cand - s['time']) >= min_sep_seconds for s in selected):
                    selected.append({'idx': idx_cand, 'time': t_cand})
            if len(selected) >= max_peaks_per_zone:
                break

        if len(selected) < max_peaks_per_zone:
            remaining = cand[~cand['idx'].isin([s['idx'] for s in selected])]
            for _, row in remaining.iterrows():
                selected.append({'idx': int(row['idx']), 'time': float(row['time'])})
                if len(selected) >= max_peaks_per_zone:
                    break

        if not selected:
            continue

        selected_sorted = sorted(selected, key=lambda x: x['time'])
        for i, s in enumerate(selected_sorted):
            lbl = s['idx']
            mark = f"peak{i+1}"
            existing = df_1.at[lbl, col_mark]
            if pd.isna(existing) or str(existing).strip() == "":
                df_1.at[lbl, col_mark] = mark
            else:
                existing_str = str(existing)
                if mark not in existing_str:
                    df_1.at[lbl, col_mark] = f"{existing_str}; {mark}"

        peaks_info.append({
            "zone": izone+1,
            "event_index": int(event_idx),
            "t_event": t_event,
            "selected": selected_sorted,
            "t_left": t_left,
            "t_right": t_event,
            "df_zone": df_zone
        })

    # Compute mean HR and MAP between peak1 and peak3 if they exist
    def find_column_by_candidates(df, candidates):
        cols_lower = {c.lower(): c for c in df.columns}
        for cand in candidates:
            for col_lower, col_orig in cols_lower.items():
                if cand == col_lower or cand in col_lower:
                    return col_orig
        return None

    hr_candidates = ["heart rate", "hr", "heart_rate", "heartrate"]
    map_candidates = ["mean arterial", "map", "mean_arterial", "meanarterial", "map/hr", "map_hr", "map / hr"]

    try:
        hr_col = find_column_by_candidates(df_1, hr_candidates)
        map_col = find_column_by_candidates(df_1, map_candidates)
    except Exception:
        hr_col = None
        map_col = None

    if hr_col is not None and "mean_heart_rate" not in df_1.columns:
        df_1["mean_heart_rate"] = np.nan
    if map_col is not None and "mean_mean_arterial" not in df_1.columns:
        df_1["mean_mean_arterial"] = np.nan

    idx_peak1_all = df_1.index[df_1['marquer'].astype(str).str.contains(r"\bpeak1\b", case=False, na=False)].tolist()
    idx_peak3_all = df_1.index[df_1['marquer'].astype(str).str.contains(r"\bpeak3\b", case=False, na=False)].tolist()

    for idx1 in idx_peak1_all:
        idx3_candidates = [i for i in idx_peak3_all if i > idx1]
        if not idx3_candidates:
            continue
        idx3 = idx3_candidates[0]
        pos1 = df_1.index.get_loc(idx1)
        pos3 = df_1.index.get_loc(idx3)
        start_pos, end_pos = (pos1, pos3) if pos1 <= pos3 else (pos3, pos1)
        segment = df_1.iloc[start_pos:end_pos+1].copy()
        if hr_col:
            segment_hr = pd.to_numeric(segment[hr_col], errors='coerce')
            mean_hr = segment_hr.mean()
            df_1.loc[idx3, "mean_heart_rate"] = mean_hr
        if map_col:
            segment_map = pd.to_numeric(segment[map_col], errors='coerce')
            mean_map = segment_map.mean()
            df_1.loc[idx3, "mean_mean_arterial"] = mean_map

    # Final output path (in memory)
    return df_1, peaks_info

# --------------------------
# UI: Load and process
# --------------------------
process_btn = st.button("Processer le fichier")

if process_btn:
    chemin = None
    temp_path = None
    if uploaded_file is not None:
        temp_path = save_uploaded_to_temp(uploaded_file)
        chemin = temp_path
    elif local_path:
        if os.path.exists(local_path):
            chemin = local_path
        else:
            st.error("Le chemin local fourni n'existe pas.")
            chemin = None
    else:
        st.error("Aucun fichier fourni. Téléversez un fichier ou indiquez un chemin local.")
        chemin = None

    if chemin:
        with st.spinner("Traitement en cours..."):
            try:
                df_result, peaks_info = process_file(chemin)
            except Exception as e:
                st.error(f"Erreur lors du traitement: {e}")
                st.stop()

        st.success("Traitement terminé.")
        st.subheader("Aperçu des données (quelques lignes)")
        st.dataframe(df_result.head(200))

        # Plot overview signals
        st.subheader("Tracé des signaux")
        fig, ax = plt.subplots(2, 1, figsize=(12,6), sharex=True)
        if 'seconde' in df_result.columns and 'Finger Pressure' in df_result.columns:
            ax[0].plot(df_result['seconde'], df_result['Finger Pressure'], label='Finger Pressure', color='blue')
        if 'seconde' in df_result.columns and 'NP/NS pressure' in df_result.columns:
            ax[0].plot(df_result['seconde'], df_result['NP/NS pressure'], label='NP/NS pressure', color='purple', alpha=0.7)
        ax[0].legend()
        ax[0].set_ylabel("Pressure")

        if 'seconde' in df_result.columns and 'Mean Arterial' in df_result.columns:
            ax[1].plot(df_result['seconde'], df_result['Mean Arterial'], label='Mean Arterial', color='green')
        if 'seconde' in df_result.columns and 'Heart Rate' in df_result.columns:
            ax[1].plot(df_result['seconde'], df_result['Heart Rate'], label='Heart Rate', color='red')
        ax[1].legend()
        ax[1].set_ylabel("MAP / HR")
        ax[1].set_xlabel("Temps (s)")
        st.pyplot(fig)

        # Show detected events and peaks with zooms
        st.subheader("Zones de peaks détectés (zoom avant événement)")
        if peaks_info:
            for pinfo in peaks_info:
                df_zone = pinfo['df_zone']
                selected = pinfo['selected']
                t_event = pinfo['t_event']
                zona = pinfo['zone']
                figz, axz = plt.subplots(1,1, figsize=(10,3))
                if not df_zone.empty and 'seconde' in df_zone.columns and 'Finger Pressure' in df_zone.columns:
                    axz.plot(df_zone['seconde'], df_zone['Finger Pressure'], label='Finger Pressure')
                    # all detected peaks (global indices may be outside df_zone index scope but markers via df_result)
                    # mark selected peaks:
                    for s in selected:
                        idx = s['idx']
                        t = s['time']
                        # fetch value from df_result to ensure correctness
                        try:
                            val = float(df_result.at[idx, 'Finger Pressure'])
                            axz.scatter([t], [val], color='orange', s=80, zorder=5)
                            axz.text(t, val + (abs(val)*0.02 if val!=0 else 0.1), f"peak@{round(t,3)}", ha='center')
                        except Exception:
                            pass
                    axz.axvline(t_event, color='green', linestyle='--', label='Événement')
                    axz.set_title(f"Zone {zona} — avant événement t={t_event}")
                    axz.set_xlabel("Temps (s)")
                    axz.set_ylabel("Finger Pressure")
                    axz.legend()
                    st.pyplot(figz)
        else:
            st.write("Aucun peak détecté selon les règles appliquées.")

        # Offer download
        st.subheader("Télécharger le fichier résultat")
        towrite = BytesIO()
        with pd.ExcelWriter(towrite, engine='openpyxl') as writer:
            df_result.to_excel(writer, index=False, sheet_name='resultats')
        towrite.seek(0)
        st.download_button(label="Télécharger le fichier Excel résultat",data=towrite,file_name="resultat-analysis.xlsx",mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        # Cleanup temp file if any
        if temp_path:
            try:
                os.unlink(temp_path)
            except Exception:
                pass

    else:
        st.error("Impossible de récupérer le fichier pour traitement.")
else:
    st.info("Chargez un fichier et cliquez sur 'Processer le fichier' pour démarrer.")