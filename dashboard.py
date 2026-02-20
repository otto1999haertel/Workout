import numpy as np
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import scipy.stats as stats
import time 
import streamlit as st
import plotly.graph_objects as go
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Daten laden
@st.cache_data
def load_data():
    df = pd.read_csv('allRuns.csv', delimiter=';')
    df['start_datetime'] = pd.to_datetime(df['start_time'], format='mixed')
    df['date'] = df['start_datetime'].dt.date
    df['year'] = df['start_datetime'].dt.year
    df['month'] = df['start_datetime'].dt.month
    df['pace_min_km'] = df['avg_pace_sec'] / 60
    return df

df = load_data()

print("Dashboard starting...")
# Titel
st.title("🏃 Lauf Dashboard")
st.markdown(f"**{len(df)} Läufe** von {df['date'].min()} bis {df['date'].max()}")

# Kennzahlen
col1, col2, col3, col4 = st.columns(4)

with col1:
    total_km = df['distance_km'].sum()
    st.metric("Gesamt Kilometer", f"{total_km:.1f} km")

with col2:
    avg_pace = df['avg_pace_sec'].mean()
    st.metric("Ø Pace", f"{int(avg_pace//60)}:{int(avg_pace%60):02d} min/km")

with col3:
    avg_hr = df['avg_hf'].mean()
    st.metric("Ø Herzfrequenz", f"{avg_hr:.0f} bpm")

with col4:
    avg_distance = df['distance_km'].mean()
    st.metric("Ø Distanz", f"{avg_distance:.1f} km")

# Filter
st.sidebar.header("Filter")
years = sorted(df['year'].unique(),reverse=True)
selected_year = st.sidebar.selectbox("Jahr", ['Alle'] + years)

if selected_year != 'Alle':
    df_filtered = df[df['year'] == selected_year]
else:
    df_filtered = df

# Visualisierungen
if selected_year != 'Alle':
    st.header(f"📊 Analysen {selected_year} ({len(df_filtered)} Läufe)")
else:
    st.header(f"📊 Analysen Gesamt ({len(df_filtered)} Läufe)")


# 1. Kilometer pro Monat
avg_km_month=0
if selected_year != 'Alle':
    avg_km_month = round(df_filtered.groupby(['year', 'month'])['distance_km'].sum().mean(),2)
else:
    avg_km_month = round(df_filtered['distance_km'].mean(),2)

st.markdown(f"Durchschnittliche Kilometer pro Monat {avg_km_month} km")
df_monthly = df_filtered.groupby(['year', 'month'])['distance_km'].sum().reset_index()
df_monthly['month_year'] = pd.to_datetime(df_monthly[['year', 'month']].assign(day=1))

# Trendlinie berechnen
x_numeric = (df_monthly['month_year'] - df_monthly['month_year'].min()).dt.days
degree = 5  # Erhöhe für mehr Kurven (2-5 empfohlen)
coefficients = np.polyfit(x_numeric, df_monthly['distance_km'], degree)
polynomial = np.poly1d(coefficients)
df_monthly['trend'] = polynomial(x_numeric)

# Chart erstellen
fig_monthly = go.Figure()

fig_monthly.add_trace(go.Bar(
    x=df_monthly['month_year'],
    y=df_monthly['distance_km'],
    name='Kilometer',
    marker_color='steelblue'
))

fig_monthly.add_trace(go.Scatter(
    x=df_monthly['month_year'],
    y=df_monthly['trend'],
    name='Trend',
    mode='lines',
    line=dict(color='red', width=2, dash='dash')
))

fig_monthly.update_layout(
    title='Kilometer pro Monat',
    xaxis_title='Monat',
    yaxis_title='Kilometer'
)

st.plotly_chart(fig_monthly, use_container_width=True)

# 2. Pace über Zeit mit HF (geglättet)
st.subheader("Pace-Entwicklung mit Herzfrequenz")

# Daten vorbereiten und sortieren
df_plot = df_filtered.dropna(subset=['avg_hf']).sort_values('start_datetime')

# Gleitenden Durchschnitt berechnen (z.B. über 5 Läufe)
window_size = 5
df_plot['pace_smooth'] = df_plot['pace_min_km'].rolling(window=window_size, min_periods=1).mean()
df_plot['avg_hf_smooth'] = df_plot['avg_hf'].rolling(window=window_size, min_periods=1).mean()
df_plot['max_hf_smooth'] = df_plot['max_hf'].rolling(window=window_size, min_periods=1).mean()

# Figure mit zwei Y-Achsen erstellen
fig_pace = go.Figure()

# Pace als geglättete Linie (linke Y-Achse)
fig_pace.add_trace(go.Scatter(
    x=df_plot['start_datetime'],
    y=df_plot['pace_smooth'],
    name='Pace (geglättet)',
    mode='lines',  # Nur Linien, keine Marker
    line=dict(color='blue', width=3),
    yaxis='y1'
))

# Optional: Originaldaten als schwache Punkte im Hintergrund
fig_pace.add_trace(go.Scatter(
    x=df_plot['start_datetime'],
    y=df_plot['pace_min_km'],
    name='Pace (Original)',
    mode='markers',
    marker=dict(color='lightblue', size=4, opacity=0.3),
    yaxis='y1',
    showlegend=False
))

# Durchschnittliche HF geglättet (rechte Y-Achse)
fig_pace.add_trace(go.Scatter(
    x=df_plot['start_datetime'],
    y=df_plot['avg_hf_smooth'],
    name='Ø HF (geglättet)',
    mode='lines',
    line=dict(color='red', width=3),
    yaxis='y2'
))

# Maximale HF geglättet (rechte Y-Achse)
fig_pace.add_trace(go.Scatter(
    x=df_plot['start_datetime'],
    y=df_plot['max_hf_smooth'],
    name='Max HF (geglättet)',
    mode='lines',
    line=dict(color='orange', width=2, dash='dash'),
    yaxis='y2'
))

# Layout mit zwei Y-Achsen
fig_pace.update_layout(
    title=f'Pace und Herzfrequenz über Zeit (geglättet über {window_size} Läufe)',
    xaxis=dict(title='Datum'),
    yaxis=dict(
        title=dict(text='Pace (min/km)', font=dict(color='blue')),
        side='left',
        tickfont=dict(color='blue')
    ),
    yaxis2=dict(
        title=dict(text='Herzfrequenz (bpm)', font=dict(color='red')),
        side='right',
        overlaying='y',
        tickfont=dict(color='red')
    ),
    hovermode='x unified',
    legend=dict(x=0.01, y=0.99)
)

st.plotly_chart(fig_pace, use_container_width=True)

# 3. Herzfrequenz vs Pace
st.subheader("❤️ Herzfrequenz-Analyse: Pace × Temperatur")

df_hr = df_filtered.dropna(subset=['avg_hf', 'temp_c', 'pace_min_km'])

if len(df_hr) < 5:
    st.info("Zu wenig Daten für eine Analyse (mind. 5 Läufe benötigt)")
else:
    # ── Model trainieren ──────────────────────────────────────────────────────
    X = df_hr[['pace_min_km', 'temp_c']].values
    y = df_hr['avg_hf'].values

    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X)
    model = LinearRegression().fit(X_poly, y)
    r2 = model.score(X_poly, y)

    # ── Prediction Grid (elevation_gain auf Median fixiert) ───────────────────
    GRID_SIZE = 60
    pace_range  = np.linspace(df_hr['pace_min_km'].min(), df_hr['pace_min_km'].max(), GRID_SIZE)
    temp_range  = np.linspace(df_hr['temp_c'].min(),      df_hr['temp_c'].max(),      GRID_SIZE)

    pace_grid, temp_grid = np.meshgrid(pace_range, temp_range)

    grid_poly = poly.transform(np.c_[pace_grid.ravel(), temp_grid.ravel()])
    hr_pred   = model.predict(grid_poly).reshape(pace_grid.shape)

    # ── Contour + Scatter Plot ────────────────────────────────────────────────
    fig = go.Figure()

    # Heatmap-Hintergrund
    fig.add_trace(go.Contour(
        x=pace_range,
        y=temp_range,
        z=hr_pred,
        colorscale='RdYlGn_r',
        contours=dict(
            showlabels=True,
            labelfont=dict(size=11, color='white'),
            start=int(hr_pred.min()),
            end=int(hr_pred.max()),
            size=5,
        ),
        colorbar=dict(title=dict(text='Pred. HF (bpm)', side='right')),
        opacity=0.85,
        name='Modell',
        hovertemplate='Pace: %{x:.1f} min/km<br>Temp: %{y:.1f}°C<br>Pred. HF: %{z:.0f} bpm<extra></extra>',
    ))

    # Echte Läufe als Scatter
    fig.add_trace(go.Scatter(
        x=df_hr['pace_min_km'],
        y=df_hr['temp_c'],
        mode='markers',
        marker=dict(
            color=df_hr['avg_hf'],
            colorscale='RdYlGn_r',
            cmin=hr_pred.min(),
            cmax=hr_pred.max(),
            size=10,
            line=dict(color='white', width=1.5),
            symbol='circle',
        ),
        name='Läufe',
        hovertemplate=(
            '<b>Lauf</b><br>'
            'Pace: %{x:.1f} min/km<br>'
            'Temp: %{y:.1f}°C<br>'
            'Tatsächliche HF: %{marker.color:.0f} bpm'
            '<extra></extra>'
        ),
        customdata=df_hr['avg_hf'],
    ))

    fig.update_layout(
        title=dict(
            text=(
                f'HF-Heatmap: Pace × Temperatur '
                f'<span style="font-size:13px;color:gray">'
                f'(R² = {r2:.2f}, n={len(df_hr)}, '
            ),
            font=dict(size=16),
        ),
        xaxis=dict(title='Pace (min/km)', tickformat='.1f'),
        yaxis=dict(title='Temperatur (°C)'),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        height=500,
        margin=dict(l=60, r=20, t=70, b=60),
    )

    st.plotly_chart(fig, use_container_width=True)

    # ── Hinweis auf Modellgüte ────────────────────────────────────────────────
    if r2 < 0.4:
        st.warning(f"⚠️ Modell-R² = {r2:.2f} – die Vorhersagequalität ist gering. Mehr Läufe verbessern das Modell.")
    elif r2 < 0.7:
        st.info(f"ℹ️ Modell-R² = {r2:.2f} – mittlere Vorhersagequalität.")
    else:
        st.success(f"✅ Modell-R² = {r2:.2f} – gute Vorhersagequalität.")

    st.divider()

    # ── Interaktiver Prediction-Slider ────────────────────────────────────────
    st.subheader("🔮 HF-Prognose für deinen nächsten Lauf")

    col1, col2 = st.columns(2)
    with col1:
        pred_pace = st.slider(
            "Geplante Pace (min/km)",
            min_value=float(round(df_hr['pace_min_km'].min(), 1)),
            max_value=float(round(df_hr['pace_min_km'].max(), 1)),
            value=float(round(df_hr['pace_min_km'].median(), 1)),
            step=0.1,
        )
    with col2:
        pred_temp = st.slider(
            "Erwartete Temperatur (°C)",
            min_value=float(round(df_hr['temp_c'].min())),
            max_value=float(round(df_hr['temp_c'].max())),
            value=float(round(df_hr['temp_c'].median())),
            step=1.0,
        )

    # Prediction mit allen 3 Features
    predicted_hr = model.predict(poly.transform([[pred_pace, pred_temp]]))[0]

    # Referenz: alle 3 Features auf Median
    ref_hr = model.predict(poly.transform([[
        df_hr['pace_min_km'].median(),
        df_hr['temp_c'].median(),
    ]]))[0]

    delta = predicted_hr - ref_hr

    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("Erwartete Herzfrequenz", f"{predicted_hr:.0f} bpm",
                delta=f"{delta:+.0f} vs. Median", delta_color="inverse")
    col_m2.metric("Pace",        f"{pred_pace:.1f} min/km")
    col_m3.metric("Temperatur",  f"{pred_temp:.0f} °C")

# 4. Temperatur-Einfluss
st.subheader("Temperatur-Einfluss auf Pace")
df_temp = df_filtered.dropna(subset=['temp_c'])
if len(df_temp) > 0:
    fig_temp = px.scatter(df_temp, x='temp_c', y='pace_min_km',
                         trendline='lowess',
                         title='Wie beeinflusst Temperatur dein Tempo?',
                         labels={'temp_c': 'Temperatur (°C)', 'pace_min_km': 'Pace (min/km)'})
    st.plotly_chart(fig_temp, use_container_width=True)
else:
    st.info("Keine Temperatur-Daten verfügbar")

# Tabelle mit letzten Läufen
st.header("📋 Letzte Läufe (ab 19.11.2025)")

start_date = '2025-11-19'
recent_filtered = df_filtered[df_filtered['start_time'] >= start_date]
recent = recent_filtered.sort_values('start_datetime', ascending=False).head(20)

if len(recent) == 0:
    st.warning(f"⚠️ Keine Läufe ab {start_date} gefunden!")
else:
    display_cols = ['date', 'distance_km', 'pace_min_km', 'avg_hf', 'temp_c']
    
    # Aufteilen in mit/ohne Temperatur
    with_temp = recent[recent['temp_c'].notna()]
    without_temp = recent[recent['temp_c'].isna()]

    df['pace_formatted'] = df['avg_pace_sec'].apply(
        lambda x: f"{int(x // 60)}:{int(x % 60):02d}" if pd.notna(x) else ""
    )
    
    # 1. Zeige Läufe MIT Temperatur (nur Ansicht)
    if len(with_temp) > 0:
        st.subheader("✅ Läufe mit Temperatur")
        st.dataframe(
            with_temp[display_cols].rename(columns={
                'date': 'Datum',
                'distance_km': 'Distanz (km)',
                'pace_formatted': 'Pace (min/km)',
                'avg_hf': 'HF (bpm)',
                'temp_c': 'Temp (°C)'
            }),
            use_container_width=True
        )
    
    # 2. Zeige Läufe OHNE Temperatur (editierbar)
    if len(without_temp) > 0:
        st.subheader("📝 Temperatur nachtragen")
        st.info(f"ℹ️ {len(without_temp)} Läufe ohne Temperatur-Daten")
        
        edited_df = st.data_editor(
            without_temp[display_cols].rename(columns={
                'date': 'Datum',
                'distance_km': 'Distanz (km)',
                'pace_formatted': 'Pace (min/km)',
                'avg_hf': 'HF (bpm)',
                'temp_c': 'Temp (°C)'
            }),
            use_container_width=True,
            num_rows="fixed",
            disabled=['Datum', 'Distanz (km)', 'Pace (min/km)', 'HF (bpm)'],
            column_config={
                "Temp (°C)": st.column_config.NumberColumn(
                    "Temp (°C)",
                    help="Temperatur in Grad Celsius",
                    min_value=-20,
                    max_value=45,
                    step=0.1,
                    format="%.1f°C"
                )
            },
            key="temp_editor"
        )
        
        if st.button("💾 Temperatur-Änderungen speichern", type="primary"):
            try:
                edited_df.columns = display_cols
                
                # Liste der Änderungen sammeln
                changes = []
                
                for idx in without_temp.index:
                    new_temp = edited_df.loc[idx, 'temp_c']
                    
                    # Nur übernehmen wenn jetzt gefüllt
                    if pd.notna(new_temp):
                        df.loc[idx, 'temp_c'] = new_temp
                        date_str = df.loc[idx, 'start_time'][:10]
                        changes.append(f"{date_str}: {new_temp}°C")
                
                if len(changes) == 0:
                    st.warning("⚠️ Keine Änderungen vorgenommen!")
                else:
                    # Nur die Original-14 Spalten speichern
                    original_cols = ['start_time', 'end_time', 'duration_sec', 'distance_km', 
                                    'avg_hf', 'max_hf', 'avg_pace_sec', 'avg_speed_kmh', 
                                    'avg_cadence', 'total_energy_kcal', 'elev_gain_m', 
                                    'elev_loss_m', 'temp_c', 'notes']
                    
                    df[original_cols].to_csv('allRuns.csv', sep=';', index=False)
                    
                    st.success(f"✅ {len(changes)} Temperatur-Einträge gespeichert:")
                    for change in changes:
                        st.write(f"  • {change}")
                    
                    time.sleep(1)
                    st.rerun()
                
            except Exception as e:
                st.error(f"❌ Fehler beim Speichern: {e}")
    else:
        st.success("✅ Alle Läufe haben bereits Temperatur-Daten!")