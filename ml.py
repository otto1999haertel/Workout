# Zu deinen Imports am Anfang hinzufügen:
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np

# Neue Funktion hinzufügen:
def __train_pace_model(data):
    """Trainiert ein ML-Modell zur Pace-Vorhersage"""
    
    # Nur Läufe mit vollständigen Daten verwenden
    df = data.dropna(subset=['distance_km', 'temp_c', 'elev_gain_m', 'avg_pace_sec'])
    
    if len(df) < 20:
        print(f"❌ Zu wenige Daten! Brauche mindestens 20 Läufe mit Temperatur, habe nur {len(df)}")
        return
    
    print(f"✓ Trainiere Modell mit {len(df)} Läufen\n")
    
    # Features (X) und Target (y)
    X = df[['distance_km', 'temp_c', 'elev_gain_m']]
    y = df['avg_pace_sec']
    
    # Train/Test Split (80% Training, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Modell trainieren
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Vorhersagen auf Testdaten
    y_pred = model.predict(X_test)
    
    # Modellqualität
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("=" * 50)
    print("MODELL-PERFORMANCE")
    print("=" * 50)
    print(f"R² Score: {r2:.3f} (1.0 = perfekt, 0 = nutzlos)")
    print()
    
    # Feature Importance (welcher Faktor ist am wichtigsten?)
    print("=" * 50)
    print("FEATURE IMPORTANCE")
    print("=" * 50)
    feature_names = ['Distanz (km)', 'Temperatur (°C)', 'Höhenmeter (m)']
    for name, coef in zip(feature_names, model.coef_):
        impact = "verlangsamt" if coef > 0 else "beschleunigt"
        print(f"{name:20s}: {coef:+.2f} sec/km ({impact})")
    print()
    
    # Beispiel-Vorhersagen
    print("=" * 50)
    print("BEISPIEL-VORHERSAGEN")
    print("=" * 50)
    
    examples = [
        [5.0, 10.0, 50.0],   # 5km, 10°C, 50hm
        [10.0, 10.0, 100.0], # 10km, 10°C, 100hm
        [10.0, 25.0, 100.0], # 10km, 25°C, 100hm (heiß!)
    ]
    
    for dist, temp, elev in examples:
        pred_pace = model.predict([[dist, temp, elev]])[0]
        print(f"{dist:5.1f}km bei {temp:4.0f}°C und {elev:3.0f}hm → Pace: {__print_duartion_from_s(pred_pace)}/km")
    
    print("\n✓ Modell erfolgreich trainiert!")
    return model

def predict_pace(data):
    """Interaktive Pace-Vorhersage"""
    
    # Modell trainieren
    model = __train_pace_model(data)
    if model is None:
        return
    
    print("\n" + "=" * 50)
    print("EIGENE VORHERSAGE")
    print("=" * 50)
    
    try:
        distance = float(input("Distanz (km): "))
        temp = float(input("Temperatur (°C): "))
        elev = float(input("Höhenmeter (m): "))
        
        predicted_pace = model.predict([[distance, temp, elev]])[0]
        return predicted_pace
        
    except ValueError:
        print("❌ Ungültige Eingabe!")
    except KeyboardInterrupt:
        print("\n\nAbgebrochen.")