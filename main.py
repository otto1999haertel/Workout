import csv
from datetime import date, datetime, timedelta
import os
import subprocess
import sys
import pandas as pd
import re
from ml import *

_all_runs = "allRuns.csv"
all_runs_columns = 0

values_not_available = {
    "Active Energy kcal",
    "Weather Temperature °C",
    "Weather Humidity %"
}

# Regex für h / m / s
DURATION_REGEX = re.compile(
    r"^(\d+):([0-5]\d):([0-5]\d)$"
)

def _validate_path(csv_file_path):
    if not os.path.exists(csv_file_path):
        raise Exception("CSV-File not found", csv_file_path)
    return csv_file_path

def _read_all_runs_file():
    return pd.read_csv(_validate_path(_all_runs), delimiter=";")

def _convert_time(tim_to_convert):
    converted_time = str.strip(tim_to_convert)
    converted_time = 'T'.join([converted_time.split()[0], converted_time.split()[1]])
    return converted_time

def _convert_duration_to_second(duration):
    match = DURATION_REGEX.match(duration)
    if not match or not any(match.groups()):
        raise ValueError(f"Ungültige Duration: {duration}")
    hours = int(match.group(1) or 0)
    minutes = int(match.group(2) or 0)
    seconds = int(match.group(3) or 0)
    return timedelta(hours=hours, minutes=minutes, seconds=seconds).seconds

def _convert_decimal(decimal):
    return decimal.replace(',', '.')

def _read_exported_allWorkOuts_file(exported_file):
    complete_data = []
    with open(exported_file, newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file, delimiter=",")
        for row in reader:
            new_row = []
            new_row.append(_convert_time(row.get("Start")))
            new_row.append(_convert_time(row.get("End")))
            new_row.append(_convert_duration_to_second(row.get("Duration")))
            new_row.append(_convert_decimal(row.get("Distance")))
            new_row.append(_convert_decimal(row.get("Average Heart Rate")))
            new_row.append(_convert_decimal(row.get("Max Heart Rate")))
            new_row.append(_convert_duration_to_second(row.get("Average Pace")))
            new_row.append(_convert_decimal(row.get("Average Speed")))
            new_row.append(_convert_decimal(row.get("Average Cadence")))
            new_row.append(_convert_decimal(row.get("Total Energy")))
            new_row.append(_convert_decimal(row.get("Elevation Ascended")))
            new_row.append(_convert_decimal(row.get("Elevation Descended")))
            new_row.append("")  # temp_c (leer)
            new_row.append("")  # notes (leer)
            complete_data.append(new_row)
    return complete_data

def _read_exported_generalData_file(exported_file):
    complete_data = []
    with open(exported_file, newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file, delimiter=",")
        new_row = []
        for row in reader:
            field = row.get("Field")
            if field != "Type":
                value = ""
                if field == "Start" or field == "End":
                    value = _convert_time(row.get("Value"))
                elif field == "Duration" or field == "Average Pace time/km":
                    value = _convert_duration_to_second(row.get("Value"))
                elif field in values_not_available:
                    continue
                else:
                    value = _convert_decimal(row.get("Value"))
                if value != "":
                    print(value)
                    new_row.append(value)
        new_row.append("")  # temp_c (leer)
        new_row.append("")  # notes (leer)
        complete_data.append(new_row)
    print(complete_data) 
    return complete_data

def _filter_duplicates(parsed_data, existing_df):
    """Entfernt Einträge, die bereits in allRuns.csv existieren"""
    existing_starts = set(existing_df['start_time'].values)
    filtered_data = []
    skipped = 0
    
    for row in parsed_data:
        start_time = row[0]
        if start_time not in existing_starts:
            filtered_data.append(row)
        else:
            skipped += 1
            print(f"⚠️  Überspringe Duplikat: {start_time}")
    
    print(f"✓ {len(filtered_data)} neue Läufe, {skipped} Duplikate übersprungen")
    return filtered_data

def _merge_parsed_data(parsed_data):
    # Falls Spaltenweise geliefert → in Zeilen umwandeln
    with open(_all_runs, 'rb') as f:
        f.seek(-1, 2)  # Gehe zum letzten Byte
        last_byte = f.read(1)
        needs_newline = last_byte != b'\n'

    with open(_all_runs, 'a', newline='', encoding='utf-8') as csv_file:
        if needs_newline:
            csv_file.write('\n')
        csv_write = csv.writer(csv_file, delimiter=';')
        csv_write.writerows(parsed_data)

def _sort_parsed_date_ascending_data(parsed_data):
    return sorted(parsed_data, key=lambda x: datetime.fromisoformat(x[0]))

def clean_time():
    with open(_all_runs, newline="") as infile, \
    open("formatted.csv", "w", newline="") as outfile:
        reader = csv.reader(infile, delimiter=";")
        writer = csv.writer(outfile, delimiter=";")
        header = next(reader)
        writer.writerow(header)

        for row in reader:
            row[0] = row[0][:-3]  # Sekunden entfernen
            row[1] = row[1][:-3]
            writer.writerow(row)

def merge_files(current_data, file_to_merge):
    _validate_path(file_to_merge)

    parsed_data = [] 
    if file_to_merge == "raw_data/allWorkouts.csv":
        parsed_data = _read_exported_allWorkOuts_file(file_to_merge)
        # Sort files
        parsed_data = _sort_parsed_date_ascending_data(parsed_data)
    if file_to_merge == "raw_data/generalData.csv":
        parsed_data = _read_exported_generalData_file(file_to_merge)
    
    # Validierung
    if not parsed_data:
        raise ValueError(f"Unbekannte Datei: {file_to_merge}")
    if len(parsed_data[0]) != 14:
        raise Exception(f"Created data not valid. Expected 14 columns, got {len(parsed_data[0])}")
    
    # Duplikate filtern
    parsed_data = _filter_duplicates(parsed_data, current_data)
    
    if len(parsed_data) == 0:
        print("⚠️  Keine neuen Läufe zum Hinzufügen!")
        return
    
    # Daten anhängen
    _merge_parsed_data(parsed_data)
    print(f"✓ {len(parsed_data)} Läufe erfolgreich hinzugefügt!")

def calc_average_s_pace(data):
    df = pd.DataFrame(data)
    avg_pace = df['avg_pace_sec'].mean()
    return avg_pace

def calc_sum_run_km(data):
    total =0
    total = pd.DataFrame(data)['distance_km'].sum()
    return total

def avg_heart_rate(data):
    avg_hr = 0
    avg_hr = pd.DataFrame(data)['avg_hf'].mean()
    return avg_hr

def avg_km_rn(data):
    avg_km=0
    avg_km=pd.DataFrame(data)['distance_km'].mean()
    return avg_km

def __print_duartion_from_s(sec):
    sec = sec % (24 * 3600)  # auf 24h begrenzen
    minutes = int(sec // 60)
    sec = int(sec % 60)
    return "{:02d}m {:02d}s".format(minutes, sec)

def _ins_temp(data, temps):
    temps_l = temps.split()
    date_format = '%d.%m.%Y'
    for temp in temps_l:
        raw_date = temp.split("=")[0]
        raw_date =  datetime.strptime(raw_date, date_format).strftime('%Y-%m-%d')
        raw_temp = temp.split("=")[1]
        mask = data['start_time'].str[:10] == raw_date
        matching_rows = data.loc[mask]
        if len(matching_rows) != 1:
            print(f"Erwarte genau 1 Lauf für {raw_date}, gefunden: {len(matching_rows)}")
            continue
        # Temperatur setzen
        data.loc[mask, 'temp_c'] = float(raw_temp)
        print("Nach Änderung:")
        print(data.loc[mask, ['start_time', 'temp_c']]) 
    data.to_csv(_all_runs, sep=';', index=False)
    
def launch_dashboard():
    """Startet das Streamlit Dashboard"""
    
    dashboard_path = os.path.join(os.path.dirname(__file__), "dashboard.py")
    
    if not os.path.exists(dashboard_path):
        print("❌ dashboard.py nicht gefunden!")
        return
    
    print("🚀 Starte Dashboard...")
    print(f"Führe aus: streamlit run {dashboard_path}")
    
    # Direkter Aufruf mit os.system
    subprocess.run(["streamlit", "run", dashboard_path]) 

def ml_prediction(data):
    pace = predict_pace(data)

if __name__ == "__main__":
    print("Passed lengt ", len(sys.argv))
    current_data = _read_all_runs_file()
    all_runs_columns = current_data.columns.value_counts().size
    if all_runs_columns != 14:
        raise Exception(f"Table not valid. Expected 14 columns, got {all_runs_columns}")
    if len(sys.argv) == 1:
        print("Please provide a command. Possible Commands:")
        print("print_lines : printing lines from csv file")
        print("merge_files required argument: file to merge into allRuns.csv")
        print("average_pace : durchschnitts Pace aller läufe berechnen")
        print("sum_km : summe aller gelaufenen kilometer")
        print("avg_km_rn : durchschnittliche Kilometer pro lauf ")
        print("avg_hr: Durschschnittliche HF")
        print("dahboard")  # visualisierung über streamlit dashboard
    if len(sys.argv) == 2:
        command = sys.argv[1]
        match sys.argv[1]:
            case "print_lines":
                print(current_data)
            case "clean_time":
                clean_time()
            case "average_pace":
                pace_s = calc_average_s_pace(current_data)
                print("Average Pace " +  __print_duartion_from_s(pace_s))
            case "sum_km":
                sum_km = calc_sum_run_km(current_data)
                print("Sum of all run km: " + str(round(sum_km, 2)))
            case "avg_hr":
                avg_hr = avg_heart_rate(current_data)
                print("Average Heart Rate: " + str(round(avg_hr, 0)))
            case "avg_km_rn":
                avg_km = avg_km_rn(current_data)
                print("Avg of all run km: " + str(round(avg_km, 2)))
            case "dashboard":
                launch_dashboard()
            case "ml":
                print(predict_pace(current_data))


    if len(sys.argv) == 3:
        command = sys.argv[1]
        match command:
            case "merge_files":
                print("merging files ...")
                merge_files(current_data, sys.argv[2])