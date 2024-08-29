import pandas as pd 
import numpy as np
import pathlib
import glob
import matplotlib.pyplot as plt
from matplotlib.dates import MonthLocator, DateFormatter


files = glob.glob("C:\\Users\\linus\\OneDrive\\Dokumente\\Masterarbeit\\Data\\formatierte Daten\\*Flux_CSFormat*.dat") # Liste für das iterative einlesen von den Time_Series Dateien
# Importieren der Flux_CSFormat Dateien Header über skiprows ausgelassen und Spaltennamen über skiprows belassen
logger_data = pd.concat([pd.read_csv(f, skiprows=[0, 2, 3], sep=",", low_memory=False) for f in files], ignore_index=True)  
logger_data["TIMESTAMP"] = pd.to_datetime(logger_data["TIMESTAMP"].str.strip(), yearfirst=True, format="mixed") # Date Time Formatierung
logger_data["Date"] = pd.to_datetime(logger_data["TIMESTAMP"]).dt.date
logger_data["Time"] = pd.to_datetime(logger_data["TIMESTAMP"]).dt.time
logger_data["Date"] = pd.to_datetime(logger_data["Date"], format = "%Y-%m-%d")
logger_data["Time"] = pd.to_datetime(logger_data["Time"], format = "%H:%M:%S")
print(logger_data.info())
print(logger_data)
# Daten vor dem fertigen Messaufbau entfernen
Deadline = "2023-07-08 00:00:30" 
Deadline = pd.to_datetime(Deadline, format="mixed")
logger_data = logger_data[logger_data["TIMESTAMP"] >=  Deadline]
# Entfernen der Record Spalte um unnötige Daten los zu werden
logger_data = logger_data.drop("RECORD", axis=1)
# Sortieren nach Date und vergeben neuer Indizes (Zeilennahmen)
logger_data = logger_data.sort_values(by="TIMESTAMP")
print(logger_data)
logger_data.index = [list(range(len(logger_data)))]
print(logger_data)
print(len(logger_data))

# Doppelte Werte!!! Werden hier entfernt
# Tag hat nur 47 (ohne Dopplungen) anstatt wie alle anderen 48 Messwerte
ts = "2023-11-02"
ts = pd.to_datetime(ts, format = "%Y-%m-%d")
print(logger_data[logger_data["Date"] ==  ts])
# Zählvariablen für die For Schleife um auf 2 verschiedene Zeilen zugreifen zu können
b = 0
n = 1
# Doppelte Zeilen entfernen
for i in range(len(logger_data)):
    if n < len(logger_data):
        doubledata1 = logger_data.iloc[b, 0]
        doubledata2 = logger_data.iloc[n, 0]
        if doubledata1 == doubledata2:
            logger_data.drop(n, axis=0, inplace=True)
            logger_data.index = [list(range(len(logger_data)))]
    b = b + 1
    n = n + 1
print(logger_data)
print(logger_data[logger_data["Date"] ==  ts])

logger_data = logger_data.rename(columns={"TIMESTAMP": "Datetime"})
logger_data.to_csv("C:\\Users\\linus\\OneDrive\\Dokumente\\Publikation\\Data\\logger_data.csv", sep = ",", header = True, index=False)




# Einlesen der Wetterdaten von der IGÖ Dachstation

# Einzelne Datensätze
files = glob.glob("C:\\Users\\linus\\OneDrive\\Dokumente\\Masterarbeit\\Data\\Wetterstation\\*TOA5*.dat") # Liste für das iterative einlesen von Daten der Wetterstation
# Importieren der Flux_CSFormat Dateien Header über skiprows ausgelassen und Spaltennamen über skiprows belassen
wdata = pd.concat([pd.read_csv(f, skiprows=[0, 2, 3], sep=",", low_memory=False) for f in files], ignore_index=True)
print(wdata.dtypes) 
wdata["TIMESTAMP"] = pd.to_datetime(wdata["TIMESTAMP"].str.strip(), yearfirst=True, format="mixed") # Date Time Formatierung
wdata = wdata.rename(columns={"TIMESTAMP": "Datetime"})
# Date und Time spalten erst nach 30 Min mean berechnung in anderem Skript erstellt.
print(wdata.info())
print(wdata)
# Datenytpen die nicht object sein sollen zu float64 umwandeln
for column in wdata.select_dtypes(include="object").columns:
    wdata[column] = wdata[column].astype(float)


# 30 Minuten Mittelwerte für wdata
wdata = wdata.resample("30min", on="Datetime").sum()
wdata.to_csv("C:\\Users\\linus\\OneDrive\\Dokumente\\Publikation\\Data\\wdata.csv", sep = ",", header = True)
print(wdata)



# Daten vom 01.07.2023 bis zum 31.01.2024 einlesen
halbstundenwerte = pd.read_csv("C:\\Users\\linus\\OneDrive\\Dokumente\\Masterarbeit\\Data\\Wetterstation\\halbstundenwerte.txt", sep = "\t", header = 0)

# Convert columns to numeric
columns_to_convert = ["LT ", " RH ", " DR ", " QN ", " KD ", " PS "]
halbstundenwerte[columns_to_convert] = halbstundenwerte[columns_to_convert].apply(pd.to_numeric, errors="coerce")
# Combine "Datum" and "Zeit(hh:mm:ss)" columns to create "Datetime" column
# Convert the "Datum" column to datetime format
halbstundenwerte["Datum "] = pd.to_datetime(halbstundenwerte["Datum "], dayfirst = True)

# Convert the "Zeit(hh:mm:ss)" column to datetime format
halbstundenwerte["Zeit"] = pd.to_datetime(halbstundenwerte[" Zeit(hh:mm:ss) "].str.strip()).dt.time

# Combine date and time columns into a single datetime column
halbstundenwerte["Datetime"] = halbstundenwerte["Datum "] + pd.to_timedelta(halbstundenwerte["Zeit"].astype(str))

halbstundenwerte.drop(columns=["Datum ", " Zeit(hh:mm:ss) ", "Zeit"], inplace=True)


halbstundenwerte.rename(columns={"LT ": "LT", " RH ": "RH", " DR ": "DR", " QN ": "QN", " KD ": "KD", " PS ": "PS"}, inplace=True)
halbstundenwerte = halbstundenwerte[halbstundenwerte["Datetime"] >= Deadline]
halbstundenwerte.set_index("Datetime", inplace = True)

# wdata und halbstundenwerte ergänzend zusammenfügen für PS
mdata = pd.DataFrame()
mdata = pd.merge(wdata["Rain_mm_Tot"], halbstundenwerte["PS"], left_index=True, right_index=True, how="outer")
mdata.ffill(inplace=True)
mdata.bfill(inplace=True)
mdata["Precipitation"] = mdata["Rain_mm_Tot"] + mdata["PS"]
mdata.drop(columns=["Rain_mm_Tot", "PS"], inplace=True)
print(mdata)
mdata.to_csv("C:\\Users\\linus\\OneDrive\\Dokumente\\Publikation\\Data\\mdata.csv", sep = ",", header = True)


# importieren Oberflächentemperaturmessung pro Tag
st_day1 = pd.read_csv("C:\\Users\\linus\\OneDrive\\Dokumente\\Masterarbeit\\Data\\Oberflächentemperatur Messung\\ST Day 1.csv", sep = ";", low_memory=False)
st_day2 = pd.read_csv("C:\\Users\\linus\\OneDrive\\Dokumente\\Masterarbeit\\Data\\Oberflächentemperatur Messung\\ST Day 2.csv", sep = ";", low_memory=False)
st_day3 = pd.read_csv("C:\\Users\\linus\\OneDrive\\Dokumente\\Masterarbeit\\Data\\Oberflächentemperatur Messung\\ST Day 3.csv", sep = ";", low_memory=False)
st_day1["Datetime"] = pd.to_datetime(st_day1["Date"] + " " + st_day1["Time"])
st_day2["Datetime"] = pd.to_datetime(st_day2["Date"] + " " + st_day2["Time"])
st_day3["Datetime"] = pd.to_datetime(st_day3["Date"] + " " + st_day3["Time"])
st_day1["Date"] = pd.to_datetime(st_day1["Datetime"]).dt.date
st_day1["Time"] = pd.to_datetime(st_day1["Datetime"]).dt.time
st_day1["Date"] = pd.to_datetime(st_day1["Date"], format = "%Y-%m-%d").dt.date
st_day1["Time"] = pd.to_datetime(st_day1["Time"], format="%H:%M:%S").dt.strftime("%H:%M:%S")
st_day2["Date"] = pd.to_datetime(st_day2["Datetime"]).dt.date
st_day2["Time"] = pd.to_datetime(st_day2["Datetime"]).dt.time
st_day2["Date"] = pd.to_datetime(st_day2["Date"], format = "%Y-%m-%d").dt.date
st_day2["Time"] = pd.to_datetime(st_day2["Time"], format="%H:%M:%S").dt.strftime("%H:%M:%S")
st_day3["Date"] = pd.to_datetime(st_day3["Datetime"]).dt.date
st_day3["Time"] = pd.to_datetime(st_day3["Datetime"]).dt.time
st_day3["Date"] = pd.to_datetime(st_day3["Date"], format = "%Y-%m-%d").dt.date
st_day3["Time"] = pd.to_datetime(st_day3["Time"], format="%H:%M:%S").dt.strftime("%H:%M:%S")
surface_temp_data = pd.concat([st_day1, st_day2, st_day3])
print(surface_temp_data)
print(surface_temp_data.dtypes)
surface_temp_data.to_csv("C:\\Users\\linus\\OneDrive\\Dokumente\\Publikation\\Data\\Oberflächentemperatur Messung\\surface_temp_data.csv", sep=",", header = True, index=False)




# 30 Minuten Mittelwerte für Oberflächentemperatur berechnen
columns_to_include = ["GD_MW", "GD_MAX", "GD_MIN", "GD_StdAbw", "HD_MW", "HD_MAX", "HD_MIN", "HD_StdAbw"]
st_day1.set_index("Datetime", inplace=True)
st_day2.set_index("Datetime", inplace=True)
st_day3.set_index("Datetime", inplace=True)
st_day1 = st_day1[columns_to_include].resample("30min").mean()
st_day2 = st_day2[columns_to_include].resample("30min").mean()
st_day3 = st_day3[columns_to_include].resample("30min").mean()
st_data = pd.concat([st_day1, st_day2, st_day3])
st_data.reset_index(drop = False, inplace = True)
print(st_data)
st_data.to_csv("C:\\Users\\linus\\OneDrive\\Dokumente\\Publikation\\Data\\Oberflächentemperatur Messung\\st_data.csv", sep = ",", header = True, index = False)