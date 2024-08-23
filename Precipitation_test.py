import pandas as pd 
import numpy as np
import pathlib
import glob
import matplotlib.pyplot as plt
from matplotlib.dates import MonthLocator, DateFormatter


# Einlesen der Wetterdaten von der IGÖ Dachstation

# Einzelne Datensätze
files = glob.glob("C:\\Users\\linus\\OneDrive\\Dokumente\\Masterarbeit\\Daten\\Wetterstation\\*TOA5*.dat") # Liste für das iterative einlesen von Daten der Wetterstation
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
wdata = wdata.resample("30min", on="Datetime").mean()
print(wdata)



# Daten vom 01.07.2023 bis zum 31.01.2024 einlesen
halbstundenwerte = pd.read_csv("C:\\Users\\linus\\OneDrive\\Dokumente\\Masterarbeit\\Daten\\Wetterstation\\halbstundenwerte.txt", sep = "\t", header = 0)

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

Deadline = "2023-07-08 00:00:30" 
Deadline = pd.to_datetime(Deadline, format="mixed")

halbstundenwerte.rename(columns={"LT ": "LT", " RH ": "RH", " DR ": "DR", " QN ": "QN", " KD ": "KD", " PS ": "PS"}, inplace=True)
halbstundenwerte = halbstundenwerte[halbstundenwerte["Datetime"] >= Deadline]

hw_nd_hp2 = halbstundenwerte.loc[(halbstundenwerte["Datetime"].dt.date >= pd.to_datetime("2023-09-04").date()) & (halbstundenwerte["Datetime"].dt.date < pd.to_datetime("2023-09-12").date())]
print(hw_nd_hp2)


# Wetterdaten vom WD
produkt_zehn_min_rr_20200101_20231231_00662 = pd.read_csv("C:\\Users\\linus\\OneDrive\\Dokumente\\Publikation\\produkt_zehn_min_rr_20200101_20231231_00662.txt", sep = ";", header = 0)
print(produkt_zehn_min_rr_20200101_20231231_00662)
produkt_zehn_min_rr_20200101_20231231_00662["MESS_DATUM"] = pd.to_datetime(produkt_zehn_min_rr_20200101_20231231_00662["MESS_DATUM"], format="%Y%m%d%H%M")
produkt_zehn_min_rr_20200101_20231231_00662.set_index("MESS_DATUM", inplace=True)
produkt_zehn_min_rr_20200101_20231231_00662 = produkt_zehn_min_rr_20200101_20231231_00662.apply(pd.to_numeric, errors="coerce")
produkt_zehn_min_rr_20200101_20231231_00662 = produkt_zehn_min_rr_20200101_20231231_00662.resample("30min").mean()
produkt_zehn_min_rr_20200101_20231231_00662.reset_index(inplace=True)
dwd_nd_hp2 = produkt_zehn_min_rr_20200101_20231231_00662.loc[(produkt_zehn_min_rr_20200101_20231231_00662["MESS_DATUM"].dt.date >= pd.to_datetime("2023-09-04").date()) & (produkt_zehn_min_rr_20200101_20231231_00662["MESS_DATUM"].dt.date < pd.to_datetime("2023-09-12").date())]

# Test ob es Differenzen im Niederschlag während HP_2 gibt
hw_nd_hp2.set_index("Datetime", inplace=True)
dwd_nd_hp2.set_index("MESS_DATUM", inplace=True)
test=hw_nd_hp2["PS"] - dwd_nd_hp2["RWS_10"]

print(test)

plt.figure(figsize=(10,6))
plt.plot(hw_nd_hp2.index, hw_nd_hp2["PS"])
plt.plot(dwd_nd_hp2.index, dwd_nd_hp2["RWS_10"])
plt.show()

# wdata und halbstundenwerte ergänzend zusammenfügen für PS
halbstundenwerte.set_index("Datetime", inplace = True)
mdata = pd.DataFrame()
mdata = pd.merge(wdata["Rain_mm_Tot"], halbstundenwerte["PS"], left_index=True, right_index=True, how="outer")
mdata.ffill(inplace=True)
mdata.bfill(inplace=True)
mdata["Precipitation"] = mdata["Rain_mm_Tot"] + mdata["PS"]
mdata.drop(columns=["Rain_mm_Tot", "PS"], inplace=True)
print(mdata)

# Test ob es Differenzen im Niederschlag während HP_2 gibt mit Daten aus mdata
mdata.reset_index(inplace=True)
mdata_nd_hp2 = mdata.loc[(mdata["Datetime"].dt.date >= pd.to_datetime("2023-09-04").date()) & (mdata["Datetime"].dt.date < pd.to_datetime("2023-09-12").date())]


mdata_nd_hp2.set_index("Datetime", inplace=True)
test=mdata_nd_hp2["Precipitation"] - dwd_nd_hp2["RWS_10"]

print(test)

plt.figure(figsize=(10,6))
plt.plot(hw_nd_hp2.index, hw_nd_hp2["PS"])
plt.plot(dwd_nd_hp2.index, dwd_nd_hp2["RWS_10"])
plt.show()
