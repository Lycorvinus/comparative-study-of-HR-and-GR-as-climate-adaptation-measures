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
wdata = wdata.resample("30min", on="Datetime").sum()
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
data_OBS_DEU_PT10M_RR = pd.read_csv("C:\\Users\\linus\\OneDrive\\Dokumente\\Publikation\\Data\\data_OBS_DEU_PT10M_RR.csv", sep = ",", header = 0, index_col=False)
print(data_OBS_DEU_PT10M_RR)
data_OBS_DEU_PT10M_RR['Zeitstempel'] = pd.to_datetime(data_OBS_DEU_PT10M_RR['Zeitstempel'])
data_OBS_DEU_PT10M_RR.set_index("Zeitstempel", inplace=True)
data_OBS_DEU_PT10M_RR = data_OBS_DEU_PT10M_RR.apply(pd.to_numeric, errors="coerce")
data_OBS_DEU_PT10M_RR = data_OBS_DEU_PT10M_RR.resample("30min").sum()
data_OBS_DEU_PT10M_RR.reset_index(inplace=True)
dwd_nd_hp2 = data_OBS_DEU_PT10M_RR.loc[(data_OBS_DEU_PT10M_RR["Zeitstempel"].dt.date >= pd.to_datetime("2023-09-04").date()) & (data_OBS_DEU_PT10M_RR["Zeitstempel"].dt.date < pd.to_datetime("2023-09-12").date())]

# Wetterdaten vom LWI (Messgerät OTT Pluvio)
hydriv_niederschlagsdaten = pd.read_csv("C:\\Users\\linus\\OneDrive\\Dokumente\\Publikation\\Data\\Hydriv_Niederschlagsdaten_072023_032024.csv", sep = ",", header = 0, index_col=False)
print(hydriv_niederschlagsdaten)
hydriv_niederschlagsdaten["Zeitstempel"] = pd.to_datetime(hydriv_niederschlagsdaten["Zeitstempel"], format="%d.%m.%Y %H:%M:%S")
hydriv_niederschlagsdaten.set_index("Zeitstempel", inplace=True)
hydriv_niederschlagsdaten["Niederschlag [mm]"] = hydriv_niederschlagsdaten["Niederschlag [mm]"].str.replace(",", ".")
hydriv_niederschlagsdaten = hydriv_niederschlagsdaten.apply(pd.to_numeric)
hydriv_niederschlagsdaten = hydriv_niederschlagsdaten.resample("30min").sum()
hydriv_niederschlagsdaten.reset_index(inplace=True)
hydriv_niederschlagsdaten.to_csv("C:\\Users\\linus\\OneDrive\\Dokumente\\Publikation\\Data\\Hydriv_Niederschlagsdaten.csv", sep = ",", header=True, index=False)


# Test ob es Differenzen im Niederschlag während HP_2 gibt
hw_nd_hp2.set_index("Datetime", inplace=True)
dwd_nd_hp2.set_index("Zeitstempel", inplace=True)
test=hw_nd_hp2["PS"] - dwd_nd_hp2["Wert"]

print(test)

plt.figure(figsize=(10,6))
plt.plot(hw_nd_hp2.index, hw_nd_hp2["PS"])
plt.plot(dwd_nd_hp2.index, dwd_nd_hp2["Wert"])
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
test=mdata_nd_hp2["Precipitation"] - dwd_nd_hp2["Wert"]

print(test)

plt.figure(figsize=(10,6))
plt.plot(hw_nd_hp2.index, hw_nd_hp2["PS"])
plt.plot(dwd_nd_hp2.index, dwd_nd_hp2["Wert"])
plt.show()


mdata.set_index("Datetime", inplace=True)
data_OBS_DEU_PT10M_RR.set_index("Zeitstempel", inplace=True)
common_indices = mdata.index.intersection(data_OBS_DEU_PT10M_RR.index)
data_OBS_DEU_PT10M_RR = data_OBS_DEU_PT10M_RR.loc[common_indices]
hydriv_niederschlagsdaten.set_index("Zeitstempel", inplace=True)
hydriv_niederschlagsdaten = hydriv_niederschlagsdaten.loc[common_indices]

plt.figure(figsize=(30,5))
plt.plot(hydriv_niederschlagsdaten.index, hydriv_niederschlagsdaten["Niederschlag [mm]"], color="green")
plt.show()

fig, (ax1,ax2) = plt.subplots(2, 1, figsize=(30,10))
ax1.bar(data_OBS_DEU_PT10M_RR.index, data_OBS_DEU_PT10M_RR["Wert"], color="blue", label="DWD Daten")
ax1.legend()
ax2.bar(hydriv_niederschlagsdaten.index, hydriv_niederschlagsdaten["Niederschlag [mm]"], color="green", label="LWI Daten")
ax2.legend()
# ax3.bar(mdata.index, mdata["Precipitation"], color="red", label="IGÖ Daten")
# ax3.legend()
plt.show()