import pandas as pd 
import numpy as np
from matplotlib import font_manager
import matplotlib.pyplot as plt
from matplotlib.dates import MonthLocator, DateFormatter
import matplotlib.dates as mdates
import statsmodels.api as sm
import statsmodels.formula.api as smf
import seaborn as sns
from mpl_toolkits.mplot3d import axes3d
from datetime import timedelta
from datetime import time
from windrose import WindroseAxes




# import completed_data
completed_data = pd.read_csv("C:\\Users\\linus\\OneDrive\\Dokumente\\Masterarbeit\\Daten\\completed_data.csv", sep = ",", low_memory=False)
completed_data["Datetime"] = pd.to_datetime(completed_data["Datetime"]) # Date Time Formatierung
completed_data["Date"] = pd.to_datetime(completed_data["Datetime"]).dt.date
completed_data["Time"] = pd.to_datetime(completed_data["Datetime"]).dt.time
completed_data["Date"] = pd.to_datetime(completed_data["Date"], format = "%Y-%m-%d").dt.date
completed_data["Time"] = pd.to_datetime(completed_data["Time"], format="%H:%M:%S").dt.strftime("%H:%M:%S")
print(completed_data)

selected_columns = ["TA_1_1_2", "TS_CS65X_1_1_1", "TS_CS65X_1_1_2", "SWC_1_1_1", "SWC_1_1_2", "VWC_1_1_1", "VWC_1_1_2", "G_plate_1_1_1", "G_plate_1_1_2", "G_1_1_1", "G_1_1_2", "SG_1_1_1", "SG_1_1_2", "WS", "WD", "WD_SONIC", "SW_IN", "SW_OUT", "LW_IN", "LW_OUT", "RH_1_1_2", "GR_ST", "HR_ST", "Precipitation"]
completed_data[selected_columns] = completed_data[selected_columns].apply(pd.to_numeric)
print(completed_data.dtypes)
# completed_data.set_index("Datetime", inplace=True)
completed_data["WS_KMH"] = completed_data["WS"] * 3.6


# Perzentile der Tageshöchsttemperatur
ta_95 = completed_data["TA_1_1_2"].quantile(q=0.95)
ta_5 = completed_data["TA_1_1_2"].mean()
ta_5 = np.quantile(ta_5, q=0.05)
# Deswegen: Heiztage (Tagesmittellufttemperatur < 12) und 95 Perzentil der TA (Tageshöchsttemperatur > 25)

## Timeseries
# VWC
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 132
plt.figure(figsize=(100,25))
plt.plot(completed_data["Datetime"], completed_data["VWC_1_1_1"], label="VWC$_{GR}$", color="dodgerblue", linestyle="-", linewidth=10)
plt.plot(completed_data["Datetime"], completed_data["VWC_1_1_2"], label="VWC$_{HR}$", color="crimson", linestyle="-", linewidth=10)
plt.ylim(0, 0.8)
plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8])
plt.xlim(completed_data["Datetime"].min(), completed_data["Datetime"].max())
date_format = mdates.DateFormatter("%b %Y")  # e.g., "Jul 2023", "Aug 2023"
plt.gca().xaxis.set_major_formatter(date_format)
plt.legend(loc="center", bbox_to_anchor=(0.5, -0.25), ncol = 2, frameon=False)
plt.grid()
plt.tight_layout()
#plt.savefig("C:\\Users\\linus\\OneDrive\\Dokumente\\Masterarbeit\\Graphen\\TimeSeries VWC.pdf", format="pdf")
plt.show()

# Surface Temp
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 132
plt.figure(figsize=(100,25))
plt.plot(completed_data["Datetime"], completed_data["GR_ST"], label="T$_{Surface}$ $_{GR}$", color="dodgerblue", linestyle="-", linewidth=5)
plt.plot(completed_data["Datetime"], completed_data["HR_ST"], label="T$_{Surface}$ $_{HR}$", color="crimson", linestyle="-",linewidth=5)
plt.xlim(completed_data["Datetime"].min(), completed_data["Datetime"].max())
date_format = mdates.DateFormatter("%b %Y") 
plt.ylabel("Surface\nTemperature (°C)")
plt.gca().xaxis.set_major_formatter(date_format)
plt.legend(loc="center", bbox_to_anchor=(0.5, -0.25), ncol = 2, frameon=False)
plt.grid()
plt.tight_layout()
#plt.savefig("C:\\Users\\linus\\OneDrive\\Dokumente\\Masterarbeit\\Graphen\\TimeSeries ST.pdf", format="pdf")
plt.show()


## Boxplots 
# TA, rH, WS
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 24
fig, ax1 = plt.subplots(figsize=(10, 6))
sns.boxplot(data=completed_data["RH_1_1_2"], linecolor="dodgerblue", fliersize=3,
            whis=[10, 90], showmeans=True, meanprops={"marker":"x","markerfacecolor":"black", "markeredgecolor":"black", "markersize":10}, ax=ax1, fill=False)
ax2 = ax1.twinx()
sns.boxplot(data=completed_data[["TA_1_1_2", "WS_KMH"]], palette=["crimson", "goldenrod"], fliersize=3,
            whis=[10, 90], showmeans=True, meanprops={"marker":"x","markerfacecolor":"black", "markeredgecolor":"black", "markersize":10}, ax=ax2, fill=False)
ax1.set_xticks(ticks=[0, 1, 2])
ax1.set_xticklabels(labels=["Relative\nHumidity\n(%)", "Air\nTemperature\n(°C)", "Windspeed\n(km/h)"])
ax1.set_ylim(0, 100)
ax2.set_ylim(15, 35)
ax1.set_ylabel("Relative Humidity")
ax2.set_ylabel("T$_{Air}$ / WS")
ax1.set_yticks([0, 20, 40, 60, 80, 100])
ax2.set_yticks([-15, -5, 5, 15, 25, 35])
ax1.grid(False)
ax2.grid(False)
ax1.grid(axis="y")
plt.tight_layout()
#plt.savefig("C:\\Users\\linus\\OneDrive\\Dokumente\\Masterarbeit\\Graphen\\Boxplot TA rH WS.pdf", format="pdf")
plt.show()


# Green Roof SWC, Bodentemperatur, Oberflächentemperatur, QG_HFP
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 28
plt.figure(figsize=(10, 6))
sns.boxplot(data=completed_data[["SWC_1_1_1", "TS_CS65X_1_1_1", "G_plate_1_1_1", "GR_ST"]], palette=["dodgerblue", "crimson","goldenrod", "dimgray"],
            whis=[10, 90], showmeans=True, meanprops={"marker":"x","markerfacecolor":"black", "markeredgecolor":"black", "markersize":10}, fliersize=3, fill=False)
plt.grid(True)
plt.xticks(ticks=[0,1,2,3], labels=["Sat.\nCoeff.\n(%)", "T$_{Soil}$\n(°C)", "Q$_G$ $_{HFP}$\n$W/m²$", "T$_{Surface}$\n(°C)"])
#plt.savefig("C:\\Users\\linus\\OneDrive\\Dokumente\\Masterarbeit\\Graphen\\Boxplot Green Roof SWC TS QG_HFP.pdf", format="pdf", bbox_inches="tight")
plt.show()

# Hybrid Roof SWC, Bodentemperatur, Oberflächentemperatur, QG_HFP
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 28
plt.figure(figsize=(10, 6))
sns.boxplot(data=completed_data[["SWC_1_1_2", "TS_CS65X_1_1_2", "G_plate_1_1_2", "HR_ST"]], palette=["dodgerblue", "crimson","goldenrod", "dimgray"],
            whis=[10, 90], showmeans=True, meanprops={"marker":"x","markerfacecolor":"black", "markeredgecolor":"black", "markersize":10}, fliersize=3, fill=False)
plt.grid(True)
plt.xticks(ticks=[0,1,2,3], labels=["Sat.\nCoeff.\n(%)", "T$_{Soil}$\n(°C)", "Q$_G$ $_{HFP}$\n$W/m²$", "T$_{Surface}$\n(°C)"])
#plt.savefig("C:\\Users\\linus\\OneDrive\\Dokumente\\Masterarbeit\\Graphen\\Boxplot Hybrid Roof SWC TS QG HFP.pdf", format="pdf", bbox_inches="tight")
plt.show()

print(completed_data["TS_CS65X_1_1_1"].mean())
print(completed_data["TS_CS65X_1_1_2"].mean())

# Strahlung
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 24
plt.figure(figsize=(17, 10))
sns.boxplot(data=completed_data[["SW_IN", "LW_IN","SW_OUT_GR_Calculated", "SW_OUT_HR_Calculated", "LW_OUT_GR_Calculated", "LW_OUT_HR_Calculated"]],
            whis=[10, 90], showmeans=True, meanprops={"marker":"x","markerfacecolor":"white", "markeredgecolor":"white"})
plt.grid(True)
plt.ylabel("$W/m^2$")
plt.xticks(ticks=[0,1,2,3,4,5], labels=["SW IN", "LW IN", "SW OUT GR", "SW OUT HR", "LW OUT GR", "LW OUT HR"])
#plt.savefig("C:\\Users\\linus\\OneDrive\\Dokumente\\Masterarbeit\\Graphen\\Boxplot Radiation.pdf", format="pdf", bbox_inches="tight")
plt.show()

# Green Roof QE, QH, QG, Q* 
palette = {
    "Q*_flux_gr": "dodgerblue",
    "QE_GR": "crimson",
    "QH_GR": "goldenrod",
    "QG_GR": "dimgray"
}
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 36
plt.figure(figsize=(10, 6))
sns.boxplot(data=completed_data[["Q*_flux_gr", "QE_GR","QH_GR", "QG_GR"]],
            whis=[10, 90], showmeans=True, meanprops={"marker":"x","markerfacecolor":"black", "markeredgecolor":"black", "markersize":10}, palette=palette, fliersize=3, fill=False)
plt.grid(True)
plt.ylabel("$W/m^2$")
plt.xticks(ticks=[0,1,2,3], labels=["Q*", "Q$_E$", "Q$_H$", "Q$_G$"])
#plt.savefig("C:\\Users\\linus\\OneDrive\\Dokumente\\Masterarbeit\\Graphen\\Boxplot Green Roof Heat Fluxes.pdf", format="pdf", bbox_inches="tight")
plt.show()

# Hybrid Roof QE, QH, QG, Q* 
palette = {
    "Q*_flux_hr": "dodgerblue",
    "QE_HR": "crimson",
    "QH_HR": "goldenrod",
    "QG_HR": "dimgray"
}
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 36
plt.figure(figsize=(10, 6))
sns.boxplot(data=completed_data[["Q*_flux_hr", "QE_HR","QH_HR", "QG_HR"]],
            whis=[10, 90], showmeans=True, meanprops={"marker":"x","markerfacecolor":"black", "markeredgecolor":"black", "markersize":10}, palette=palette, fliersize=3, fill=False)
plt.grid(True)
plt.ylabel("$W/m^2$")
plt.xticks(ticks=[0,1,2,3], labels=["Q*", "Q$_E$", "Q$_H$", "Q$_G$"])
#plt.savefig("C:\\Users\\linus\\OneDrive\\Dokumente\\Masterarbeit\\Graphen\\Boxplot Hybrid Roof Heat Fluxes.pdf", format="pdf", bbox_inches="tight")
plt.show()

print(completed_data["QE_GR"].quantile(0.9))
## Boxplots Heiztage
# TA, rH, WS
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 16
fig, ax1 = plt.subplots(figsize=(10, 6))
sns.boxplot(data=completed_data.loc[(completed_data.groupby(completed_data["Datetime"].dt.date)["TA_1_1_2"].transform("mean") < 12), "RH_1_1_2"], color="red",
            whis=[10, 90], showmeans=True,
            meanprops={"marker":"x","markerfacecolor":"white", "markeredgecolor":"white"}, ax=ax1)
ax2 = ax1.twinx()
sns.boxplot(data=completed_data.loc[(completed_data.groupby(completed_data["Datetime"].dt.date)["TA_1_1_2"].transform("mean") < 12), ["TA_1_1_2", "WS"]],whis=[10, 90],
            showmeans=True,
            meanprops={"marker":"x","markerfacecolor":"white", "markeredgecolor":"white"}, ax=ax2)
ax1.set_xticks(ticks=[0, 1, 2])
ax1.set_xticklabels(labels=["Relative Humidity (%)", "Air Temperature (°C)", "Windspeed ($km/h$)"])
ax1.set_ylim(0, 100)
ax2.set_ylim(-12, 18)
ax1.set_ylabel("Relative Humidity")
ax2.set_ylabel("Airtemperature (°C) / Windspeed ($km/h$)")
ax1.set_yticks([0, 20, 40, 60, 80, 100])
ax2.set_yticks([-12, -6, 0, 6, 12, 18])
ax1.grid(False)
ax2.grid(False)
ax1.grid(axis="y")
#plt.savefig("C:\\Users\\linus\\OneDrive\\Dokumente\\Masterarbeit\\Graphen\\Boxplot Heiztage TA rH WS.pdf", format="pdf", bbox_inches="tight")
plt.show()



# Green Roof SWC, Bodentemperatur, Oberflächentemperatur, QG_HFP
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 16
plt.figure(figsize=(10, 6))
sns.boxplot(data=completed_data.loc[(completed_data.groupby(completed_data["Datetime"].dt.date)["TA_1_1_2"].transform("mean") < 12), ["SWC_1_1_1", "TS_CS65X_1_1_1","GR_ST", "G_plate_1_1_1"]],
            whis=[10, 90], showmeans = True,
            meanprops={"marker":"x","markerfacecolor":"white", "markeredgecolor":"white"})
plt.grid(True)
plt.xticks(ticks=[0,1,2,3], labels=["Saturation Coefficient\n(%)", "Soil Temperature\n(°C)", "Surface Temperature\n(°C)", "Q$_G$ $_{Heat}$ $_{Flux}$ $_{Plate}$"])
#plt.savefig("C:\\Users\\linus\\OneDrive\\Dokumente\\Masterarbeit\\Graphen\\Boxplot Heiztage Green Roof SWC TS ST QGHFP.pdf", format="pdf", bbox_inches="tight")
plt.show()


# Hybrid Roof SWC, Bodentemperatur, Oberflächentemperatur, QG_HFP
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 16
plt.figure(figsize=(10, 6))
sns.boxplot(data=completed_data.loc[(completed_data.groupby(completed_data["Datetime"].dt.date)["TA_1_1_2"].transform("mean") < 12), ["SWC_1_1_2", "TS_CS65X_1_1_2","HR_ST", "G_plate_1_1_2"]],
            whis=[10, 90], showmeans = True,
            meanprops={"marker":"x","markerfacecolor":"white", "markeredgecolor":"white"})
plt.grid(True)
plt.xticks(ticks=[0,1,2,3], labels=["Saturation Coefficient\n(%)", "Soil Temperature\n(°C)", "Surface Temperature\n(°C)", "Q$_G$ $_{Heat}$ $_{Flux}$ $_{Plate}$\nduring Frost Days"])
#plt.savefig("C:\\Users\\linus\\OneDrive\\Dokumente\\Masterarbeit\\Graphen\\Boxplot Heiztage Hybrid Roof SWC TS ST QG_HFP.pdf", format="pdf", bbox_inches="tight")
plt.show()

# Strahlung
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 24
plt.figure(figsize=(17, 10))
sns.boxplot(data=completed_data.loc[(completed_data.groupby(completed_data["Datetime"].dt.date)["TA_1_1_2"].transform("mean") < 12),["SW_IN", "LW_IN","SW_OUT_GR_Calculated", "SW_OUT_HR_Calculated", "LW_OUT_GR_Calculated", "LW_OUT_HR_Calculated"]],
            whis=[10, 90], showmeans=True, meanprops={"marker":"x","markerfacecolor":"white", "markeredgecolor":"white"})
plt.grid(True)
plt.ylabel("$W/m^2$")
plt.xticks(ticks=[0,1,2,3,4,5], labels=["SW IN", "LW IN", "SW OUT GR", "SW OUT HR", "LW OUT GR", "LW OUT HR"])
#plt.savefig("C:\\Users\\linus\\OneDrive\\Dokumente\\Masterarbeit\\Graphen\\Boxplot Heiztage Radiation.pdf", format="pdf", bbox_inches="tight")
plt.show()

# Green Roof QE, QH, QG, Q* 
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 20
plt.figure(figsize=(10, 6))
sns.boxplot(data=completed_data.loc[(completed_data.groupby(completed_data["Datetime"].dt.date)["TA_1_1_2"].transform("mean") < 12),["Q*_flux_gr", "QE_GR","QH_GR", "QG_GR"]],
            whis=[10, 90], showmeans=True, meanprops={"marker":"x","markerfacecolor":"white", "markeredgecolor":"white"})
plt.grid(True)
plt.ylabel("$W/m^2$")
plt.xticks(ticks=[0,1,2,3], labels=["Q*", "Q$_E$", "Q$_H$", "Q$_G$"])
#plt.savefig("C:\\Users\\linus\\OneDrive\\Dokumente\\Masterarbeit\\Graphen\\Boxplot Heiztage Green Roof Heat Fluxes.pdf", format="pdf", bbox_inches="tight")
plt.show()

# Hybrid Roof QE, QH, QG, Q* 
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 20
plt.figure(figsize=(10, 6))
sns.boxplot(data=completed_data.loc[(completed_data.groupby(completed_data["Datetime"].dt.date)["TA_1_1_2"].transform("mean") < 12),["Q*_flux_hr", "QE_HR","QH_HR", "QG_HR"]],
            whis=[10, 90], showmeans=True, meanprops={"marker":"x","markerfacecolor":"white", "markeredgecolor":"white"})
plt.grid(True)
plt.ylabel("$W/m^2$")
plt.xticks(ticks=[0,1,2,3], labels=["Q*", "Q$_E$", "Q$_H$", "Q$_G$"])
#plt.savefig("C:\\Users\\linus\\OneDrive\\Dokumente\\Masterarbeit\\Graphen\\Boxplot Heiztage Hybrid Roof Heat Fluxes.pdf", format="pdf", bbox_inches="tight")
plt.show()


## Boxplots Sommertage
# TA, rH, WS
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 16
fig, ax1 = plt.subplots(figsize=(10, 6))
sns.boxplot(data=completed_data.loc[completed_data.groupby(completed_data["Datetime"].dt.date)["TA_1_1_2"].transform("max") >= 25, ["RH_1_1_2"]], color="red",
            whis=[10, 90], showmeans=True,
            meanprops={"marker":"x","markerfacecolor":"white", "markeredgecolor":"white"}, ax=ax1)
ax2 = ax1.twinx()
sns.boxplot(data=completed_data.loc[completed_data.groupby(completed_data["Datetime"].dt.date)["TA_1_1_2"].transform("max") >= 25, ["TA_1_1_2", "WS_KMH"]],whis=[10, 90],
            showmeans=True,
            meanprops={"marker":"x","markerfacecolor":"white", "markeredgecolor":"white"}, ax=ax2)
ax1.set_xticks(ticks=[0, 1, 2])
ax1.set_xticklabels(labels=["Relative Humidity (%)", "Air Temperature (°C)", "Windspeed ($km/h$)"])
ax1.set_ylim(0, 100)
ax2.set_ylim(0, 35)
ax1.set_ylabel("Relative Humidity")
ax2.set_ylabel("Airtemperature (°C) / Windspeed ($km/h$)")
ax1.set_yticks([0, 20, 40, 60, 80, 100])
ax2.set_yticks([0, 7, 14, 21, 28, 35])
ax1.grid(False)
ax2.grid(False)
ax1.grid(axis="y")
#plt.savefig("C:\\Users\\linus\\OneDrive\\Dokumente\\Masterarbeit\\Graphen\\Boxplot Sommertage TA rH WS.pdf", format="pdf", bbox_inches="tight")
plt.show()


# Green Roof SWC, Bodentemperatur, Oberflächentemperatur, QG_HFP
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 16
plt.figure(figsize=(10, 6))
sns.boxplot(data=completed_data.loc[completed_data.groupby(completed_data["Datetime"].dt.date)["TA_1_1_2"].transform("max") >= 25, ["SWC_1_1_1", "TS_CS65X_1_1_1","GR_ST", "G_plate_1_1_1"]],
            whis=[10, 90], showmeans = True,
            meanprops={"marker":"x","markerfacecolor":"white", "markeredgecolor":"white"})
plt.grid(True)
plt.xticks(ticks=[0,1,2,3], labels=["Saturation Coefficient\n(%)", "Soil Temperature\n(°C)", "Surface Temperature\n(°C)", "Q$_G$ $_{Heat}$ $_{Flux}$ $_{Plate}$"])
#plt.savefig("C:\\Users\\linus\\OneDrive\\Dokumente\\Masterarbeit\\Graphen\\Boxplot Sommertage Green Roof SWC TS ST QG_HFP.pdf", format="pdf", bbox_inches="tight")
plt.show()


# Hybrid Roof SWC, Bodentemperatur, Oberflächentemperatur, QG_HFP
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 16
plt.figure(figsize=(10, 6))
sns.boxplot(data=completed_data.loc[completed_data.groupby(completed_data["Datetime"].dt.date)["TA_1_1_2"].transform("max") >= 25, ["SWC_1_1_2", "TS_CS65X_1_1_2","HR_ST", "G_plate_1_1_2"]],
            whis=[10, 90], showmeans = True,
            meanprops={"marker":"x","markerfacecolor":"white", "markeredgecolor":"white"})
plt.grid(True)
plt.xticks(ticks=[0,1,2,3], labels=["Saturation Coefficient\n(%)", "Soil Temperature\n(°C)", "Surface Temperature\n(°C)", "Q$_G$ $_{Heat}$ $_{Flux}$ $_{Plate}$\nduring Frost Days"])
#plt.savefig("C:\\Users\\linus\\OneDrive\\Dokumente\\Masterarbeit\\Graphen\\Boxplot Sommertage Hybrid Roof SWC TS ST QG_HFP.pdf", format="pdf")
plt.show()

# Strahlung
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 24
plt.figure(figsize=(17, 10))
sns.boxplot(data=completed_data.loc[completed_data.groupby(completed_data["Datetime"].dt.date)["TA_1_1_2"].transform("max") >= 25,["SW_IN", "LW_IN","SW_OUT_GR_Calculated", "SW_OUT_HR_Calculated", "LW_OUT_GR_Calculated", "LW_OUT_HR_Calculated"]],
            whis=[10, 90], showmeans=True, meanprops={"marker":"x","markerfacecolor":"white", "markeredgecolor":"white"})
plt.grid(True)
plt.ylim(-50, 1000)
plt.ylabel("$W/m^2$")
plt.xticks(ticks=[0,1,2,3,4,5], labels=["SW IN", "LW IN", "SW OUT GR", "SW OUT HR", "LW OUT GR", "LW OUT HR"])
#plt.savefig("C:\\Users\\linus\\OneDrive\\Dokumente\\Masterarbeit\\Graphen\\Boxplot Sommertage Radiation.pdf", format="pdf", bbox_inches="tight")
plt.show()

# Green Roof QE, QH, QG, Q* 
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 20
plt.figure(figsize=(10, 6))
sns.boxplot(data=completed_data.loc[completed_data.groupby(completed_data["Datetime"].dt.date)["TA_1_1_2"].transform("max") >= 25,["Q*_flux_gr", "QE_GR","QH_GR", "QG_GR"]],
            whis=[10, 90], showmeans=True, meanprops={"marker":"x","markerfacecolor":"white", "markeredgecolor":"white"})
plt.grid(True)
plt.ylabel("$W/m^2$")
plt.xticks(ticks=[0,1,2,3], labels=["Q*", "Q$_E$", "Q$_H$", "Q$_G$"])
#plt.savefig("C:\\Users\\linus\\OneDrive\\Dokumente\\Masterarbeit\\Graphen\\Boxplot Sommertage Green Roof Heat Fluxes.pdf", format="pdf", bbox_inches="tight")
plt.show()

# Hybrid Roof QE, QH, QG, Q* 
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 20
plt.figure(figsize=(10, 6))
sns.boxplot(data=completed_data.loc[completed_data.groupby(completed_data["Datetime"].dt.date)["TA_1_1_2"].transform("max") >= 25,["Q*_flux_hr", "QE_HR","QH_HR", "QG_HR"]],
            whis=[10, 90], showmeans=True, meanprops={"marker":"x","markerfacecolor":"white", "markeredgecolor":"white"})
plt.grid(True)
plt.ylabel("$W/m^2$")
plt.xticks(ticks=[0,1,2,3], labels=["Q*", "Q$_E$", "Q$_H$", "Q$_G$"])
#plt.savefig("C:\\Users\\linus\\OneDrive\\Dokumente\\Masterarbeit\\Graphen\\Boxplot Sommertage Hybrid Roof Heat Fluxes.pdf", format="pdf", bbox_inches="tight")
plt.show()

## Mittlere Tagesgänge
# TA, rH, WS
timelabels = ["00:00", "03:00", "06:00", "09:00", "12:00", "15:00", "18:00", "21:00"]
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 24
fig, ax1 = plt.subplots(figsize=(12, 6))
ax1.plot(completed_data.groupby("Time")["RH_1_1_2"].mean().index, completed_data.groupby("Time")["RH_1_1_2"].mean(), linestyle="-", label="relative Humidity", color="dodgerblue", linewidth=2)
ax2 = ax1.twinx()
ax2.plot(completed_data.groupby("Time")["TA_1_1_2"].mean().index, completed_data.groupby("Time")["TA_1_1_2"].mean(), linestyle="--", label="Air Temperature", color="crimson", linewidth=2)
ax2.plot(completed_data.groupby("Time")["WS_KMH"].mean().index, completed_data.groupby("Time")["WS_KMH"].mean(), linestyle="-.", label="Windspeed", color="goldenrod", linewidth=2)
ax1.grid(False)
ax2.grid(False)
ax1.grid(axis="y")
ax1.set_xlabel("Hour of the Day")
ax1.set_ylabel("relative Humidity (%)")
ax2.set_ylabel("Air Temperature (°C) / Windspeed (km/h)")
ax1.set_xlim(completed_data.groupby("Time")["RH_1_1_2"].mean().index.min(), completed_data.groupby("Time")["RH_1_1_2"].mean().index.max())
ax1.set_ylim(55, 80)
ax2.set_ylim(5, 15)
ax1.set_yticks([55, 60, 65, 70, 75, 80])
ax2.set_yticks([5, 7, 9, 11, 13, 15])
ax1.set_xticks(completed_data.groupby("Time")["RH_1_1_2"].mean().index[::6], labels=timelabels, rotation=45, ha="right")
ax1.legend(loc="center", bbox_to_anchor=(0.15, -0.35), ncol = 1, frameon=False)
ax2.legend(loc="center", bbox_to_anchor=(0.75, -0.35), ncol = 2, frameon=False)
#plt.savefig("C:\\Users\\linus\\OneDrive\\Dokumente\\Masterarbeit\\Graphen\\MDC TA rH WS.pdf", format="pdf", bbox_inches="tight")
plt.show()

# Green Roof QE, QH, QG, Q* 
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 24
plt.figure(figsize=(15, 10))
plt.plot(completed_data.groupby("Time")["Q*_flux_gr"].mean().index, completed_data.groupby("Time")["Q*_flux_gr"].mean(), linestyle="-", linewidth=3.2, color="dodgerblue", label="Q*")
plt.plot(completed_data.groupby("Time")["QH_GR"].mean().index, completed_data.groupby("Time")["QH_GR"].mean(), linestyle="--", linewidth=3.2, color="crimson", label="Q$_H$")
plt.plot(completed_data.groupby("Time")["QE_GR"].mean().index, completed_data.groupby("Time")["QE_GR"].mean(), linestyle="-.", linewidth=3.2, color="goldenrod", label="Q$_E$")
plt.plot(completed_data.groupby("Time")["QG_GR"].mean().index, completed_data.groupby("Time")["QG_GR"].mean(), linestyle="-", linewidth=3.2, color="dimgray", label="Q$_G$")
plt.grid(False)
plt.grid(axis="y")
plt.xlabel("Hour of the Day")
plt.ylabel("$W/m^2$")
plt.xlim(completed_data.groupby("Time")["SWC_1_1_1"].mean().index.min(), completed_data.groupby("Time")["SWC_1_1_1"].mean().index.max())
plt.xticks(completed_data.groupby("Time")["SWC_1_1_1"].mean().index[::6], labels=timelabels)
plt.legend(loc="center", bbox_to_anchor=(0.5, -0.15), ncol=4, frameon=False)
#plt.savefig("C:\\Users\\linus\\OneDrive\\Dokumente\\Masterarbeit\\Graphen\\MDC Green Roof Heat Fluxes.pdf", format="pdf", bbox_inches="tight")
plt.show()


subset = completed_data.dropna(subset=["QE_GR", "QG_GR", "QH_GR", "QE_HR", "QG_HR", "QH_HR", "GR_ST", "HR_ST", "Q*_flux_gr", "Q*_rad_gr", "Q*_flux_hr", "Q*_rad_hr", "LW_IN", "SW_IN", "LW_OUT_GR_Calculated", "LW_OUT_HR_Calculated"])

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 24
plt.figure(figsize=(15, 10))
plt.plot(subset.groupby("Time")["Q*_flux_gr"].mean().index, subset.groupby("Time")["Q*_flux_gr"].mean(), linestyle="-", linewidth=3.2, color="dodgerblue", label="Q*")
plt.plot(subset.groupby("Time")["QH_GR"].mean().index, subset.groupby("Time")["QH_GR"].mean(), linestyle="--", linewidth=3.2, color="crimson", label="Q$_H$")
plt.plot(subset.groupby("Time")["QE_GR"].mean().index, subset.groupby("Time")["QE_GR"].mean(), linestyle="-.", linewidth=3.2, color="goldenrod", label="Q$_E$")
plt.plot(subset.groupby("Time")["QG_GR"].mean().index, subset.groupby("Time")["QG_GR"].mean(), linestyle="-", linewidth=3.2, color="dimgray", label="Q$_G$")
plt.grid(False)
plt.grid(axis="y")
plt.xlabel("Hour of the Day")
plt.ylabel("$W/m^2$")
plt.xlim(subset.groupby("Time")["SWC_1_1_1"].mean().index.min(), subset.groupby("Time")["SWC_1_1_1"].mean().index.max())
plt.xticks(subset.groupby("Time")["SWC_1_1_1"].mean().index[::6], labels=timelabels)
plt.legend(loc="center", bbox_to_anchor=(0.5, -0.15), ncol=4, frameon=False)
#plt.savefig("C:\\Users\\linus\\OneDrive\\Dokumente\\Masterarbeit\\Graphen\\MDC Green Roof Heat Fluxes no NaNs.pdf", format="pdf", bbox_inches="tight")
plt.show()

# Hybrid Roof QE, QH, QG, Q* 
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 24
plt.figure(figsize=(15, 10))
plt.plot(completed_data.groupby("Time")["Q*_flux_hr"].mean().index, completed_data.groupby("Time")["Q*_flux_hr"].mean(), linestyle="-", linewidth=3.2, color="dodgerblue", label="Q*")
plt.plot(completed_data.groupby("Time")["QH_HR"].mean().index, completed_data.groupby("Time")["QH_HR"].mean(), linestyle="--", linewidth=3.2, color="crimson", label="Q$_H$")
plt.plot(completed_data.groupby("Time")["QE_HR"].mean().index, completed_data.groupby("Time")["QE_HR"].mean(), linestyle="-.", linewidth=3.2, color="goldenrod", label="Q$_E$")
plt.plot(completed_data.groupby("Time")["QG_HR"].mean().index, completed_data.groupby("Time")["QG_HR"].mean(), linestyle="-", linewidth=3.2, color="dimgray", label="Q$_G$")
plt.plot(completed_data.groupby("Time")["QH_NU"].mean().index, completed_data.groupby("Time")["QH_NU"].mean(), linestyle="--", linewidth=3.2, color="black", label="Q$_H$$_{Nu}$")
plt.grid(False)
plt.grid(axis="y")
plt.xlabel("Hour of the Day")
plt.ylabel("$W/m^2$")
plt.xlim(completed_data.groupby("Time")["SWC_1_1_2"].mean().index.min(), completed_data.groupby("Time")["SWC_1_1_2"].mean().index.max())
plt.xticks(completed_data.groupby("Time")["SWC_1_1_2"].mean().index[::6], labels=timelabels)
plt.legend(loc="center", bbox_to_anchor=(0.5, -0.15), ncol=5, frameon=False)
#plt.savefig("C:\\Users\\linus\\OneDrive\\Dokumente\\Masterarbeit\\Graphen\\MDC Hybrid Roof Heat Fluxes.pdf", format="pdf", bbox_inches="tight")
plt.show()

subset = completed_data.dropna(subset=["QE_GR", "QG_GR", "QH_GR", "QE_HR", "QG_HR", "QH_HR", "GR_ST", "HR_ST", "Q*_flux_gr", "Q*_rad_gr", "Q*_flux_hr", "Q*_rad_hr", "LW_IN", "SW_IN", "LW_OUT_GR_Calculated", "LW_OUT_HR_Calculated"])

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 24
plt.figure(figsize=(15, 10))
plt.plot(subset.groupby("Time")["Q*_flux_hr"].mean().index, subset.groupby("Time")["Q*_flux_hr"].mean(), linestyle="-", linewidth=3.2, color="dodgerblue", label="Q*")
plt.plot(subset.groupby("Time")["QE_HR"].mean().index, subset.groupby("Time")["QE_HR"].mean(), linestyle="--", linewidth=3.2, color="crimson", label="Q$_H$")
plt.plot(subset.groupby("Time")["QH_HR"].mean().index, subset.groupby("Time")["QH_HR"].mean(), linestyle="-.", linewidth=3.2, color="goldenrod", label="Q$_E$")
plt.plot(subset.groupby("Time")["QG_HR"].mean().index, subset.groupby("Time")["QG_HR"].mean(), linestyle="-", linewidth=3.2, color="dimgray", label="Q$_G$")
plt.plot(subset.groupby("Time")["QH_NU"].mean().index, subset.groupby("Time")["QH_NU"].mean(), linestyle="--", linewidth=3.2, color="black", label="Q$_H$$_{Nu}$")
plt.grid(False)
plt.grid(axis="y")
plt.xlabel("Hour of the Day")
plt.ylabel("$W/m^2$")
plt.xlim(subset.groupby("Time")["SWC_1_1_2"].mean().index.min(), subset.groupby("Time")["SWC_1_1_2"].mean().index.max())
plt.xticks(subset.groupby("Time")["SWC_1_1_2"].mean().index[::6], labels=timelabels)
plt.legend(loc="center", bbox_to_anchor=(0.5, -0.15), ncol=5, frameon=False)
#plt.savefig("C:\\Users\\linus\\OneDrive\\Dokumente\\Masterarbeit\\Graphen\\MDC Hybrid Roof Heat Fluxes no NaNs.pdf", format="pdf", bbox_inches="tight")
plt.show()


print(completed_data["Q*_flux_gr"].max())
print(completed_data["Q*_flux_gr"].min())
print(completed_data["QE_GR"].max())
print(completed_data["QG_GR"].max())
print(completed_data["QH_GR"].max())
print(completed_data["QE_GR"].min())
print(completed_data["QG_GR"].min())
print(completed_data["QH_GR"].min())
print(completed_data["Q*_flux_hr"].max())
print(completed_data["Q*_flux_hr"].min())
print(completed_data["QE_HR"].max())
print(completed_data["QG_HR"].max())
print(completed_data["QH_HR"].max())
print(completed_data["QE_HR"].min())
print(completed_data["QG_HR"].min())
print(completed_data["QH_HR"].min())
print((completed_data["QH_GR"].mean()/completed_data["QE_GR"].mean()))
print((completed_data["QH_HR"].mean()/completed_data["QE_HR"].mean()))
print((completed_data[(completed_data["Datetime"].dt.hour >= 10) & 
                      (completed_data["Datetime"].dt.hour <= 16)]["QH_GR"].mean() /
       completed_data[(completed_data["Datetime"].dt.hour >= 10) & 
                      (completed_data["Datetime"].dt.hour <= 16)]["QE_GR"].mean()))

print((completed_data[(completed_data["Datetime"].dt.hour >= 10) & 
                      (completed_data["Datetime"].dt.hour <= 16)]["QH_HR"].mean() /
       completed_data[(completed_data["Datetime"].dt.hour >= 10) & 
                      (completed_data["Datetime"].dt.hour <= 16)]["QE_HR"].mean()))  

print((completed_data[(completed_data["Datetime"].dt.hour <= 10) & 
                      (completed_data["Datetime"].dt.hour >= 16)]["QH_GR"].mean() /
       completed_data[(completed_data["Datetime"].dt.hour <= 10) & 
                      (completed_data["Datetime"].dt.hour >= 16)]["QE_GR"].mean()))

print((completed_data[(completed_data["Datetime"].dt.hour <= 10) & 
                      (completed_data["Datetime"].dt.hour >= 16)]["QH_HR"].mean() /
       completed_data[(completed_data["Datetime"].dt.hour <= 10) & 
                      (completed_data["Datetime"].dt.hour >= 16)]["QE_HR"].mean()))  

# Green Roof SWC, Bodentemperatur, QG_HFP
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 28
fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.plot(completed_data.groupby("Time")["SWC_1_1_1"].mean().index, completed_data.groupby("Time")["SWC_1_1_1"].mean(), linestyle="-", label="Sat. Coeff.", color="dodgerblue", linewidth=3)
ax2 = ax1.twinx()
ax2.plot(completed_data.groupby("Time")["TS_CS65X_1_1_1"].mean().index, completed_data.groupby("Time")["TS_CS65X_1_1_1"].mean(), linestyle="--", label="T$_{Soil}$", color="crimson", linewidth=3)
ax2.plot(completed_data.groupby("Time")["G_plate_1_1_1"].mean().index, completed_data.groupby("Time")["G_plate_1_1_1"].mean(), linestyle="-.", label="Q$_G$ $_{HFP}$", color="goldenrod", linewidth=3)
ax1.grid(False)
ax2.grid(False)
ax2.grid(axis="y")
ax1.set_xlabel("Hour of the Day")
ax1.set_ylabel("Saturation Coefficient (%)")
ax2.set_ylabel("Q$_G$ $_{HFP}$ ($W/m^2$)/ T$_{Soil}$ (°C)")
plt.xlim(completed_data.groupby("Time")["SWC_1_1_1"].mean().index.min(), completed_data.groupby("Time")["SWC_1_1_1"].mean().index.max())
ax1.set_xticks(completed_data.groupby("Time")["SWC_1_1_1"].mean().index[::6], labels=timelabels, rotation=45)
ax1.set_ylim(13.5, 14)
ax2.set_ylim(-5,15)
ax1.set_yticks([13.5, 13.75, 14])
ax2.set_yticks([-5, 0, 5, 10, 15])
ax1.legend(loc="center", bbox_to_anchor=(0.15, -0.4), ncol = 4, frameon=False)
ax2.legend(loc="center", bbox_to_anchor=(0.75, -0.4), ncol = 4, frameon=False)
#plt.savefig("C:\\Users\\linus\\OneDrive\\Dokumente\\Masterarbeit\\Graphen\\MDC Green Roof SWC TS QG HFP.pdf", format="pdf", bbox_inches="tight")
plt.show()

# Hybrid Roof SWC, Bodentemperatur, QG_HFP
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 28
fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.plot(completed_data.groupby("Time")["SWC_1_1_2"].mean().index, completed_data.groupby("Time")["SWC_1_1_2"].mean(), linestyle="-", label="Sat. Coeff.", color="dodgerblue", linewidth=3)
ax2 = ax1.twinx()
ax2.plot(completed_data.groupby("Time")["TS_CS65X_1_1_2"].mean().index, completed_data.groupby("Time")["TS_CS65X_1_1_2"].mean(), linestyle="--", label="T$_{Soil}$", color="crimson", linewidth=3)
ax2.plot(completed_data.groupby("Time")["G_plate_1_1_2"].mean().index, completed_data.groupby("Time")["G_plate_1_1_2"].mean(), linestyle="-.", label="Q$_G$ $_{HFP}$", color="goldenrod", linewidth=3)
ax1.grid(False)
ax2.grid(False)
ax2.grid(axis="y")
ax1.set_xlabel("Hour of the Day")
ax1.set_ylabel("Saturation Coefficient (%)")
ax2.set_ylabel("Q$_G$ $_{HFP}$ ($W/m^2$)/ T$_{Soil}$ (°C)")
plt.xlim(completed_data.groupby("Time")["SWC_1_1_2"].mean().index.min(), completed_data.groupby("Time")["SWC_1_1_2"].mean().index.max())
ax1.set_xticks(completed_data.groupby("Time")["SWC_1_1_2"].mean().index[::6], labels=timelabels, rotation=45)
ax1.set_ylim(62.5, 64)
ax2.set_ylim(-7,13)
ax1.set_yticks([62.5, 63.25, 64])
ax2.set_yticks([-7, -2, 3, 8, 13])
ax1.legend(loc="center", bbox_to_anchor=(0.15, -0.4), ncol = 4, frameon=False)
ax2.legend(loc="center", bbox_to_anchor=(0.75, -0.4), ncol = 4, frameon=False)
#plt.savefig("C:\\Users\\linus\\OneDrive\\Dokumente\\Masterarbeit\\Graphen\\MDC Hybrid Roof SWC TS QG HFP.pdf", format="pdf", bbox_inches="tight")
plt.show()

# ST, SW IN, TA, RH
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 36
fig, ax1 = plt.subplots(figsize=(17,13))
ax1.plot(completed_data.groupby("Time")["RH_1_1_2"].mean().index, completed_data.groupby("Time")["RH_1_1_2"].mean(), linestyle="-", label="rH", color="dodgerblue", linewidth=4)
ax1.plot(completed_data.groupby("Time")["SW_IN"].mean().index, completed_data.groupby("Time")["SW_IN"].mean(), linestyle="--", label="SW$_{IN}$", color="crimson", linewidth=4)
ax2 = ax1.twinx()
ax2.plot(completed_data.groupby("Time")["GR_ST"].mean().index, completed_data.groupby("Time")["GR_ST"].mean(), linestyle="-.", label="ST$_{GR}$", color="goldenrod", linewidth=4)
ax2.plot(completed_data.groupby("Time")["HR_ST"].mean().index, completed_data.groupby("Time")["HR_ST"].mean(), linestyle="-", label="ST$_{HR}$", color="dimgray", linewidth=4)
ax2.plot(completed_data.groupby("Time")["TA_1_1_2"].mean().index, completed_data.groupby("Time")["TA_1_1_2"].mean(), linestyle="--", label="T$_{Air}$", color="black", linewidth=4)
plt.xlim(completed_data.groupby("Time")["SW_IN"].mean().index.min(), completed_data.groupby("Time")["SW_IN"].mean().index.max())
plt.xticks(completed_data.groupby("Time")["SW_IN"].mean().index[::6], labels=timelabels)
ax1.legend(loc="center", bbox_to_anchor=(0.125, -0.1), ncol = 2, frameon=False)
ax2.legend(loc="center", bbox_to_anchor=(0.75, -0.1), ncol = 3, frameon=False)
ax1.set_ylabel("rel. Humidity (%) / Radiation ($W/m^2$)")
ax2.set_ylabel("Temperature (°C)")
plt.grid()
#plt.savefig("C:\\Users\\linus\\OneDrive\\Dokumente\\Masterarbeit\\Graphen\\MDC ST SW TA RH.pdf", format="pdf", bbox_inches="tight")
plt.show()

# QE GR vs TA, rH, SW IN, LW IN, and Windspeed
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 36
fig, ax1 = plt.subplots(figsize=(17, 13))
ax1.plot(completed_data.groupby("Time")["RH_1_1_2"].mean().index,
         completed_data.groupby("Time")["RH_1_1_2"].mean(),
         label="rH", color="dodgerblue", linestyle="-", linewidth=3.5)
ax2 = ax1.twinx()
ax2.plot(completed_data.groupby("Time")["TA_1_1_2"].mean().index,
         completed_data.groupby("Time")["TA_1_1_2"].mean(),
         label="T$_{Air}$", color="crimson", linestyle="--", linewidth=3.5)
ax3 = ax1.twinx()
ax3.spines["right"].set_position(("outward", 105))
ax3.plot(completed_data.groupby("Time")["QE_GR"].mean().index,
         completed_data.groupby("Time")["QE_GR"].mean(),
         label="Q$_E$", color="goldenrod", linestyle="-.", linewidth=3.5)
ax4 = ax1.twinx()
ax4.spines["left"].set_position(("outward", 105))
ax4.plot(completed_data.groupby("Time")["SW_IN"].mean().index,
         completed_data.groupby("Time")["SW_IN"].mean(),
         label="SW$_{IN}$", color="dimgray", linestyle="-", linewidth=3.5)
ax4.plot(completed_data.groupby("Time")["LW_IN"].mean().index,
         completed_data.groupby("Time")["LW_IN"].mean(),
         label="LW$_{IN}$", color="black", linestyle="--", linewidth=3.5)
ax4.yaxis.set_label_position("left")
ax4.yaxis.set_ticks_position("left")
ax3.set_ylabel("Q$_E$ (W/m²)")
ax1.set_ylabel("Relative Humidity (%)")
ax4.set_ylabel("Radiation (W/m²)"),
ax2.set_ylabel("Air Temperature (°C)")
ax1.yaxis.set_label_coords(-0.055, 0.5)
ax2.yaxis.set_label_coords(1.055,0.5)
ax3.yaxis.set_label_coords(1.2,0.5)
ax4.yaxis.set_label_coords(-0.16, 0.5)
ax1.legend(loc="center", bbox_to_anchor=(-0.05, -0.2), ncol=1, frameon=False)
ax2.legend(loc="center", bbox_to_anchor=(0.2, -0.2), ncol=1, frameon=False)
ax3.legend(loc="center", bbox_to_anchor=(0.45, -0.2), ncol=1, frameon=False)
ax4.legend(loc="center", bbox_to_anchor=(0.85, -0.2), ncol=2, frameon=False)
ax3.set_ylim(-15,35)
ax3.set_yticks([-15,-5,5,15,25,35])
ax1.set_ylim(55,75)
ax1.set_yticks([55,59,63,67,71,75])
ax4.set_ylim(0,350)
ax4.set_yticks([0,70,140,210,280,350])
ax2.set_ylim(10,15)
ax2.set_yticks([10,11,12,13,14,15])
plt.xlim(completed_data.groupby("Time")["QE_GR"].mean().index.min(),
         completed_data.groupby("Time")["QE_GR"].mean().index.max())
ax1.set_xticks(completed_data.groupby("Time")["QE_GR"].mean().index[::6], labels=timelabels, rotation=45)
plt.grid()
#plt.savefig("C:\\Users\\linus\\OneDrive\\Dokumente\\Masterarbeit\\Graphen\\MDC Green Roof QE TA RH SW LW.pdf", format="pdf", bbox_inches="tight")
plt.show()

# QH GR vs TA, rH, SW IN, LW IN, and Windspeed
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 36
fig, ax1 = plt.subplots(figsize=(17, 13))
ax1.plot(completed_data.groupby("Time")["RH_1_1_2"].mean().index,
         completed_data.groupby("Time")["RH_1_1_2"].mean(),
         label="rH", color="dodgerblue", linewidth=3.5)
ax2 = ax1.twinx()
ax2.plot(completed_data.groupby("Time")["QH_GR"].mean().index,
         completed_data.groupby("Time")["QH_GR"].mean(),
         label="Q$_H$", color="crimson", linestyle="--", linewidth=3.5)
ax4 = ax1.twinx()
ax4.spines["left"].set_position(("outward", 105))
ax4.plot(completed_data.groupby("Time")["TA_1_1_2"].mean().index,
         completed_data.groupby("Time")["TA_1_1_2"].mean(),
         label="T$_{Air}$", color="black", linestyle="--", linewidth=3.5)
ax4.yaxis.set_label_position("left")
ax4.yaxis.set_ticks_position("left")
ax3 = ax1.twinx()
ax3.spines["right"].set_position(("outward", 105))
ax3.plot(completed_data.groupby("Time")["SW_IN"].mean().index,
         completed_data.groupby("Time")["SW_IN"].mean(),
         label="SW$_{IN}$", color="goldenrod", linestyle="-.", linewidth=3.5)
ax3.plot(completed_data.groupby("Time")["LW_IN"].mean().index,
         completed_data.groupby("Time")["LW_IN"].mean(),
         label="LW$_{IN}$", color="dimgray", linestyle="-", linewidth=3.5)
ax2.set_ylabel("Q$_H$ (W/m²)")
ax1.set_ylabel("Relative Humidity (%)")
ax3.set_ylabel("Radiation (W/m²)"),
ax4.set_ylabel("Air Temperature (°C)")
ax1.yaxis.set_label_coords(-0.055, 0.5)
ax2.yaxis.set_label_coords(1.055,0.5)
ax3.yaxis.set_label_coords(1.2,0.5)
ax4.yaxis.set_label_coords(-0.16, 0.5)
ax1.legend(loc="center", bbox_to_anchor=(0, -0.2), ncol=1, frameon=False)
ax2.legend(loc="center", bbox_to_anchor=(0.25, -0.2), ncol=1, frameon=False)
ax3.legend(loc="center", bbox_to_anchor=(0.65, -0.2), ncol=2, frameon=False)
ax4.legend(loc="center", bbox_to_anchor=(1.05, -0.2), ncol=1, frameon=False)
ax2.set_ylim(-30,50)
ax2.set_yticks([-30,-14,2,18,34,50])
ax1.set_ylim(55,75)
ax1.set_yticks([55,59,63,67,71,75])
ax3.set_ylim(0,350)
ax3.set_yticks([0,70,140,210,280,350])
ax4.set_ylim(10,15)
ax4.set_yticks([10,11,12,13,14,15])
plt.xlim(completed_data.groupby("Time")["QE_GR"].mean().index.min(),
         completed_data.groupby("Time")["QE_GR"].mean().index.max())
ax1.set_xticks(completed_data.groupby("Time")["QE_GR"].mean().index[::6], labels=timelabels, rotation=45)
plt.grid()
#plt.savefig("C:\\Users\\linus\\OneDrive\\Dokumente\\Masterarbeit\\Graphen\\MDC Green Roof QH TA RH SW LW.pdf", format="pdf", bbox_inches="tight")
plt.show()

# QG GR vs TA, rH, SW IN, LW IN, and Windspeed
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 36
fig, ax1 = plt.subplots(figsize=(17, 13))
ax1.plot(completed_data.groupby("Time")["RH_1_1_2"].mean().index,
         completed_data.groupby("Time")["RH_1_1_2"].mean(),
         label="rH", color="dodgerblue", linestyle="-", linewidth=3.5)
ax2 = ax1.twinx()
ax2.spines["right"].set_position(("outward", 105))
ax2.plot(completed_data.groupby("Time")["SW_IN"].mean().index,
         completed_data.groupby("Time")["SW_IN"].mean(),
         label="SW$_{IN}$", color="crimson", linestyle="--", linewidth=3.5)
ax2.plot(completed_data.groupby("Time")["LW_IN"].mean().index,
         completed_data.groupby("Time")["LW_IN"].mean(),
         label="LW$_{IN}$", color="goldenrod", linestyle="-.", linewidth=3.5)
ax3 = ax1.twinx()
ax3.spines["left"].set_position(("outward", 105))
ax3.yaxis.set_label_position("left")
ax3.yaxis.set_ticks_position("left")
ax3.plot(completed_data.groupby("Time")["QG_GR"].mean().index,
         completed_data.groupby("Time")["QG_GR"].mean(),
         label="Q$_G$", color="dimgray", linestyle="-", linewidth=3.5)
ax4 = ax1.twinx()
ax4.plot(completed_data.groupby("Time")["TA_1_1_2"].mean().index,
         completed_data.groupby("Time")["TA_1_1_2"].mean(),
         label="T$_{Air}$", color="black", linestyle="--", linewidth=3.5)
ax3.set_ylabel("Q$_G$ (W/m²)")
ax1.set_ylabel("Relative Humidity (%)")
ax2.set_ylabel("Radiation (W/m²)"),
ax4.set_ylabel("Air Temperature (°C)")
ax1.yaxis.set_label_coords(-0.055, 0.5)
ax2.yaxis.set_label_coords(1.055,0.5)
ax3.yaxis.set_label_coords(1.2,0.5)
ax4.yaxis.set_label_coords(-0.16, 0.5)
ax1.legend(loc="center", bbox_to_anchor=(0, -0.2), ncol=1, frameon=False)
ax2.legend(loc="center", bbox_to_anchor=(0.4, -0.2), ncol=2, frameon=False)
ax3.legend(loc="center", bbox_to_anchor=(0.8, -0.2), ncol=1, frameon=False)
ax4.legend(loc="center", bbox_to_anchor=(1.05, -0.2), ncol=1, frameon=False)
ax3.set_ylim(-12,18)
ax3.set_yticks([-12,-6,0,6,12,18])
ax1.set_ylim(55,75)
ax1.set_yticks([55,59,63,67,71,75])
ax2.set_ylim(0,350)
ax2.set_yticks([0,70,140,210,280,350])
ax4.set_ylim(10,15)
ax4.set_yticks([10,11,12,13,14,15])
plt.xlim(completed_data.groupby("Time")["QE_GR"].mean().index.min(),
         completed_data.groupby("Time")["QE_GR"].mean().index.max())
ax1.set_xticks(completed_data.groupby("Time")["QE_GR"].mean().index[::6], labels=timelabels, rotation=45)
plt.grid()
#plt.savefig("C:\\Users\\linus\\OneDrive\\Dokumente\\Masterarbeit\\Graphen\\MDC Green Roof QG TA RH SW LW.pdf", format="pdf", bbox_inches="tight")
plt.show()


# QE HR vs TA, rH, SW IN, LW IN, and Windspeed
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 36
fig, ax1 = plt.subplots(figsize=(17, 13))
ax1.plot(completed_data.groupby("Time")["RH_1_1_2"].mean().index,
         completed_data.groupby("Time")["RH_1_1_2"].mean(),
         label="rH", color="dodgerblue", linestyle="-", linewidth=3.5)
ax2 = ax1.twinx()
ax2.plot(completed_data.groupby("Time")["TA_1_1_2"].mean().index,
         completed_data.groupby("Time")["TA_1_1_2"].mean(),
         label="T$_{Air}$", color="crimson", linestyle="--", linewidth=3.5)
ax3 = ax1.twinx()
ax3.spines["right"].set_position(("outward", 105))
ax3.plot(completed_data.groupby("Time")["QE_HR"].mean().index,
         completed_data.groupby("Time")["QE_HR"].mean(),
         label="Q$_E$", color="goldenrod", linestyle="-.", linewidth=3.5)
ax4 = ax1.twinx()
ax4.spines["left"].set_position(("outward", 105))
ax4.yaxis.set_label_position("left")
ax4.yaxis.set_ticks_position("left")
ax4.plot(completed_data.groupby("Time")["SW_IN"].mean().index,
         completed_data.groupby("Time")["SW_IN"].mean(),
         label="SW$_{IN}$", color="dimgray", linestyle="-", linewidth=3.5)
ax4.plot(completed_data.groupby("Time")["LW_IN"].mean().index,
         completed_data.groupby("Time")["LW_IN"].mean(),
         label="LW$_{IN}$", color="black", linestyle="--", linewidth=3.5)
ax1.set_ylabel("Relative Humidity (%)")
ax2.set_ylabel("Air Temperature (°C)")
ax3.set_ylabel("Q$_E$ (W/m²)")
ax4.set_ylabel("Radiation (W/m²)")
ax1.yaxis.set_label_coords(-0.055, 0.5)
ax2.yaxis.set_label_coords(1.055,0.5)
ax3.yaxis.set_label_coords(1.2,0.5)
ax4.yaxis.set_label_coords(-0.18, 0.5)
ax1.legend(loc="center", bbox_to_anchor=(0, -0.2), ncol=1, frameon=False)
ax2.legend(loc="center", bbox_to_anchor=(0.25, -0.2), ncol=1, frameon=False)
ax3.legend(loc="center", bbox_to_anchor=(0.5, -0.2), ncol=1, frameon=False)
ax4.legend(loc="center", bbox_to_anchor=(0.90, -0.2), ncol=2, frameon=False)
ax1.set_ylim(55,75)
ax1.set_yticks([55,59,63,67,71,75])
ax2.set_ylim(10,15)
ax2.set_yticks([10,11,12,13,14,15])
ax3.set_ylim(-15,115)
ax3.set_yticks([-15,11,37,63,89,115])
ax4.set_ylim(0,350)
ax4.set_yticks([0,70,140,210,280,350])
plt.xlim(completed_data.groupby("Time")["QE_HR"].mean().index.min(),
         completed_data.groupby("Time")["QE_HR"].mean().index.max())
ax1.set_xticks(completed_data.groupby("Time")["QE_HR"].mean().index[::6], labels=timelabels, rotation=45)
plt.grid()
#plt.savefig("C:\\Users\\linus\\OneDrive\\Dokumente\\Masterarbeit\\Graphen\\MDC Hybrid Roof QE TA RH SW LW.pdf", format="pdf", bbox_inches="tight")
plt.show()

# QH HR vs TA, rH, SW IN, LW IN, and Windspeed
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 36
fig, ax1 = plt.subplots(figsize=(17, 13))
ax1.plot(completed_data.groupby("Time")["RH_1_1_2"].mean().index,
         completed_data.groupby("Time")["RH_1_1_2"].mean(),
         label="rH", color="dodgerblue", linewidth=3.5, linestyle="-")
ax2 = ax1.twinx()
ax2.plot(completed_data.groupby("Time")["QH_HR"].mean().index,
         completed_data.groupby("Time")["QH_HR"].mean(),
         label="Q$_H$", color="crimson", linewidth=3.5, linestyle="--")
ax3 = ax1.twinx()
ax3.spines["right"].set_position(("outward", 120))
ax3.plot(completed_data.groupby("Time")["TA_1_1_2"].mean().index,
         completed_data.groupby("Time")["TA_1_1_2"].mean(),
         label="T$_{Air}$", color="goldenrod", linewidth=3.5, linestyle="-.")
ax4 = ax1.twinx()
ax4.spines["left"].set_position(("outward", 105))
ax4.yaxis.set_label_position("left")
ax4.yaxis.set_ticks_position("left")
ax4.plot(completed_data.groupby("Time")["SW_IN"].mean().index,
         completed_data.groupby("Time")["SW_IN"].mean(),
         label="SW$_{IN}$", color="dimgray", linewidth=3.5, linestyle="-")
ax4.plot(completed_data.groupby("Time")["LW_IN"].mean().index,
         completed_data.groupby("Time")["LW_IN"].mean(),
         label="LW$_{IN}$", color="black", linewidth=3.5, linestyle="--")
ax2.set_ylabel("Q$_H$ (W/m²)")
ax1.set_ylabel("Relative Humidity (%)")
ax4.set_ylabel("Radiation (W/m²)"),
ax3.set_ylabel("Air Temperature (°C)")
ax1.yaxis.set_label_coords(-0.055, 0.5)
ax2.yaxis.set_label_coords(1.08,0.5)
ax3.yaxis.set_label_coords(1.2,0.5)
ax4.yaxis.set_label_coords(-0.18, 0.5)
ax1.legend(loc="center", bbox_to_anchor=(0, -0.2), ncol=1, frameon=False)
ax2.legend(loc="center", bbox_to_anchor=(0.25, -0.2), ncol=1, frameon=False)
ax3.legend(loc="center", bbox_to_anchor=(0.5, -0.2), ncol=1, frameon=False)
ax4.legend(loc="center", bbox_to_anchor=(0.9, -0.2), ncol=2, frameon=False)
ax2.set_ylim(-50,-10)
ax2.set_yticks([-50,-42,-34,-26,-18,-10])
ax1.set_ylim(55,75)
ax1.set_yticks([55,59,63,67,71,75])
ax4.set_ylim(0,350)
ax4.set_yticks([0,70,140,210,280,350])
ax3.set_ylim(10,15)
ax3.set_yticks([10,11,12,13,14,15])
plt.xlim(completed_data.groupby("Time")["QE_HR"].mean().index.min(),
         completed_data.groupby("Time")["QE_HR"].mean().index.max())
ax1.set_xticks(completed_data.groupby("Time")["QE_HR"].mean().index[::6], labels=timelabels, rotation=45)
plt.grid()
#plt.savefig("C:\\Users\\linus\\OneDrive\\Dokumente\\Masterarbeit\\Graphen\\MDC Hybrid Roof QH TA RH SW LW.pdf", format="pdf", bbox_inches="tight")
plt.show()

# QG HR vs TA, rH, SW IN, LW IN, and Windspeed
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 36
fig, ax1 = plt.subplots(figsize=(17, 13))
ax1.plot(completed_data.groupby("Time")["RH_1_1_2"].mean().index,
         completed_data.groupby("Time")["RH_1_1_2"].mean(),
         label="rH", color="dodgerblue", linewidth=3.5, linestyle="-")
ax2 = ax1.twinx()
ax2.plot(completed_data.groupby("Time")["SW_IN"].mean().index,
         completed_data.groupby("Time")["SW_IN"].mean(),
         label="SW$_{IN}$", color="crimson", linewidth=3.5, linestyle="--")
ax2.plot(completed_data.groupby("Time")["LW_IN"].mean().index,
         completed_data.groupby("Time")["LW_IN"].mean(),
         label="LW$_{IN}$", color="goldenrod", linewidth=3.5, linestyle="-.")
ax3 = ax1.twinx()
ax3.plot(completed_data.groupby("Time")["QG_HR"].mean().index,
         completed_data.groupby("Time")["QG_HR"].mean(),
         label="Q$_G$", color="dimgray", linewidth=3.5, linestyle="-")
ax3.spines["right"].set_position(("outward", 120))
ax4 = ax1.twinx()
ax4.spines["left"].set_position(("outward", 105))
ax4.plot(completed_data.groupby("Time")["TA_1_1_2"].mean().index,
         completed_data.groupby("Time")["TA_1_1_2"].mean(),
         label="T$_{Air}$", color="black", linewidth=3.5, linestyle="--")
ax4.yaxis.set_label_position("left")
ax4.yaxis.set_ticks_position("left")
ax3.set_ylabel("Q$_G$ (W/m²)")
ax1.set_ylabel("Relative Humidity (%)")
ax2.set_ylabel("Radiation (W/m²)"),
ax4.set_ylabel("Air Temperature (°C)")
ax1.yaxis.set_label_coords(-0.055, 0.5)
ax2.yaxis.set_label_coords(1.08,0.5)
ax3.yaxis.set_label_coords(1.18,0.5)
ax4.yaxis.set_label_coords(-0.16, 0.5)
ax1.legend(loc="center", bbox_to_anchor=(0, -0.2), ncol=1, frameon=False)
ax2.legend(loc="center", bbox_to_anchor=(0.4, -0.2), ncol=2, frameon=False)
ax3.legend(loc="center", bbox_to_anchor=(0.8, -0.2), ncol=1, frameon=False)
ax4.legend(loc="center", bbox_to_anchor=(1.05, -0.2), ncol=1, frameon=False)
ax3.set_ylim(-26,24)
ax3.set_yticks([-24,-16,-6,4,14,26])
ax1.set_ylim(55,75)
ax1.set_yticks([55,59,63,67,71,75])
ax2.set_ylim(0,350)
ax2.set_yticks([0,70,140,210,280,350])
ax4.set_ylim(10,15)
ax4.set_yticks([10,11,12,13,14,15])
plt.xlim(completed_data.groupby("Time")["QE_HR"].mean().index.min(),
         completed_data.groupby("Time")["QE_HR"].mean().index.max())
ax1.set_xticks(completed_data.groupby("Time")["QE_HR"].mean().index[::6], labels=timelabels, rotation=45)
plt.grid()
#plt.savefig("C:\\Users\\linus\\OneDrive\\Dokumente\\Masterarbeit\\Graphen\\MDC Hybrid Roof QG TA RH SW LW.pdf", format="pdf", bbox_inches="tight")
plt.show()

# Strahlungen
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 24
plt.figure(figsize=(15, 10))
plt.plot(completed_data.groupby("Time")["Q*_rad_gr"].mean().index, completed_data.groupby("Time")["Q*_rad_gr"].mean(), linestyle="-", label="Q*$_{GR}$")
plt.plot(completed_data.groupby("Time")["Q*_rad_hr"].mean().index, completed_data.groupby("Time")["Q*_rad_hr"].mean(), linestyle="-", label="Q*$_{HR}$")
plt.plot(completed_data.groupby("Time")["SW_IN"].mean().index, completed_data.groupby("Time")["SW_IN"].mean(), linestyle="--", label="SW IN")
plt.plot(completed_data.groupby("Time")["LW_IN"].mean().index, completed_data.groupby("Time")["LW_IN"].mean(), linestyle="--", label="LW IN")
plt.plot(completed_data.groupby("Time")["SW_OUT_GR_Calculated"].mean().index, (completed_data.groupby("Time")["SW_OUT_GR_Calculated"].mean() * (-1)), linestyle="-.", label="SW OUT GR")
plt.plot(completed_data.groupby("Time")["LW_OUT_GR_Calculated"].mean().index, (completed_data.groupby("Time")["LW_OUT_GR_Calculated"].mean() * (-1)), linestyle=":", label="LW OUT GR")
plt.plot(completed_data.groupby("Time")["SW_OUT_HR_Calculated"].mean().index, (completed_data.groupby("Time")["SW_OUT_HR_Calculated"].mean() * (-1)), linestyle="-.", label="SW OUT HR")
plt.plot(completed_data.groupby("Time")["LW_OUT_HR_Calculated"].mean().index, (completed_data.groupby("Time")["LW_OUT_HR_Calculated"].mean() * (-1)), linestyle=":", label="LW OUT HR")
plt.grid(False)
plt.grid(axis="y")
plt.xlabel("Hour of the Day")
plt.ylabel("$W/m^2$")
plt.xlim(completed_data.groupby("Time")["SW_IN"].mean().index.min(), completed_data.groupby("Time")["SW_IN"].mean().index.max())
plt.xticks(completed_data.groupby("Time")["SW_IN"].mean().index[::6], labels=timelabels)
plt.legend(loc="center", bbox_to_anchor=(0.5, -0.2), ncol=4, frameon=False)
#plt.savefig("C:\\Users\\linus\\OneDrive\\Dokumente\\Masterarbeit\\Graphen\\MDC Strahlung.pdf", format="pdf", bbox_inches="tight")
plt.show()

subset = completed_data.dropna(subset=["QE_GR", "QG_GR", "QH_GR", "QE_HR", "QG_HR", "QH_HR", "GR_ST", "HR_ST", "Q*_flux_gr", "Q*_rad_gr", "Q*_flux_hr", "Q*_rad_hr", "LW_IN", "SW_IN", "LW_OUT_GR_Calculated", "LW_OUT_HR_Calculated"])

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 24
plt.figure(figsize=(15, 10))
plt.plot(subset.groupby("Time")["Q*_rad_gr"].mean().index, subset.groupby("Time")["Q*_rad_gr"].mean(), linestyle="-", label="Q*$_{GR}$")
plt.plot(subset.groupby("Time")["Q*_rad_hr"].mean().index, subset.groupby("Time")["Q*_rad_hr"].mean(), linestyle="-", label="Q*$_{HR}$")
plt.plot(subset.groupby("Time")["SW_IN"].mean().index, subset.groupby("Time")["SW_IN"].mean(), linestyle="--", label="SW IN")
plt.plot(subset.groupby("Time")["LW_IN"].mean().index, subset.groupby("Time")["LW_IN"].mean(), linestyle="--", label="LW IN")
plt.plot(subset.groupby("Time")["SW_OUT_GR_Calculated"].mean().index, (subset.groupby("Time")["SW_OUT_GR_Calculated"].mean() * (-1)), linestyle="-.", label="SW OUT GR")
plt.plot(subset.groupby("Time")["LW_OUT_GR_Calculated"].mean().index, (subset.groupby("Time")["LW_OUT_GR_Calculated"].mean() * (-1)), linestyle=":", label="LW OUT GR")
plt.plot(subset.groupby("Time")["SW_OUT_HR_Calculated"].mean().index, (subset.groupby("Time")["SW_OUT_HR_Calculated"].mean() * (-1)), linestyle="-.", label="SW OUT HR")
plt.plot(subset.groupby("Time")["LW_OUT_HR_Calculated"].mean().index, (subset.groupby("Time")["LW_OUT_HR_Calculated"].mean() * (-1)), linestyle=":", label="LW OUT HR")
plt.grid(False)
plt.grid(axis="y")
plt.xlabel("Hour of the Day")
plt.ylabel("$W/m^2$")
plt.xlim(subset.groupby("Time")["SW_IN"].mean().index.min(), subset.groupby("Time")["SW_IN"].mean().index.max())
plt.xticks(subset.groupby("Time")["SW_IN"].mean().index[::6], labels=timelabels)
plt.legend(loc="center", bbox_to_anchor=(0.5, -0.2), ncol=4, frameon=False)
#plt.savefig("C:\\Users\\linus\\OneDrive\\Dokumente\\Masterarbeit\\Graphen\\MDC Strahlung no NaNs.pdf", format="pdf", bbox_inches="tight")
plt.show()


## Mittlere Tagesgänge Heiztage
# TA, rH, WS
timelabels = ["", "03:00", "06:00", "09:00", "12:00", "15:00", "18:00", "21:00"]
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 20
fig, ax1 = plt.subplots(figsize=(12, 6))
ax1.plot(completed_data.loc[(completed_data.groupby(completed_data["Datetime"].dt.date)["TA_1_1_2"].transform("mean") < 12)].groupby("Time")["RH_1_1_2"].mean().index, completed_data.loc[completed_data.groupby(completed_data["Datetime"].dt.date)["TA_1_1_2"].transform("mean") < 12].groupby("Time")["RH_1_1_2"].mean(), linestyle="--", label="relative Humidity")
ax2 = ax1.twinx()
ax2.plot(completed_data.loc[(completed_data.groupby(completed_data["Datetime"].dt.date)["TA_1_1_2"].transform("mean") < 12)].groupby("Time")["TA_1_1_2"].mean().index, completed_data.loc[completed_data.groupby(completed_data["Datetime"].dt.date)["TA_1_1_2"].transform("mean") < 12].groupby("Time")["TA_1_1_2"].mean(), linestyle="-", label="Air Temperature", color="orange")
ax2.plot(completed_data.loc[(completed_data.groupby(completed_data["Datetime"].dt.date)["TA_1_1_2"].transform("mean") < 12)].groupby("Time")["WS"].mean().index, completed_data.loc[completed_data.groupby(completed_data["Datetime"].dt.date)["TA_1_1_2"].transform("mean") < 12].groupby("Time")["WS"].mean(), linestyle="-.", label="Windspeed", color="forestgreen")
ax1.grid(False)
ax2.grid(False)
ax1.grid(axis="y")
ax1.set_xlabel("Hour of the Day")
ax1.set_ylabel("relative Humidity (%)")
ax2.set_ylabel("Air Temperature (°C) / Windspeed ($km/h$)")
ax1.set_xlim(completed_data.groupby("Time")["RH_1_1_2"].mean().index.min(), completed_data.groupby("Time")["RH_1_1_2"].mean().index.max())
ax1.set_ylim(55, 70)
ax2.set_ylim(0, 8)
ax1.set_yticks([55, 60, 65, 70])
ax2.set_yticks([0, 3, 6, 9])
plt.xticks(completed_data.groupby("Time")["RH_1_1_2"].mean().index[::6], labels=timelabels)
ax1.legend(loc="center", bbox_to_anchor=(0.2, -0.2), ncol = 1, frameon=False)
ax2.legend(loc="center", bbox_to_anchor=(0.7, -0.2), ncol = 2, frameon=False)
#plt.savefig("C:\\Users\\linus\\OneDrive\\Dokumente\\Masterarbeit\\Graphen\\MDC Heiztage TA rH WS.pdf", format="pdf", bbox_inches="tight")
plt.show()


# Green Roof QE, QH, QG, Q* 
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 20
plt.figure(figsize=(12, 6))
plt.plot(completed_data.loc[(completed_data.groupby(completed_data["Datetime"].dt.date)["TA_1_1_2"].transform("mean") < 12)].groupby("Time")["Q*_flux_gr"].mean().index,
         completed_data.loc[(completed_data.groupby(completed_data["Datetime"].dt.date)["TA_1_1_2"].transform("mean") < 12)].groupby("Time")["Q*_flux_gr"].mean(), label="Q*")
plt.plot(completed_data.loc[(completed_data.groupby(completed_data["Datetime"].dt.date)["TA_1_1_2"].transform("mean") < 12)].groupby("Time")["QE_GR"].mean().index,
         completed_data.loc[(completed_data.groupby(completed_data["Datetime"].dt.date)["TA_1_1_2"].transform("mean") < 12)].groupby("Time")["QE_GR"].mean(), label="Q$_E$", color="forestgreen")
plt.plot(completed_data.loc[(completed_data.groupby(completed_data["Datetime"].dt.date)["TA_1_1_2"].transform("mean") < 12)].groupby("Time")["QG_GR"].mean().index,
         completed_data.loc[(completed_data.groupby(completed_data["Datetime"].dt.date)["TA_1_1_2"].transform("mean") < 12)].groupby("Time")["QG_GR"].mean(), label="Q$_G$", color="orange")
plt.plot(completed_data.loc[(completed_data.groupby(completed_data["Datetime"].dt.date)["TA_1_1_2"].transform("mean") < 12)].groupby("Time")["QH_GR"].mean().index,
         completed_data.loc[(completed_data.groupby(completed_data["Datetime"].dt.date)["TA_1_1_2"].transform("mean") < 12)].groupby("Time")["QH_GR"].mean(), label="Q$_H$", color="red")
plt.grid(False)
plt.grid(axis="y")
plt.xlabel("Hour of the Day")
plt.ylabel("$W/m^2$")
plt.xlim(completed_data.groupby("Time")["RH_1_1_2"].mean().index.min(), completed_data.groupby("Time")["RH_1_1_2"].mean().index.max())
plt.xticks(completed_data.groupby("Time")["RH_1_1_2"].mean().index[::6], labels=timelabels)
plt.legend(loc="center", bbox_to_anchor=(0.5, -0.2), ncol = 4, frameon=False)
#plt.savefig("C:\\Users\\linus\\OneDrive\\Dokumente\\Masterarbeit\\Graphen\\MDC Heiztage Green Roof Heat Fluxes.pdf", format="pdf", bbox_inches="tight")
plt.show()

# Hybrid Roof QE, QH, QG, Q* 
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 20
plt.figure(figsize=(12, 6))
plt.plot(completed_data.loc[(completed_data.groupby(completed_data["Datetime"].dt.date)["TA_1_1_2"].transform("mean") < 12)].groupby("Time")["Q*_flux_hr"].mean().index,
         completed_data.loc[(completed_data.groupby(completed_data["Datetime"].dt.date)["TA_1_1_2"].transform("mean") < 12)].groupby("Time")["Q*_flux_hr"].mean(), linestyle="-", label="Q*")
plt.plot(completed_data.loc[(completed_data.groupby(completed_data["Datetime"].dt.date)["TA_1_1_2"].transform("mean") < 12)].groupby("Time")["QE_HR"].mean().index,
         completed_data.loc[(completed_data.groupby(completed_data["Datetime"].dt.date)["TA_1_1_2"].transform("mean") < 12)].groupby("Time")["QE_HR"].mean(), label="Q$_E$", color="forestgreen")
plt.plot(completed_data.loc[(completed_data.groupby(completed_data["Datetime"].dt.date)["TA_1_1_2"].transform("mean") < 12)].groupby("Time")["QG_HR"].mean().index,
         completed_data.loc[(completed_data.groupby(completed_data["Datetime"].dt.date)["TA_1_1_2"].transform("mean") < 12)].groupby("Time")["QG_HR"].mean(), label="Q$_G$", color="orange")
plt.plot(completed_data.loc[(completed_data.groupby(completed_data["Datetime"].dt.date)["TA_1_1_2"].transform("mean") < 12)].groupby("Time")["QH_HR"].mean().index,
         completed_data.loc[(completed_data.groupby(completed_data["Datetime"].dt.date)["TA_1_1_2"].transform("mean") < 12)].groupby("Time")["QH_HR"].mean(), label="Q$_H$", color="red")
plt.plot(completed_data.loc[(completed_data.groupby(completed_data["Datetime"].dt.date)["TA_1_1_2"].transform("mean") < 12)].groupby("Time")["QH_NU"].mean().index,
         completed_data.loc[(completed_data.groupby(completed_data["Datetime"].dt.date)["TA_1_1_2"].transform("mean") < 12)].groupby("Time")["QH_NU"].mean(), label="Q$_H$ $_{Nu}$", color="lime")
plt.grid(False)
plt.grid(axis="y")
plt.xlabel("Hour of the Day")
plt.ylabel("$W/m^2$")
plt.xlim(completed_data.groupby("Time")["RH_1_1_2"].mean().index.min(), completed_data.groupby("Time")["RH_1_1_2"].mean().index.max())
plt.xticks(completed_data.groupby("Time")["RH_1_1_2"].mean().index[::6], labels=timelabels)
plt.legend(loc="center", bbox_to_anchor=(0.5, -0.2), ncol = 4, frameon=False)
#plt.savefig("C:\\Users\\linus\\OneDrive\\Dokumente\\Masterarbeit\\Graphen\\MDC Heiztage Hybrid Roof Heat Fluxes.pdf", format="pdf", bbox_inches="tight")
plt.show()

# Green Roof SWC, Bodentemperatur, Oberflächentemperatur, QG_HFP
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 24
fig, ax1 = plt.subplots(figsize=(15, 10))
ax1.plot(completed_data.loc[(completed_data.groupby(completed_data["Datetime"].dt.date)["TA_1_1_2"].transform("mean") < 12)].groupby("Time")["GR_ST"].mean().index,
         completed_data.loc[(completed_data.groupby(completed_data["Datetime"].dt.date)["TA_1_1_2"].transform("mean") < 12)].groupby("Time")["GR_ST"].mean(), linestyle="-", label="Surface Temperature")
ax1.plot(completed_data.loc[completed_data.groupby(completed_data["Datetime"].dt.date)["TA_1_1_2"].transform("mean") < 12].groupby("Time")["SWC_1_1_1"].mean().index,
         completed_data.loc[completed_data.groupby(completed_data["Datetime"].dt.date)["TA_1_1_2"].transform("mean") < 12].groupby("Time")["SWC_1_1_1"].mean(), linestyle="--", label="S$_r$", color="red")
ax2 = ax1.twinx()
ax2.plot(completed_data.loc[completed_data.groupby(completed_data["Datetime"].dt.date)["TA_1_1_2"].transform("mean") < 12].groupby("Time")["G_plate_1_1_1"].mean().index,
         completed_data.loc[completed_data.groupby(completed_data["Datetime"].dt.date)["TA_1_1_2"].transform("mean") < 12].groupby("Time")["G_plate_1_1_1"].mean(), linestyle=":", label="Q$_G$ $_{HFP}$", color="forestgreen")
ax2.plot(completed_data.loc[(completed_data.groupby(completed_data["Datetime"].dt.date)["TA_1_1_2"].transform("mean") < 12)].groupby("Time")["TS_CS65X_1_1_1"].mean().index,
         completed_data.loc[(completed_data.groupby(completed_data["Datetime"].dt.date)["TA_1_1_2"].transform("mean") < 12)].groupby("Time")["TS_CS65X_1_1_1"].mean(), linestyle="-.", label="Soil Temperature", color="orange")
ax1.grid(False)
ax2.grid(False)
ax1.grid(axis="y")
ax1.set_xlabel("Hour of the Day")
ax1.set_ylabel("Surface Temperature (°C) / $S_r$ (%)")
ax2.set_ylabel("Soil Temperature (°C) / Q$_G$ $_{HFP}$ ($W/m^2$)")
ax1.set_ylim(0, 12)
ax2.set_ylim(-4, 8)
ax1.set_yticks([0, 4, 8, 12])
ax2.set_yticks([-4, 0, 4, 8])
plt.xlim(completed_data.groupby("Time")["SWC_1_1_1"].mean().index.min(), completed_data.groupby("Time")["SWC_1_1_1"].mean().index.max())
plt.xticks(completed_data.groupby("Time")["SWC_1_1_1"].mean().index[::6], labels=timelabels)
ax1.legend(loc="center", bbox_to_anchor=(0.2, -0.15), ncol = 4, frameon=False)
ax2.legend(loc="center", bbox_to_anchor=(0.8, -0.15), ncol = 4, frameon=False)
#plt.savefig("C:\\Users\\linus\\OneDrive\\Dokumente\\Masterarbeit\\Graphen\\MDC Heiztage Green Roof SWC ST TS QG HFP.pdf", format="pdf", bbox_inches="tight")
plt.show()


# Hybrid Roof SWC, Bodentemperatur, Oberflächentemperatur, QG_HFP
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 24
fig, ax1 = plt.subplots(figsize=(15, 10))
ax1.plot(completed_data.loc[(completed_data.groupby(completed_data["Datetime"].dt.date)["TA_1_1_2"].transform("mean") < 12)].groupby("Time")["HR_ST"].mean().index,
         completed_data.loc[(completed_data.groupby(completed_data["Datetime"].dt.date)["TA_1_1_2"].transform("mean") < 12)].groupby("Time")["HR_ST"].mean(), linestyle="-", label="Surface Temperature")
ax1.plot(completed_data.loc[(completed_data.groupby(completed_data["Datetime"].dt.date)["TA_1_1_2"].transform("mean") < 12)].groupby("Time")["TS_CS65X_1_1_2"].mean().index,
         completed_data.loc[(completed_data.groupby(completed_data["Datetime"].dt.date)["TA_1_1_2"].transform("mean") < 12)].groupby("Time")["TS_CS65X_1_1_2"].mean(), linestyle="-.", label="Soil Temperature")
ax1.plot(completed_data.loc[completed_data.groupby(completed_data["Datetime"].dt.date)["TA_1_1_2"].transform("mean") < 12].groupby("Time")["G_plate_1_1_2"].mean().index,
         completed_data.loc[completed_data.groupby(completed_data["Datetime"].dt.date)["TA_1_1_2"].transform("mean") < 12].groupby("Time")["G_plate_1_1_2"].mean(), linestyle=":", label="Q$_G$ $_{HFP}$", color="forestgreen")
ax2 = ax1.twinx()
ax2.plot(completed_data.loc[completed_data.groupby(completed_data["Datetime"].dt.date)["TA_1_1_2"].transform("mean") < 12].groupby("Time")["SWC_1_1_2"].mean().index,
         completed_data.loc[completed_data.groupby(completed_data["Datetime"].dt.date)["TA_1_1_2"].transform("mean") < 12].groupby("Time")["SWC_1_1_2"].mean(), linestyle="--", label="S$_r$", color="red")
ax1.grid(False)
ax2.grid(False)
ax1.grid(axis="y")
ax1.set_xlabel("Hour of the Day")
ax1.set_ylabel("Surface/Surface Temperature (°C) / Q$_G$ $_{HFP}$ ($W/m^2$)")
ax2.set_ylabel("$S_r$ (%)")
ax1.set_ylim(-6,6)
ax2.set_ylim(41.5, 42.5)
ax1.set_yticks([-6, -3, 0, 3, 6])
ax2.set_yticks([41.5, 42, 42.5])
plt.xlim(completed_data.groupby("Time")["SWC_1_1_1"].mean().index.min(), completed_data.groupby("Time")["SWC_1_1_1"].mean().index.max())
plt.xticks(completed_data.groupby("Time")["SWC_1_1_1"].mean().index[::6], labels=timelabels)
ax1.legend(loc="center", bbox_to_anchor=(0.3, -0.15), ncol = 4, frameon=False)
ax2.legend(loc="center", bbox_to_anchor=(0.9, -0.15), ncol = 4, frameon=False)
#plt.savefig("C:\\Users\\linus\\OneDrive\\Dokumente\\Masterarbeit\\Graphen\\MDC Heiztage Hybrid Roof SWC ST TS QG HFP.pdf", format="pdf", bbox_inches="tight")
plt.show()

# Strahlungen
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 24
plt.figure(figsize=(15, 10))
plt.plot(completed_data.loc[(completed_data.groupby(completed_data["Datetime"].dt.date)["TA_1_1_2"].transform("mean") < 12)].groupby("Time")["Q*_rad_gr"].mean().index,
         completed_data.loc[(completed_data.groupby(completed_data["Datetime"].dt.date)["TA_1_1_2"].transform("mean") < 12)].groupby("Time")["Q*_rad_gr"].mean(), linestyle="-", label="Q*$_{GR}$")
plt.plot(completed_data.loc[(completed_data.groupby(completed_data["Datetime"].dt.date)["TA_1_1_2"].transform("mean") < 12)].groupby("Time")["Q*_rad_hr"].mean().index,
         completed_data.loc[(completed_data.groupby(completed_data["Datetime"].dt.date)["TA_1_1_2"].transform("mean") < 12)].groupby("Time")["Q*_rad_hr"].mean(), linestyle="-", label="Q*$_{HR}$")
plt.plot(completed_data.loc[(completed_data.groupby(completed_data["Datetime"].dt.date)["TA_1_1_2"].transform("mean") < 12)].groupby("Time")["SW_IN"].mean().index,
         completed_data.loc[(completed_data.groupby(completed_data["Datetime"].dt.date)["TA_1_1_2"].transform("mean") < 12)].groupby("Time")["SW_IN"].mean(), linestyle="--", label="SW IN")
plt.plot(completed_data.loc[(completed_data.groupby(completed_data["Datetime"].dt.date)["TA_1_1_2"].transform("mean") < 12)].groupby("Time")["LW_IN"].mean().index,
         completed_data.loc[(completed_data.groupby(completed_data["Datetime"].dt.date)["TA_1_1_2"].transform("mean") < 12)].groupby("Time")["LW_IN"].mean(), linestyle="--", label="LW IN")
plt.plot(completed_data.loc[(completed_data.groupby(completed_data["Datetime"].dt.date)["TA_1_1_2"].transform("mean") < 12)].groupby("Time")["SW_OUT_GR_Calculated"].mean().index,
         completed_data.loc[(completed_data.groupby(completed_data["Datetime"].dt.date)["TA_1_1_2"].transform("mean") < 12)].groupby("Time")["SW_OUT_GR_Calculated"].mean(), linestyle="-.", label="SW OUT GR")
plt.plot(completed_data.loc[(completed_data.groupby(completed_data["Datetime"].dt.date)["TA_1_1_2"].transform("mean") < 12)].groupby("Time")["SW_OUT_HR_Calculated"].mean().index,
         completed_data.loc[(completed_data.groupby(completed_data["Datetime"].dt.date)["TA_1_1_2"].transform("mean") < 12)].groupby("Time")["SW_OUT_HR_Calculated"].mean(), linestyle="-.", label="SW OUT HR")
plt.plot(completed_data.loc[(completed_data.groupby(completed_data["Datetime"].dt.date)["TA_1_1_2"].transform("mean") < 12)].groupby("Time")["LW_OUT_GR_Calculated"].mean().index,
         completed_data.loc[(completed_data.groupby(completed_data["Datetime"].dt.date)["TA_1_1_2"].transform("mean") < 12)].groupby("Time")["LW_OUT_GR_Calculated"].mean(), linestyle=":", label="LW OUT GR")
plt.plot(completed_data.loc[(completed_data.groupby(completed_data["Datetime"].dt.date)["TA_1_1_2"].transform("mean") < 12)].groupby("Time")["LW_OUT_HR_Calculated"].mean().index,
         completed_data.loc[(completed_data.groupby(completed_data["Datetime"].dt.date)["TA_1_1_2"].transform("mean") < 12)].groupby("Time")["LW_OUT_HR_Calculated"].mean(), linestyle=":", label="LW OUT HR")
plt.grid(False)
plt.grid(False)
plt.grid(axis="y")
plt.xlabel("Hour of the Day")
plt.ylabel("$W/m^2$")
plt.xlim(completed_data.groupby("Time")["SW_IN"].mean().index.min(), completed_data.groupby("Time")["SW_IN"].mean().index.max())
plt.xticks(completed_data.groupby("Time")["SW_IN"].mean().index[::6], labels=timelabels)
plt.legend(loc="center", bbox_to_anchor=(0.5, -0.2), ncol = 4, frameon=False)
#plt.savefig("C:\\Users\\linus\\OneDrive\\Dokumente\\Masterarbeit\\Graphen\\MDC Heiztage Strahlung.pdf", format="pdf", bbox_inches="tight")
plt.show()

## Mittlere Tagesgänge Sommertage
# TA, rH, WS
timelabels = ["", "03:00", "06:00", "09:00", "12:00", "15:00", "18:00", "21:00"]
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 20
fig, ax1 = plt.subplots(figsize=(12, 6))
ax1.plot(completed_data.loc[completed_data.groupby(completed_data["Datetime"].dt.date)["TA_1_1_2"].transform("max") >= 25].groupby("Time")["RH_1_1_2"].mean().index, completed_data.loc[completed_data.groupby(completed_data["Datetime"].dt.date)["TA_1_1_2"].transform("max") >= 25].groupby("Time")["RH_1_1_2"].mean(), linestyle="--", label="relative Humidity")
ax2 = ax1.twinx()
ax2.plot(completed_data.loc[completed_data.groupby(completed_data["Datetime"].dt.date)["TA_1_1_2"].transform("max") >= 25].groupby("Time")["TA_1_1_2"].mean().index, completed_data.loc[completed_data.groupby(completed_data["Datetime"].dt.date)["TA_1_1_2"].transform("max") >= 25].groupby("Time")["TA_1_1_2"].mean(), linestyle="-", label="Air Temperature", color="orange")
ax2.plot(completed_data.loc[completed_data.groupby(completed_data["Datetime"].dt.date)["TA_1_1_2"].transform("max") >= 25].groupby("Time")["WS"].mean().index, completed_data.loc[completed_data.groupby(completed_data["Datetime"].dt.date)["TA_1_1_2"].transform("max") >= 25].groupby("Time")["WS"].mean(), linestyle="-.", label="Windspeed", color="forestgreen")
ax1.grid(False)
ax2.grid(False)
ax1.grid(axis="y")
ax1.set_xlabel("Hour of the Day")
ax1.set_ylabel("relative Humidity (%)")
ax2.set_ylabel("Air Temperature (°C) / Windspeed ($km/h$)")
ax1.set_xlim(completed_data.groupby("Time")["RH_1_1_2"].mean().index.min(), completed_data.groupby("Time")["RH_1_1_2"].mean().index.max())
ax1.set_ylim(40, 80)
ax2.set_ylim(0, 27)
ax1.set_yticks([40, 48, 56, 64, 72, 80])
ax2.set_yticks([0, 6, 12, 18, 24, 30])
plt.xticks(completed_data.groupby("Time")["RH_1_1_2"].mean().index[::6], labels=timelabels)
ax1.legend(loc="center", bbox_to_anchor=(0.2, -0.2), ncol = 1, frameon=False)
ax2.legend(loc="center", bbox_to_anchor=(0.7, -0.2), ncol = 2, frameon=False)
#plt.savefig("C:\\Users\\linus\\OneDrive\\Dokumente\\Masterarbeit\\Graphen\\MDC Sommertage TA rH WS.pdf", format="pdf", bbox_inches="tight")
plt.show()


# Green Roof QE, QH, QG, Q* 
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 20
plt.figure(figsize=(12, 6))
plt.plot(completed_data.loc[completed_data.groupby(completed_data["Datetime"].dt.date)["TA_1_1_2"].transform("max") >= 25].groupby("Time")["Q*_flux_gr"].mean().index, completed_data.loc[completed_data.groupby(completed_data["Datetime"].dt.date)["TA_1_1_2"].transform("max") >= 25].groupby("Time")["Q*_flux_gr"].mean(), label="Q*")
plt.plot(completed_data.loc[completed_data.groupby(completed_data["Datetime"].dt.date)["TA_1_1_2"].transform("max") >= 25].groupby("Time")["QE_GR"].mean().index, completed_data.loc[completed_data.groupby(completed_data["Datetime"].dt.date)["TA_1_1_2"].transform("max") >= 25].groupby("Time")["QE_GR"].mean(), label="Q$_E$", color="forestgreen")
plt.plot(completed_data.loc[completed_data.groupby(completed_data["Datetime"].dt.date)["TA_1_1_2"].transform("max") >= 25].groupby("Time")["QG_GR"].mean().index, completed_data.loc[completed_data.groupby(completed_data["Datetime"].dt.date)["TA_1_1_2"].transform("max") >= 25].groupby("Time")["QG_GR"].mean(), label="Q$_G$", color="orange")
plt.plot(completed_data.loc[completed_data.groupby(completed_data["Datetime"].dt.date)["TA_1_1_2"].transform("max") >= 25].groupby("Time")["QH_GR"].mean().index, completed_data.loc[completed_data.groupby(completed_data["Datetime"].dt.date)["TA_1_1_2"].transform("max") >= 25].groupby("Time")["QH_GR"].mean(), label="Q$_H$", color="red")
plt.grid(False)
plt.grid(axis="y")
plt.xlabel("Hour of the Day")
plt.ylabel("$W/m^2$")
plt.xlim(completed_data.groupby("Time")["RH_1_1_2"].mean().index.min(), completed_data.groupby("Time")["RH_1_1_2"].mean().index.max())
plt.xticks(completed_data.groupby("Time")["RH_1_1_2"].mean().index[::6], labels=timelabels)
plt.legend(loc="center", bbox_to_anchor=(0.5, -0.2), ncol = 4, frameon=False)
#plt.savefig("C:\\Users\\linus\\OneDrive\\Dokumente\\Masterarbeit\\Graphen\\MDC Sommertage Green Roof Heat Fluxes.pdf", format="pdf", bbox_inches="tight")
plt.show()


# Hybrid Roof QE, QH, QG, Q* 
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 20
plt.figure(figsize=(12, 6))
plt.plot(completed_data.loc[completed_data.groupby(completed_data["Datetime"].dt.date)["TA_1_1_2"].transform("max") >= 25].groupby("Time")["Q*_flux_hr"].mean().index, completed_data.loc[completed_data.groupby(completed_data["Datetime"].dt.date)["TA_1_1_2"].transform("max") >= 25].groupby("Time")["Q*_flux_hr"].mean(), label="Q*")
plt.plot(completed_data.loc[completed_data.groupby(completed_data["Datetime"].dt.date)["TA_1_1_2"].transform("max") >= 25].groupby("Time")["QE_HR"].mean().index, completed_data.loc[completed_data.groupby(completed_data["Datetime"].dt.date)["TA_1_1_2"].transform("max") >= 25].groupby("Time")["QE_HR"].mean(), label="Q$_E$", color="forestgreen")
plt.plot(completed_data.loc[completed_data.groupby(completed_data["Datetime"].dt.date)["TA_1_1_2"].transform("max") >= 25].groupby("Time")["QG_HR"].mean().index, completed_data.loc[completed_data.groupby(completed_data["Datetime"].dt.date)["TA_1_1_2"].transform("max") >= 25].groupby("Time")["QG_HR"].mean(), label="Q$_G$", color="orange")
plt.plot(completed_data.loc[completed_data.groupby(completed_data["Datetime"].dt.date)["TA_1_1_2"].transform("max") >= 25].groupby("Time")["QH_HR"].mean().index, completed_data.loc[completed_data.groupby(completed_data["Datetime"].dt.date)["TA_1_1_2"].transform("max") >= 25].groupby("Time")["QH_HR"].mean(), label="Q$_H$", color="red")
plt.plot(completed_data.loc[completed_data.groupby(completed_data["Datetime"].dt.date)["TA_1_1_2"].transform("max") >= 25].groupby("Time")["QH_NU"].mean().index, completed_data.loc[completed_data.groupby(completed_data["Datetime"].dt.date)["TA_1_1_2"].transform("max") >= 25].groupby("Time")["QH_NU"].mean(), color="lime", label="Q$_H$$_{Nu}$")
plt.grid(False)
plt.grid(axis="y")
plt.xlabel("Hour of the Day")
plt.ylabel("$W/m^2$")
plt.xlim(completed_data.groupby("Time")["RH_1_1_2"].mean().index.min(), completed_data.groupby("Time")["RH_1_1_2"].mean().index.max())
plt.xticks(completed_data.groupby("Time")["RH_1_1_2"].mean().index[::6], labels=timelabels)
plt.legend(loc="center", bbox_to_anchor=(0.5, -0.2), ncol = 4, frameon=False)
#plt.savefig("C:\\Users\\linus\\OneDrive\\Dokumente\\Masterarbeit\\Graphen\\MDC Sommertage Hybrid Roof Heat Fluxes.pdf", format="pdf", bbox_inches="tight")
plt.show()

# Green Roof SWC, Bodentemperatur, Oberflächentemperatur, QG_HFP
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 24
fig, ax1 = plt.subplots(figsize=(15, 10))
ax1.plot(completed_data.loc[completed_data.groupby(completed_data["Datetime"].dt.date)["TA_1_1_2"].transform("max") >= 25].groupby("Time")["GR_ST"].mean().index, completed_data.loc[completed_data.groupby(completed_data["Datetime"].dt.date)["TA_1_1_2"].transform("max") >= 25].groupby("Time")["GR_ST"].mean(), linestyle="-", label="Surface Temperature")
ax1.plot(completed_data.loc[completed_data.groupby(completed_data["Datetime"].dt.date)["TA_1_1_2"].transform("max") >= 25].groupby("Time")["TS_CS65X_1_1_1"].mean().index, completed_data.loc[completed_data.groupby(completed_data["Datetime"].dt.date)["TA_1_1_2"].transform("max") >= 25].groupby("Time")["TS_CS65X_1_1_1"].mean(), linestyle="-.", label="Soil Temperature")
ax2 = ax1.twinx()
ax2.plot(completed_data.loc[completed_data.groupby(completed_data["Datetime"].dt.date)["TA_1_1_2"].transform("max") >= 25].groupby("Time")["G_plate_1_1_1"].mean().index, completed_data.loc[completed_data.groupby(completed_data["Datetime"].dt.date)["TA_1_1_2"].transform("max") >= 25].groupby("Time")["G_plate_1_1_1"].mean(), linestyle=":", label="Q$_G$ $_{HFP}$", color="forestgreen")
ax2.plot(completed_data.loc[completed_data.groupby(completed_data["Datetime"].dt.date)["TA_1_1_2"].transform("max") >= 25].groupby("Time")["SWC_1_1_1"].mean().index, completed_data.loc[completed_data.groupby(completed_data["Datetime"].dt.date)["TA_1_1_2"].transform("max") >= 25].groupby("Time")["SWC_1_1_1"].mean(), linestyle="--", label="S$_r$", color="red")
ax1.grid(False)
ax2.grid(False)
ax1.grid(axis="y")
ax1.set_xlabel("Hour of the Day")
ax1.set_ylabel("Temperature (°C)")
ax2.set_ylabel("$S_r$ (%) / Q$_G$ $_{HFP}$ ($W/m^2$)")
ax1.set_ylim(15,45)
ax2.set_ylim(-15,15)
ax1.set_yticks([15, 21, 27, 33, 39, 45])
ax2.set_yticks([-15, -9, -3, 3, 9, 15])
plt.xlim(completed_data.groupby("Time")["SWC_1_1_1"].mean().index.min(), completed_data.groupby("Time")["SWC_1_1_1"].mean().index.max())
plt.xticks(completed_data.groupby("Time")["SWC_1_1_1"].mean().index[::6], labels=timelabels)
ax1.legend(loc="center", bbox_to_anchor=(0.3, -0.15), ncol = 4, frameon=False)
ax2.legend(loc="center", bbox_to_anchor=(0.9, -0.15), ncol = 4, frameon=False)
#plt.savefig("C:\\Users\\linus\\OneDrive\\Dokumente\\Masterarbeit\\Graphen\\MDC Sommertage Green Roof SWC ST TS QG HFP.pdf", format="pdf", bbox_inches="tight")
plt.show()


# Hybrid Roof SWC, Bodentemperatur, Oberflächentemperatur, QG_HFP
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 24
fig, ax1 = plt.subplots(figsize=(15, 10))
ax1.plot(completed_data.loc[completed_data.groupby(completed_data["Datetime"].dt.date)["TA_1_1_2"].transform("max") >= 25].groupby("Time")["HR_ST"].mean().index, completed_data.loc[completed_data.groupby(completed_data["Datetime"].dt.date)["TA_1_1_2"].transform("max") >= 25].groupby("Time")["HR_ST"].mean(), linestyle="-", label="Surface Temperature")
ax1.plot(completed_data.loc[completed_data.groupby(completed_data["Datetime"].dt.date)["TA_1_1_2"].transform("max") >= 25].groupby("Time")["TS_CS65X_1_1_2"].mean().index, completed_data.loc[completed_data.groupby(completed_data["Datetime"].dt.date)["TA_1_1_2"].transform("max") >= 25].groupby("Time")["TS_CS65X_1_1_2"].mean(), linestyle="-.", label="Soil Temperature")
ax2 = ax1.twinx()
ax2.plot(completed_data.loc[completed_data.groupby(completed_data["Datetime"].dt.date)["TA_1_1_2"].transform("max") >= 25].groupby("Time")["G_plate_1_1_2"].mean().index, completed_data.loc[completed_data.groupby(completed_data["Datetime"].dt.date)["TA_1_1_2"].transform("max") >= 25].groupby("Time")["G_plate_1_1_2"].mean(), linestyle=":", label="Q$_G$ $_{HFP}$", color="forestgreen")
ax2.plot(completed_data.loc[completed_data.groupby(completed_data["Datetime"].dt.date)["TA_1_1_2"].transform("max") >= 25].groupby("Time")["SWC_1_1_2"].mean().index, completed_data.loc[completed_data.groupby(completed_data["Datetime"].dt.date)["TA_1_1_2"].transform("max") >= 25].groupby("Time")["SWC_1_1_2"].mean(), linestyle="--", label="S$_r$", color="red")
ax1.grid(False)
ax2.grid(False)
ax1.grid(axis="y")
ax1.set_xlabel("Hour of the Day")
ax1.set_ylabel("Temperature (°C)")
ax2.set_ylabel("$S_r$ (%) / Q$_G$ $_{HFP}$ ($W/m^2$)")
ax1.set_ylim(15,35)
ax2.set_ylim(-15,35)
ax1.set_yticks([15, 19, 23, 27, 31, 35])
ax2.set_yticks([-15, -5, 5, 15, 25, 35])
plt.xlim(completed_data.groupby("Time")["SWC_1_1_1"].mean().index.min(), completed_data.groupby("Time")["SWC_1_1_1"].mean().index.max())
plt.xticks(completed_data.groupby("Time")["SWC_1_1_1"].mean().index[::6], labels=timelabels)
ax1.legend(loc="center", bbox_to_anchor=(0.3, -0.15), ncol = 4, frameon=False)
ax2.legend(loc="center", bbox_to_anchor=(0.9, -0.15), ncol = 4, frameon=False)
#plt.savefig("C:\\Users\\linus\\OneDrive\\Dokumente\\Masterarbeit\\Graphen\\MDC Sommertage Hybrid Roof SWC ST TS QG HFP.pdf", format="pdf", bbox_inches="tight")
plt.show()

# Strahlungen
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 24
plt.figure(figsize=(15, 10))
plt.plot(completed_data.loc[completed_data.groupby(completed_data["Datetime"].dt.date)["TA_1_1_2"].transform("max") >= 25].groupby("Time")["Q*_rad_gr"].mean().index, completed_data.loc[completed_data.groupby(completed_data["Datetime"].dt.date)["TA_1_1_2"].transform("max") >= 25].groupby("Time")["Q*_rad_gr"].mean(), linestyle="-", label="Q*$_{GR}$")
plt.plot(completed_data.loc[completed_data.groupby(completed_data["Datetime"].dt.date)["TA_1_1_2"].transform("max") >= 25].groupby("Time")["Q*_rad_hr"].mean().index, completed_data.loc[completed_data.groupby(completed_data["Datetime"].dt.date)["TA_1_1_2"].transform("max") >= 25].groupby("Time")["Q*_rad_hr"].mean(), linestyle="-", label="Q*$_{HR}$")
plt.plot(completed_data.loc[completed_data.groupby(completed_data["Datetime"].dt.date)["TA_1_1_2"].transform("max") >= 25].groupby("Time")["SW_IN"].mean().index, completed_data.loc[completed_data.groupby(completed_data["Datetime"].dt.date)["TA_1_1_2"].transform("max") >= 25].groupby("Time")["SW_IN"].mean(), linestyle="--", label="SW IN")
plt.plot(completed_data.loc[completed_data.groupby(completed_data["Datetime"].dt.date)["TA_1_1_2"].transform("max") >= 25].groupby("Time")["LW_IN"].mean().index, completed_data.loc[completed_data.groupby(completed_data["Datetime"].dt.date)["TA_1_1_2"].transform("max") >= 25].groupby("Time")["LW_IN"].mean(), linestyle="--", label="LW IN")
plt.plot(completed_data.loc[completed_data.groupby(completed_data["Datetime"].dt.date)["TA_1_1_2"].transform("max") >= 25].groupby("Time")["SW_OUT_GR_Calculated"].mean().index, completed_data.loc[completed_data.groupby(completed_data["Datetime"].dt.date)["TA_1_1_2"].transform("max") >= 25].groupby("Time")["SW_OUT_GR_Calculated"].mean(), linestyle="-.", label="SW OUT GR")
plt.plot(completed_data.loc[completed_data.groupby(completed_data["Datetime"].dt.date)["TA_1_1_2"].transform("max") >= 25].groupby("Time")["SW_OUT_HR_Calculated"].mean().index, completed_data.loc[completed_data.groupby(completed_data["Datetime"].dt.date)["TA_1_1_2"].transform("max") >= 25].groupby("Time")["SW_OUT_HR_Calculated"].mean(), linestyle="-.", label="SW OUT HR")
plt.plot(completed_data.loc[completed_data.groupby(completed_data["Datetime"].dt.date)["TA_1_1_2"].transform("max") >= 25].groupby("Time")["LW_OUT_GR_Calculated"].mean().index, completed_data.loc[completed_data.groupby(completed_data["Datetime"].dt.date)["TA_1_1_2"].transform("max") >= 25].groupby("Time")["LW_OUT_GR_Calculated"].mean(), linestyle=":", label="LW OUT GR")
plt.plot(completed_data.loc[completed_data.groupby(completed_data["Datetime"].dt.date)["TA_1_1_2"].transform("max") >= 25].groupby("Time")["LW_OUT_HR_Calculated"].mean().index, completed_data.loc[completed_data.groupby(completed_data["Datetime"].dt.date)["TA_1_1_2"].transform("max") >= 25].groupby("Time")["LW_OUT_HR_Calculated"].mean(), linestyle=":", label="LW OUT HR")
plt.grid(False)
plt.grid(False)
plt.grid(axis="y")
plt.xlabel("Hour of the Day")
plt.ylabel("$W/m^2$")
plt.xlim(completed_data.groupby("Time")["SW_IN"].mean().index.min(), completed_data.groupby("Time")["SW_IN"].mean().index.max())
plt.xticks(completed_data.groupby("Time")["SW_IN"].mean().index[::6], labels=timelabels)
plt.legend(loc="center", bbox_to_anchor=(0.5, -0.2), ncol = 4, frameon=False)
#plt.savefig("C:\\Users\\linus\\OneDrive\\Dokumente\\Masterarbeit\\Graphen\\MDC Sommertage Strahlung.pdf", format="pdf", bbox_inches="tight")
plt.show()


# plt.rcParams["font.family"] = "Arial"
# plt.rcParams["font.size"] = 24
# ax = WindroseAxes.from_ax()
# ax.bar(completed_data["WD"], completed_data["WS"], normed=True, opening=1,
#         edgecolor="black", bins=9, cmap=plt.get_cmap("hot"))
# ax.tick_params(axis="both")
# ax.legend(loc="center", labels=["< 1.5 $m/s$", "< 3 $m/s$", "< 4.5 $m/s$", "< 6 $m/s$", "< 7.5 $m/s$", ">= 7.5 $m/s$"], ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.2))
# plt.show()


sommertage = completed_data.loc[completed_data.groupby(completed_data["Datetime"].dt.date)["TA_1_1_2"].transform("max") >= 25][["TA_1_1_2", "Datetime", "Precipitation"]]
sommertage = sommertage.loc[completed_data.groupby(completed_data["Datetime"].dt.date)["Precipitation"].transform("max")==0]
print(sommertage["Datetime"].dt.date.unique())
plt.figure(figsize=(30,5))
plt.plot(sommertage["Datetime"], sommertage["Precipitation"])
plt.grid()
plt.show()
# Sommertage in Reihe (mindestens 3 in Folge) (auch ohne Regen damit QE vollständig):
# 08.07 bis zum 12.07 
# 20.08 bis zum 24.08
# 04.09 bis zum 11.09


heißtage = completed_data.loc[completed_data.groupby(completed_data["Datetime"].dt.date)["TA_1_1_2"].transform("max") >= 30][["TA_1_1_2", "Datetime"]]
print(heißtage["Datetime"].dt.date.unique())
# nie mehr als 2 Tage in Folge

tropennacht = completed_data.loc[completed_data.groupby(completed_data["Datetime"].dt.date)["TA_1_1_2"].transform("min") >= 20][["TA_1_1_2", "Datetime"]]
print(tropennacht["Datetime"].dt.date.unique() )
# nie mehr als 2 Tage in Folge

Heiztage = completed_data.loc[(completed_data.groupby(completed_data["Datetime"].dt.date)["TA_1_1_2"].transform("mean") < 12) & (completed_data.groupby(completed_data["Datetime"].dt.date)["Precipitation"].transform("mean") == 0)][["TA_1_1_2", "Datetime"]]
print(Heiztage["Datetime"].dt.date.unique())
# Heiztage (mindestens 3 in Folge):
# 2024-01-06 bis 2024-01-11
# 2024-01-16 bis 2024-01-21
# 2024-01-27 bis 2024-02-06
# 2024-02-24 bis 2024-02-29
# 2024-03-02 bis 2024-03-04



# Nach initialen Testplots wurden folgende Zeiträume für die Hitzeperiode ausgesucht:
# 18. August bis zum 25. August
HP_1 = completed_data.loc[(completed_data["Datetime"].dt.date >= pd.to_datetime("2023-08-20").date()) & (completed_data["Datetime"].dt.date < pd.to_datetime("2023-08-25").date())]

# 04. September bis zum 11. September
HP_2 = completed_data.loc[(completed_data["Datetime"].dt.date >= pd.to_datetime("2023-09-04").date()) & (completed_data["Datetime"].dt.date < pd.to_datetime("2023-09-12").date())]

# Plots zur HP_1 20. August bis zum 24. August
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 36
fig, ax1 = plt.subplots(figsize=(27,10))
ax1.plot(HP_1["Datetime"],
         HP_1["TA_1_1_2"], label="Air Temperature", color="dodgerblue", linewidth=3.5, linestyle="-")
ax1.plot(HP_1["Datetime"],
         HP_1["GR_ST"], label="T$_{Surface}$ $_{GR}$", color="crimson", linewidth=3.5, linestyle="--")
ax1.plot(HP_1["Datetime"],
         HP_1["HR_ST"], label="T$_{Surface}$ $_{HR}$", color="goldenrod", linewidth=3.5, linestyle="-.")
ax1.plot(HP_1["Datetime"], HP_1["TS_CS65X_1_1_1"], label="T$_{Soil}$ $_{GR}$", color="dimgray", linewidth=3, linestyle="-")
ax1.plot(HP_1["Datetime"], HP_1["TS_CS65X_1_1_2"], label="T$_{Soil}$ $_{HR}$", color="black", linewidth=3, linestyle="--")
ax2 = ax1.twinx()
ax2.plot(HP_1["Datetime"],
         HP_1["SWC_1_1_1"], color="mediumseagreen", label="$S_r$ $_{GR}$", linewidth=3.5, linestyle="-.")
ax2.plot(HP_1["Datetime"],
         HP_1["SWC_1_1_2"], color="midnightblue", label="$S_r$ $_{HR}$", linewidth=3.5, linestyle="-.")
ax1.set_ylabel("Temperature (°C)")
ax2.set_ylabel("Saturation Coefficient (%)")
ax1.set_ylim(10,45)
ax2.set_ylim(0,75)
ax1.set_yticks([10,17,24,31,38,45])
ax2.set_yticks([0,15,30,45,60,75])
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%d. %b"))
ax1.set_xlim(HP_1["Datetime"].min(),
             HP_1["Datetime"].max())
ax1.set_xticks(HP_1["Datetime"].dt.date[::48])
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
ax1.legend(loc="center", bbox_to_anchor=(0.5, -0.25), ncol = 5, frameon=False)
ax2.legend(loc="center", bbox_to_anchor=(0.5, -0.35), ncol = 2, frameon=False)
plt.grid()
#plt.savefig("C:\\Users\\linus\\OneDrive\\Dokumente\\Masterarbeit\\Graphen\\Zeitreihe Hitzeperiode 1 TA SWCs .pdf", format="pdf", bbox_inches="tight")
plt.show()



# HP 1 MDC Flux GR
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 20
plt.figure(figsize=(10,6))
plt.plot(HP_1["Datetime"],HP_1["Q*_flux_gr"], label="Q*", color="dodgerblue", linewidth=2, linestyle="-")
plt.plot(HP_1["Datetime"],HP_1["QH_GR"], label="Q$_H$", color="crimson", linewidth=2, linestyle="--")
plt.plot(HP_1["Datetime"],HP_1["QE_GR"], label="Q$_E$", color="goldenrod", linewidth=2, linestyle="-.")
plt.plot(HP_1["Datetime"],HP_1["QG_GR"], label="Q$_G$", color="dimgray", linewidth=2, linestyle="-")
plt.xlim(HP_1["Datetime"].min(), HP_1["Datetime"].max())
plt.ylabel("$W/m²$")
plt.legend(loc="center", bbox_to_anchor=(0.5, -0.25), ncol = 4, frameon=False)
plt.xticks(HP_1["Datetime"][::48], HP_1["Datetime"].dt.strftime("%d. %b")[::48], rotation=45, ha="right")
plt.grid(False)
plt.grid(axis="y")
#plt.savefig("C:\\Users\\linus\\OneDrive\\Dokumente\\Masterarbeit\\Graphen\\Zeitreihe Hitzeperiode 1 Green Roof Heat Fluxes.pdf", format="pdf", bbox_inches="tight")
plt.show()


# HP 1 MDC Flux HR
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 20
plt.figure(figsize=(10,6))
plt.plot(HP_1["Datetime"],HP_1["Q*_flux_hr"], label="Q*", color="dodgerblue", linewidth=2, linestyle="-")
plt.plot(HP_1["Datetime"],HP_1["QH_HR"], label="Q$_H$", color="crimson", linewidth=2, linestyle="--")
plt.plot(HP_1["Datetime"],HP_1["QE_HR"], label="Q$_E$", color="goldenrod", linewidth=2, linestyle="-.")
plt.plot(HP_1["Datetime"],HP_1["QG_HR"], label="Q$_G$", color="dimgray", linewidth=2, linestyle="-")
plt.xlim(HP_1["Datetime"].min(), HP_1["Datetime"].max())
plt.ylabel("$W/m²$")
plt.legend(loc="center", bbox_to_anchor=(0.5, -0.25), ncol = 4, frameon=False)
plt.xticks(HP_1["Datetime"][::48], HP_1["Datetime"].dt.strftime("%d. %b")[::48], rotation=45, ha="right")
plt.grid(False)
plt.grid(axis="y")
#plt.savefig("C:\\Users\\linus\\OneDrive\\Dokumente\\Masterarbeit\\Graphen\\Zeitreihe Hitzeperiode 1 Hybrid Roof Heat Fluxes.pdf", format="pdf", bbox_inches="tight")
plt.show()

print(HP_1["QH_GR"].mean()/HP_1["QE_GR"].mean())
print(HP_1["QH_HR"].mean()/HP_1["QE_HR"].mean())

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 32
plt.figure(figsize=(17,7))
plt.plot(HP_1["Datetime"], HP_1["G_plate_1_1_1"], label="Q$_G$ $_{HFP}$ $_{GR}$", color="dodgerblue", linewidth=2.5, linestyle="-")
plt.plot(HP_1["Datetime"], HP_1["G_plate_1_1_2"], label="Q$_G$ $_{HFP}$ $_{HR}$", color="crimson", linewidth=2.5, linestyle="--")
plt.xlim(HP_1["Datetime"].min(), HP_1["Datetime"].max())
plt.legend(loc="center", bbox_to_anchor=(0.5, -0.15), ncol = 2, frameon=False)
plt.xticks(HP_1["Datetime"][::48], HP_1["Datetime"].dt.strftime("%d. %b")[::48])
plt.grid(False)
plt.grid(axis="y")
#plt.savefig("C:\\Users\\linus\\OneDrive\\Dokumente\\Masterarbeit\\Graphen\\Zeitreihe Hitzeperiode 1 QG HFP.pdf", format="pdf", bbox_inches="tight")
plt.show()

# Plots zur HP_2 04. September bis zum 11. September
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 36
fig, ax1 = plt.subplots(figsize=(27,10))
ax1.plot(HP_2["Datetime"],
         HP_2["TA_1_1_2"], label="Air Temperature", color="dodgerblue", linewidth=3.5, linestyle="-")
ax1.plot(HP_2["Datetime"],
         HP_2["GR_ST"], label="T$_{Surface}$ $_{GR}$", color="crimson", linewidth=3.5, linestyle="--")
ax1.plot(HP_2["Datetime"],
         HP_2["HR_ST"], label="T$_{Surface}$ $_{HR}$", color="goldenrod", linewidth=3.5, linestyle="-.")
ax1.plot(HP_2["Datetime"], HP_2["TS_CS65X_1_1_1"], label="T$_{Soil}$ $_{GR}$", color="dimgray", linewidth=3, linestyle="-")
ax1.plot(HP_2["Datetime"], HP_2["TS_CS65X_1_1_2"], label="T$_{Soil}$ $_{HR}$", color="black", linewidth=3, linestyle="--")
ax2 = ax1.twinx()
ax2.plot(HP_2["Datetime"],
         HP_2["SWC_1_1_1"], color="mediumseagreen", label="$S_r$ $_{GR}$", linewidth=3.5, linestyle="-.")
ax2.plot(HP_2["Datetime"],
         HP_2["SWC_1_1_2"], color="midnightblue", label="$S_r$ $_{HR}$", linewidth=3.5, linestyle="-.")
ax1.set_ylabel("Temperature (°C)")
ax2.set_ylabel("Saturation Coefficient (%)")
ax1.set_ylim(10,45)
ax2.set_ylim(0,60)
ax1.set_yticks([10,17,24,31,38,45])
ax2.set_yticks([0,12,24,36,48,60])
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%d. %b"))
ax1.set_xlim(HP_2["Datetime"].min(),
             HP_2["Datetime"].max())
ax1.set_xticks(HP_2["Datetime"].dt.date[::48])
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
ax1.legend(loc="center", bbox_to_anchor=(0.5, -0.25), ncol = 5, frameon=False)
ax2.legend(loc="center", bbox_to_anchor=(0.5, -0.35), ncol = 2, frameon=False)
plt.grid()
#plt.savefig("C:\\Users\\linus\\OneDrive\\Dokumente\\Masterarbeit\\Graphen\\Zeitreihe Hitzeperiode 2 TA SWCs .pdf", format="pdf", bbox_inches="tight")
plt.show()



# HP 2 MDC Flux GR
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 20
plt.figure(figsize=(10,6))
plt.plot(HP_2["Datetime"],HP_2["Q*_flux_gr"], label="Q*", color="dodgerblue", linewidth=2, linestyle="-")
plt.plot(HP_2["Datetime"],HP_2["QH_GR"], label="Q$_H$", color="crimson", linewidth=2, linestyle="--")
plt.plot(HP_2["Datetime"],HP_2["QE_GR"], label="Q$_E$", color="goldenrod", linewidth=2, linestyle="-.")
plt.plot(HP_2["Datetime"],HP_2["QG_GR"], label="Q$_G$", color="#676767", linewidth=2, linestyle="-")
plt.xlim(HP_2["Datetime"].min(), HP_2["Datetime"].max())
plt.ylabel("$W/m²$")
plt.legend(loc="center", bbox_to_anchor=(0.5, -0.25), ncol = 4, frameon=False)
plt.xticks(HP_2["Datetime"][::48], HP_2["Datetime"].dt.strftime("%d. %b")[::48], rotation=45, ha="right")
plt.grid(False)
plt.grid(axis="y")
#plt.savefig("C:\\Users\\linus\\OneDrive\\Dokumente\\Masterarbeit\\Graphen\\Zeitreihe Hitzeperiode 2 Green Roof Heat Fluxes.pdf", format="pdf", bbox_inches="tight")
plt.show()


# HP 2 MDC Flux HR
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 20
plt.figure(figsize=(10,6))
plt.plot(HP_2["Datetime"],HP_2["Q*_flux_hr"], label="Q*", color="dodgerblue", linewidth=2, linestyle="-")
plt.plot(HP_2["Datetime"],HP_2["QH_HR"], label="Q$_H$", color="crimson", linewidth=2, linestyle="--")
plt.plot(HP_2["Datetime"],HP_2["QE_HR"], label="Q$_E$", color="goldenrod", linewidth=2, linestyle="-.")
plt.plot(HP_2["Datetime"],HP_2["QG_HR"], label="Q$_G$", color="#676767", linewidth=2, linestyle="-")
plt.xlim(HP_2["Datetime"].min(), HP_2["Datetime"].max())
plt.ylabel("$W/m²$")
plt.legend(loc="center", bbox_to_anchor=(0.5, -0.25), ncol = 4, frameon=False)
plt.xticks(HP_2["Datetime"][::48], HP_2["Datetime"].dt.strftime("%d. %b")[::48], rotation=45, ha="right")
plt.grid(False)
plt.grid(axis="y")
#plt.savefig("C:\\Users\\linus\\OneDrive\\Dokumente\\Masterarbeit\\Graphen\\Zeitreihe Hitzeperiode 2 Hybrid Roof Heat Fluxes.pdf", format="pdf", bbox_inches="tight")
plt.show()

print(HP_2["QH_GR"].mean()/HP_2["QE_GR"].mean())
print(HP_2["QH_HR"].mean()/HP_2["QE_HR"].mean())

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 32
plt.figure(figsize=(17,7))
plt.plot(HP_2["Datetime"], HP_2["G_plate_1_1_1"], label="Q$_G$ $_{HFP}$ $_{GR}$", color="dodgerblue", linewidth=2.5, linestyle="-")
plt.plot(HP_2["Datetime"], HP_2["G_plate_1_1_2"], label="Q$_G$ $_{HFP}$ $_{HR}$", color="crimson", linewidth=2.5, linestyle="--")
plt.xlim(HP_2["Datetime"].min(), HP_2["Datetime"].max())
plt.legend(loc="center", bbox_to_anchor=(0.5, -0.15), ncol = 2, frameon=False)
plt.xticks(HP_2["Datetime"][::48], HP_2["Datetime"].dt.strftime("%d. %b")[::48])
plt.grid(False)
plt.grid(axis="y")
#plt.savefig("C:\\Users\\linus\\OneDrive\\Dokumente\\Masterarbeit\\Graphen\\Zeitreihe Hitzeperiode 2 QG HFP.pdf", format="pdf", bbox_inches="tight")
plt.show()

print(HP_2["QE_GR"].mean()/(HP_2["QE_GR"].mean()+HP_2["QG_GR"].mean()+HP_2["QH_GR"].mean()))
print(HP_2["QE_HR"].mean()/(HP_2["QE_HR"].mean()+HP_2["QG_HR"].mean()+HP_2["QH_HR"].mean()))

# Nach initialen Testplots wurden folgende Zeiträume für die Kälteperiode ausgesucht:
# 2024-01-06 bis 2024-01-11
# 2024-01-16 bis 2024-01-21
# 2024-01-27 bis 2024-02-06
# 2024-02-24 bis 2024-02-29
# 2024-03-02 bis 2024-03-04
# 07.01. bis zum 11.01.
KP_1 = completed_data.loc[(completed_data["Datetime"].dt.date >= pd.to_datetime("2024-01-06").date()) & (completed_data["Datetime"].dt.date < pd.to_datetime("2024-01-12").date())]

# 27.01. bis zum 06.02.
KP_2 = completed_data.loc[(completed_data["Datetime"].dt.date >= pd.to_datetime("2024-02-24").date()) & (completed_data["Datetime"].dt.date < pd.to_datetime("2024-03-01").date())]

# Plots zur KP_1 06.01. bis zum 11.01.
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 36
fig, ax1 = plt.subplots(figsize=(27,10))
ax1.plot(KP_1["Datetime"],
         KP_1["TA_1_1_2"], label="Air Temperature", color="dodgerblue", linewidth=3.5, linestyle="-")
ax1.plot(KP_1["Datetime"],
         KP_1["GR_ST"], label="T$_{Surface}$ $_{GR}$", color="crimson", linewidth=3.5, linestyle="--")
ax1.plot(KP_1["Datetime"],
         KP_1["HR_ST"], label="T$_{Surface}$ $_{HR}$", color="goldenrod", linewidth=3.5, linestyle="-.")
ax1.plot(KP_1["Datetime"], KP_1["TS_CS65X_1_1_1"], label="T$_{Soil}$ $_{GR}$", color="dimgray", linewidth=3, linestyle="-")
ax1.plot(KP_1["Datetime"], KP_1["TS_CS65X_1_1_2"], label="T$_{Soil}$ $_{HR}$", color="black", linewidth=3, linestyle="--")
ax2 = ax1.twinx()
ax2.plot(KP_1["Datetime"],
         KP_1["SWC_1_1_1"], color="mediumseagreen", label="$S_r$ $_{GR}$", linewidth=3.5, linestyle="-.")
ax2.plot(KP_1["Datetime"],
         KP_1["SWC_1_1_2"], color="midnightblue", label="$S_r$ $_{HR}$", linewidth=3.5, linestyle="-.")
ax1.set_ylabel("Temperature (°C)")
ax2.set_ylabel("Saturation Coefficient (%)")
ax1.set_ylim(-27,18)
ax2.set_ylim(0,100)
ax1.set_yticks([-27,-18,-9,0,9,18])
ax2.set_yticks([0,20,40,60,80,100])
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%d. %b"))
ax1.set_xlim(KP_1["Datetime"].min(),
             KP_1["Datetime"].max())
ax1.set_xticks(KP_1["Datetime"].dt.date[::48])
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
ax1.legend(loc="center", bbox_to_anchor=(0.5, -0.25), ncol = 5, frameon=False)
ax2.legend(loc="center", bbox_to_anchor=(0.5, -0.35), ncol = 2, frameon=False)
plt.grid()
#plt.savefig("C:\\Users\\linus\\OneDrive\\Dokumente\\Masterarbeit\\Graphen\\Zeitreihe Kälteperiode 1 TA SWCs .pdf", format="pdf", bbox_inches="tight")
plt.show()



# KP 1 MDC Flux GR
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 20
plt.figure(figsize=(10,6))
plt.plot(KP_1["Datetime"],KP_1["Q*_flux_gr"], label="Q*", color="dodgerblue", linewidth=2, linestyle="-")
plt.plot(KP_1["Datetime"],KP_1["QH_GR"], label="Q$_H$", color="crimson", linewidth=2, linestyle="--")
plt.plot(KP_1["Datetime"],KP_1["QE_GR"], label="Q$_E$", color="goldenrod", linewidth=2, linestyle="-.")
plt.plot(KP_1["Datetime"],KP_1["QG_GR"], label="Q$_G$", color="dimgray", linewidth=2.5, linestyle="-")
plt.xlim(KP_1["Datetime"].min(), KP_1["Datetime"].max())
plt.ylabel("$W/m²$")
plt.legend(loc="center", bbox_to_anchor=(0.5, -0.25), ncol = 4, frameon=False)
plt.xticks(KP_1["Datetime"][::48], KP_1["Datetime"].dt.strftime("%d. %b")[::48], rotation=45, ha="right")
plt.grid(False)
plt.grid(axis="y")
#plt.savefig("C:\\Users\\linus\\OneDrive\\Dokumente\\Masterarbeit\\Graphen\\Zeitreihe Kälteperiode 1 Green Roof Heat Fluxes.pdf", format="pdf", bbox_inches="tight")
plt.show()


# KP 1 MDC Flux HR
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 20
plt.figure(figsize=(10,6))
plt.plot(KP_1["Datetime"],KP_1["Q*_flux_hr"], label="Q*", color="dodgerblue", linewidth=2, linestyle="-")
plt.plot(KP_1["Datetime"],KP_1["QH_HR"], label="Q$_H$", color="crimson", linewidth=2, linestyle="--")
plt.plot(KP_1["Datetime"],KP_1["QE_HR"], label="Q$_E$", color="goldenrod", linewidth=2, linestyle="-.")
plt.plot(KP_1["Datetime"],KP_1["QG_HR"], label="Q$_G$", color="dimgray", linewidth=2.5, linestyle="-")
plt.xlim(KP_1["Datetime"].min(), KP_1["Datetime"].max())
plt.ylabel("$W/m²$")
plt.legend(loc="center", bbox_to_anchor=(0.5, -0.25), ncol = 4, frameon=False)
plt.xticks(KP_1["Datetime"][::48], KP_1["Datetime"].dt.strftime("%d. %b")[::48], rotation=45, ha="right")
plt.grid(False)
plt.grid(axis="y")
#plt.savefig("C:\\Users\\linus\\OneDrive\\Dokumente\\Masterarbeit\\Graphen\\Zeitreihe Kälteperiode 1 Hybrid Roof Heat Fluxes.pdf", format="pdf", bbox_inches="tight")
plt.show()

print(KP_1["QH_GR"].mean()/KP_1["QE_GR"].mean())
print(KP_1["QH_HR"].mean()/KP_1["QE_HR"].mean())

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 32
plt.figure(figsize=(17,7))
plt.plot(KP_1["Datetime"], KP_1["G_plate_1_1_1"], label="Q$_G$ $_{HFP}$ $_{GR}$", color="dodgerblue", linewidth=2.5, linestyle="-")
plt.plot(KP_1["Datetime"], KP_1["G_plate_1_1_2"], label="Q$_G$ $_{HFP}$ $_{HR}$", color="crimson", linewidth=2.5, linestyle="--")
plt.xlim(KP_1["Datetime"].min(), KP_1["Datetime"].max())
plt.legend(loc="center", bbox_to_anchor=(0.5, -0.15), ncol = 2, frameon=False)
plt.xticks(KP_1["Datetime"][::48], KP_1["Datetime"].dt.strftime("%d. %b")[::48])
plt.grid(False)
plt.grid(axis="y")
#plt.savefig("C:\\Users\\linus\\OneDrive\\Dokumente\\Masterarbeit\\Graphen\\Zeitreihe Kälteperiode 1 QG HFP.pdf", format="pdf", bbox_inches="tight")
plt.show()

# Plots zur KP_2
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 36
fig, ax1 = plt.subplots(figsize=(27,10))
ax1.plot(KP_2["Datetime"],
         KP_2["TA_1_1_2"], label="Air Temperature", color="dodgerblue", linewidth=3.5, linestyle="-")
ax1.plot(KP_2["Datetime"],
         KP_2["GR_ST"], label="T$_{Surface}$ $_{GR}$", color="crimson", linewidth=3.5, linestyle="--")
ax1.plot(KP_2["Datetime"],
         KP_2["HR_ST"], label="T$_{Surface}$ $_{HR}$", color="goldenrod", linewidth=3.5, linestyle="-.")
ax1.plot(KP_2["Datetime"], KP_2["TS_CS65X_1_1_1"], label="T$_{Soil}$ $_{GR}$", color="dimgray", linewidth=3, linestyle="-")
ax1.plot(KP_2["Datetime"], KP_2["TS_CS65X_1_1_2"], label="T$_{Soil}$ $_{HR}$", color="black", linewidth=3, linestyle="--")
ax2 = ax1.twinx()
ax2.plot(KP_2["Datetime"],
         KP_2["SWC_1_1_1"], color="mediumseagreen", label="$S_r$ $_{GR}$", linewidth=3.5, linestyle="-.")
ax2.plot(KP_2["Datetime"],
         KP_2["SWC_1_1_2"], color="midnightblue", label="$S_r$ $_{HR}$", linewidth=3.5, linestyle="-.")
ax1.set_ylabel("Temperature (°C)")
ax2.set_ylabel("Saturation Coefficient (%)")
ax1.set_ylim(-6,24)
ax2.set_ylim(0,100)
ax1.set_yticks([-6,0,6,12,18,24])
ax2.set_yticks([0,20,40,60,80,100])
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%d. %b"))
ax1.set_xlim(KP_2["Datetime"].min(),
             KP_2["Datetime"].max())
ax1.set_xticks(KP_2["Datetime"].dt.date[::48])
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
ax1.legend(loc="center", bbox_to_anchor=(0.5, -0.25), ncol = 5, frameon=False)
ax2.legend(loc="center", bbox_to_anchor=(0.5, -0.35), ncol = 2, frameon=False)
plt.grid()
#plt.savefig("C:\\Users\\linus\\OneDrive\\Dokumente\\Masterarbeit\\Graphen\\Zeitreihe Kälteperiode 2 TA SWCs .pdf", format="pdf", bbox_inches="tight")
plt.show()



# KP 2 MDC Flux GR
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 20
plt.figure(figsize=(10,6))
plt.plot(KP_2["Datetime"],KP_2["Q*_flux_gr"], label="Q*", color="dodgerblue", linewidth=2, linestyle="-")
plt.plot(KP_2["Datetime"],KP_2["QH_GR"], label="Q$_H$", color="crimson", linewidth=2, linestyle="--")
plt.plot(KP_2["Datetime"],KP_2["QE_GR"], label="Q$_E$", color="goldenrod", linewidth=2, linestyle="-.")
plt.plot(KP_2["Datetime"],KP_2["QG_GR"], label="Q$_G$", color="dimgray", linewidth=2, linestyle="-")
plt.xlim(KP_2["Datetime"].min(), KP_2["Datetime"].max())
plt.ylabel("$W/m²$")
plt.legend(loc="center", bbox_to_anchor=(0.5, -0.25), ncol = 4, frameon=False)
plt.xticks(KP_2["Datetime"][::48], KP_2["Datetime"].dt.strftime("%d. %b")[::48], rotation=45, ha="right")
plt.grid(False)
plt.grid(axis="y")
#plt.savefig("C:\\Users\\linus\\OneDrive\\Dokumente\\Masterarbeit\\Graphen\\Zeitreihe Kälteperiode 2 Green Roof Heat Fluxes.pdf", format="pdf", bbox_inches="tight")
plt.show()


# KP 2 MDC Flux HR
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 20
plt.figure(figsize=(10,6))
plt.plot(KP_2["Datetime"],KP_2["Q*_flux_hr"], label="Q*", color="dodgerblue", linewidth=2, linestyle="-")
plt.plot(KP_2["Datetime"],KP_2["QH_HR"], label="Q$_H$", color="crimson", linewidth=2, linestyle="--")
plt.plot(KP_2["Datetime"],KP_2["QE_HR"], label="Q$_E$", color="goldenrod", linewidth=2, linestyle="-.")
plt.plot(KP_2["Datetime"],KP_2["QG_HR"], label="Q$_G$", color="dimgray", linewidth=2, linestyle="-")
plt.xlim(KP_2["Datetime"].min(), KP_2["Datetime"].max())
plt.ylabel("$W/m²$")
plt.legend(loc="center", bbox_to_anchor=(0.5, -0.25), ncol = 4, frameon=False)
plt.xticks(KP_2["Datetime"][::48], KP_2["Datetime"].dt.strftime("%d. %b")[::48], rotation=45, ha="right")
plt.grid(False)
plt.grid(axis="y")
#plt.savefig("C:\\Users\\linus\\OneDrive\\Dokumente\\Masterarbeit\\Graphen\\Zeitreihe Kälteperiode 2 Hybrid Roof Heat Fluxes.pdf", format="pdf", bbox_inches="tight")
plt.show()

print(KP_2["QH_GR"].mean()/KP_2["QE_GR"].mean())
print(KP_2["QH_HR"].mean()/KP_2["QE_HR"].mean())

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 32
plt.figure(figsize=(17,7))
plt.plot(KP_2["Datetime"], KP_2["G_plate_1_1_1"], label="Q$_G$ $_{HFP}$ $_{GR}$", color="dodgerblue", linewidth=2.5, linestyle="-")
plt.plot(KP_2["Datetime"], KP_2["G_plate_1_1_2"], label="Q$_G$ $_{HFP}$ $_{HR}$", color="crimson", linewidth=2.5, linestyle="--")
plt.xlim(KP_2["Datetime"].min(), KP_2["Datetime"].max())
plt.legend(loc="center", bbox_to_anchor=(0.5, -0.15), ncol = 2, frameon=False)
plt.xticks(KP_2["Datetime"][::48], KP_2["Datetime"].dt.strftime("%d. %b")[::48])
plt.grid(False)
plt.grid(axis="y")
#plt.savefig("C:\\Users\\linus\\OneDrive\\Dokumente\\Masterarbeit\\Graphen\\Zeitreihe Kälteperiode 2 QG HFP.pdf", format="pdf", bbox_inches="tight")
plt.show()


# Barplots/Histogram mit Linienplot
# Niederschlag vs TA für Messzeitraum täglich
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 36
fig, ax1 = plt.subplots(figsize=(30,10))
ax1.bar(completed_data.groupby(completed_data["Datetime"].dt.date)["Datetime"].first(),
        completed_data.groupby(completed_data["Datetime"].dt.date)["Precipitation"].sum(), color="lightgray", edgecolor="dimgray", label="Precipitation", width=0.75)
ax2 = ax1.twinx()
ax2.plot(completed_data.groupby(completed_data["Datetime"].dt.date)["Datetime"].first(),
         completed_data.groupby(completed_data["Datetime"].dt.date)["TA_1_1_2"].mean(), linestyle="--", color="orange", label="Air Temperature")
ax1.set_xlim(completed_data.groupby(completed_data["Datetime"].dt.date)["Datetime"].mean().index.min(),
             completed_data.groupby(completed_data["Datetime"].dt.date)["Datetime"].mean().index.max())
ax1.set_ylabel("Precipitation (mm)")
ax2.set_ylabel("Air Temperature (°C)")
ax1.set_xticklabels(["Aug 2023", "Sep 2023", "Oct 2023", "Nov 2023", "Dec 2023", "Jan 2024", "Feb 2024", "Mar 2024"])
ax1.legend(loc="center", bbox_to_anchor=(0.25, -0.15), ncol = 1, frameon=False)
ax2.legend(loc="center", bbox_to_anchor=(0.75, -0.15), ncol = 1, frameon=False)
#plt.savefig("C:\\Users\\linus\\OneDrive\\Dokumente\\Masterarbeit\\Graphen\\Wetter.pdf", format="pdf", bbox_inches="tight")
plt.tight_layout()
plt.grid()
plt.show()


completed_data.set_index("Datetime", inplace=True)

monthly_precipitation = pd.DataFrame()
monthly_temperature = pd.DataFrame()
monthly_precipitation["Precipitation"] = completed_data["Precipitation"].resample("M").sum()
monthly_temperature["TA_1_1_2"] = completed_data["TA_1_1_2"].resample("M").mean()
monthly_precipitation.index = monthly_precipitation.index.to_period("M").to_timestamp("D")
monthly_temperature.index = monthly_temperature.index.to_period("M").to_timestamp("D")

completed_data.reset_index(inplace=True, drop=False)
# Niederschlag vs TA für Messzeitraum monatlich
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 24
fig, ax1 = plt.subplots(figsize=(15,5))
ax1.bar(monthly_precipitation.index, monthly_precipitation["Precipitation"], color="lightgray", edgecolor="dimgray", label="Precipitation", width=20)
ax2 = ax1.twinx()
ax2.plot(monthly_temperature.index, monthly_temperature["TA_1_1_2"], color="red", marker="o", label="Air Temperature")
ax1.set_xticklabels(monthly_precipitation.index.strftime("%b"))  # Format x-axis tick labels as "Month Year"
ax1.set_ylabel("Precipitation (mm)")
ax2.set_ylabel("Air Temperature (°C)")
ax1.legend(loc="center", bbox_to_anchor=(0.25, -0.25), ncol=1, frameon=False)
ax2.legend(loc="center", bbox_to_anchor=(0.75, -0.25), ncol=1, frameon=False)
plt.title("2023/2024")
plt.tight_layout()
#plt.savefig("C:\\Users\\linus\\OneDrive\\Dokumente\\Masterarbeit\\Graphen\\Wetter monatlich.pdf", format="pdf", bbox_inches="tight")
plt.grid()
plt.show()

# Niederschlag vs TA für vorherige Klimaperiode
klima_t = pd.read_csv("C:\\Users\\linus\\OneDrive\\Dokumente\\Masterarbeit\\Daten\\Klimadaten\\Lufttemperatur\\data\\data_OBS_DEU_PT1H_T2M_662.csv", sep = ",", low_memory=False, index_col=False)
klima_t["Zeitstempel"] = pd.to_datetime(klima_t["Zeitstempel"]) # Date Time Formatierung
klima_t.drop(columns=["Produkt_Code"], inplace=True)
klima_t.set_index("Zeitstempel", inplace=True)
klima_t.index = klima_t.index.to_period("M").to_timestamp("D")
klima_t = klima_t.resample("M").mean()
klima_t["Produkt_Code"] = "OBS_DEU_PT1H_T2M"
klima_t.rename(columns={"Wert": "Lufttemperatur"}, inplace=True)
print(klima_t)

klima_p = pd.read_csv("C:\\Users\\linus\\OneDrive\\Dokumente\\Masterarbeit\\Daten\\Klimadaten\\Niederschlag\\data\\data_OBS_DEU_P1M_RR_662.csv", sep = ",", low_memory=False,  index_col=False)
klima_p["Zeitstempel"] = pd.to_datetime(klima_p["Zeitstempel"]) # Date Time Formatierung
klima_p.drop(columns=["Produkt_Code"], inplace=True)
klima_p.set_index("Zeitstempel", inplace=True)
klima_p.index = klima_p.index.to_period("M").to_timestamp("D")
klima_p = klima_p.resample("M").sum()
klima_p["Produkt_Code"] = "OBS_DEU_P401M_RR"
klima_p.rename(columns={"Wert": "Niederschlag"}, inplace=True)
print(klima_p)


#längen angleichen
klima_p = klima_p.loc[klima_t.index]
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 25
fig, axs = plt.subplots(len(klima_t.index.year.unique())//2, 2, figsize=(20, len(klima_t.index.year.unique())), sharex=True)

for i, year in enumerate(klima_t.index.year.unique()):
    temp_year = klima_t[klima_t.index.year == year]
    precip_year = klima_p[klima_p.index.year == year]
    row, col = i // 2, i % 2
    axs[row, col].bar(temp_year.index.month, precip_year["Niederschlag"], color="lightgrey", edgecolor="dimgray", label="Precipitation", zorder=1)
    ax1 = axs[row, col].twinx()
    ax1.plot(temp_year.index.month, temp_year["Lufttemperatur"], color="red", marker="o", label="Air Temperature", zorder=2)
    axs[row, col].set_title(f"Year {year}")
    axs[row, col].set_xticks(temp_year.index.month[::3])
    axs[row, col].set_xticklabels(temp_year.index.strftime("%b")[::3])
plt.tight_layout()
#plt.savefig("C:\\Users\\linus\\OneDrive\\Dokumente\\Masterarbeit\\Graphen\\Klimadaten.pdf", format="pdf", bbox_inches="tight")
plt.show()


plt.figure(figsize=(15,6))
plt.plot(HP_1.loc[(HP_1["Datetime"].dt.date == pd.to_datetime("2023-08-23").date()) | (HP_1["Datetime"].dt.date == pd.to_datetime("2023-08-22").date()),"Datetime"],
         HP_1.loc[(HP_1["Datetime"].dt.date == pd.to_datetime("2023-08-23").date()) | (HP_1["Datetime"].dt.date == pd.to_datetime("2023-08-22").date()),"SWC_1_1_2"])
plt.xticks(HP_1.loc[(HP_1["Datetime"].dt.date == pd.to_datetime("2023-08-23").date()) | (HP_1["Datetime"].dt.date == pd.to_datetime("2023-08-22").date()),"Datetime"][::12])
plt.grid()
plt.show()

# Bekommen Dachmodule SWC in der Nacht dazu?
results_1_1_1 = []
results_1_1_2 = []
dates = []

daily_min = HP_1.groupby(pd.Grouper(key="Datetime", freq="D")).min().reset_index()
daily_max = HP_1.groupby(pd.Grouper(key="Datetime", freq="D")).max().reset_index()
for i in range(0, len(daily_min)):
    if i >= 1:
        current_day_min = daily_min.iloc[i-1]
        next_day_max = daily_max.iloc[i]
        date = current_day_min["Datetime"]
        result_1_1_1 = next_day_max["SWC_1_1_1"] > current_day_min["SWC_1_1_1"]
        result_1_1_2 = next_day_max["SWC_1_1_2"] > current_day_min["SWC_1_1_2"]
        dates.append(date)
        results_1_1_1.append(result_1_1_1)
        results_1_1_2.append(result_1_1_2)


comparison_df = pd.DataFrame({
    "date": dates,
    "SWC_1_1_1": results_1_1_1,
    "SWC_1_1_2": results_1_1_2
})


print(comparison_df)

results_1_1_1 = []
results_1_1_2 = []
dates = []

daily_min = HP_2.groupby(pd.Grouper(key="Datetime", freq="D")).min().reset_index()
daily_max = HP_2.groupby(pd.Grouper(key="Datetime", freq="D")).max().reset_index()

for i in range(0, len(daily_min)):
    if i >= 1:
        current_day_min = daily_min.iloc[i-1]
        next_day_max = daily_max.iloc[i]
        date = current_day_min["Datetime"]
        result_1_1_1 = next_day_max["SWC_1_1_1"] > current_day_min["SWC_1_1_1"]
        result_1_1_2 = next_day_max["SWC_1_1_2"] > current_day_min["SWC_1_1_2"]
        dates.append(date)
        results_1_1_1.append(result_1_1_1)
        results_1_1_2.append(result_1_1_2)


comparison_df = pd.DataFrame({
    "date": dates,
    "SWC_1_1_1": results_1_1_1,
    "SWC_1_1_2": results_1_1_2
})


print(comparison_df)


results_1_1_1 = []
results_1_1_2 = []
dates = []

daily_min = KP_1.groupby(pd.Grouper(key="Datetime", freq="D")).min().reset_index()
daily_max = KP_1.groupby(pd.Grouper(key="Datetime", freq="D")).max().reset_index()
for i in range(0, len(daily_min)):
    if i >= 1:
        current_day_min = daily_min.iloc[i-1]
        next_day_max = daily_max.iloc[i]
        date = current_day_min["Datetime"]
        result_1_1_1 = next_day_max["SWC_1_1_1"] > current_day_min["SWC_1_1_1"]
        result_1_1_2 = next_day_max["SWC_1_1_2"] > current_day_min["SWC_1_1_2"]
        dates.append(date)
        results_1_1_1.append(result_1_1_1)
        results_1_1_2.append(result_1_1_2)


comparison_df = pd.DataFrame({
    "date": dates,
    "SWC_1_1_1": results_1_1_1,
    "SWC_1_1_2": results_1_1_2
})


print(comparison_df)

results_1_1_1 = []
results_1_1_2 = []
dates = []

daily_min = KP_2.groupby(pd.Grouper(key="Datetime", freq="D")).min().reset_index()
daily_max = KP_2.groupby(pd.Grouper(key="Datetime", freq="D")).max().reset_index()

for i in range(0, len(daily_min)):
    if i >= 1:
        current_day_min = daily_min.iloc[i-1]
        next_day_max = daily_max.iloc[i]
        date = current_day_min["Datetime"]
        result_1_1_1 = next_day_max["SWC_1_1_1"] > current_day_min["SWC_1_1_1"]
        result_1_1_2 = next_day_max["SWC_1_1_2"] > current_day_min["SWC_1_1_2"]
        dates.append(date)
        results_1_1_1.append(result_1_1_1)
        results_1_1_2.append(result_1_1_2)


comparison_df = pd.DataFrame({
    "date": dates,
    "SWC_1_1_1": results_1_1_1,
    "SWC_1_1_2": results_1_1_2
})


print(comparison_df)

# Plots für Section Albedo and Evapotranspiration

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 27
plt.figure(figsize=(17, 7))
plt.plot(completed_data.groupby("Time")["Q*_rad_gr"].mean().index, completed_data.groupby("Time")["Q*_rad_gr"].mean(), label="Q* $_{rad}$ $_{GR}$", color="dodgerblue", linestyle="-", linewidth=2.5)
plt.plot(completed_data.groupby("Time")["Q*_rad_hr"].mean().index, completed_data.groupby("Time")["Q*_rad_hr"].mean(), label="Q* $_{rad}$ $_{HR}$", color="crimson", linestyle="-", linewidth=2.5)
plt.plot(completed_data.groupby("Time")["Q*_flux_gr"].mean().index, completed_data.groupby("Time")["Q*_flux_gr"].mean(), label="Q* $_{flux}$ $_{GR}$", color="goldenrod", linestyle="--", linewidth=2.5)
plt.plot(completed_data.groupby("Time")["Q*_flux_hr"].mean().index, completed_data.groupby("Time")["Q*_flux_hr"].mean(), label="Q* $_{flux}$ $_{HR}$", color="dimgray", linestyle="--", linewidth=2.5)
plt.grid(False)
plt.grid(axis="y")
plt.xlabel("Hour of the Day")
plt.ylabel("$W/m^2$")
plt.xlim(completed_data.groupby("Time")["SW_IN"].mean().index.min(), completed_data.groupby("Time")["SW_IN"].mean().index.max())
plt.xticks(completed_data.groupby("Time")["SW_IN"].mean().index[::6], labels=timelabels)
plt.legend(loc="center", bbox_to_anchor=(0.5, -0.2), ncol=4, frameon=False)
#plt.savefig("C:\\Users\\linus\\OneDrive\\Dokumente\\Masterarbeit\\Graphen\\MDC rad flux dif.pdf", format="pdf", bbox_inches="tight")
plt.show()


plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 27
plt.figure(figsize=(17,7))
plt.plot(completed_data.groupby("Time")["QE_HR"].mean().index, completed_data.groupby("Time")["QE_HR"].mean(), label="Q$_E$ $_{HR}$", color="dodgerblue", linestyle="-", linewidth=2.5)
plt.plot(completed_data.groupby("Time")["SW_OUT_HR_Calculated"].mean().index, completed_data.groupby("Time")["SW_OUT_HR_Calculated"].mean(), label="SW$_{OUT}$ $_{HR}$", color="crimson", linestyle="--", linewidth=2.5)
plt.plot(completed_data.groupby("Time")["QE_GR"].mean().index, completed_data.groupby("Time")["QE_GR"].mean(), label="Q$_E$ $_{HR}$", color="goldenrod", linestyle="-", linewidth=2.5)
plt.plot(completed_data.groupby("Time")["SW_OUT_GR_Calculated"].mean().index, completed_data.groupby("Time")["SW_OUT_GR_Calculated"].mean(), label="SW$_{OUT}$ $_{HR}$", color="dimgray", linestyle="--", linewidth=2.5)
plt.legend(loc="center", bbox_to_anchor=(0.5, -0.2), ncol = 4, frameon=False)
plt.xticks(completed_data.groupby("Time")["QE_HR"].mean().index[::6],labels=timelabels)
plt.xlim(completed_data.groupby("Time")["QE_HR"].mean().index.min(),completed_data.groupby("Time")["QE_HR"].mean().index.max())
plt.xlabel("Hour of Day")
plt.ylabel("W/$m^2$")
plt.grid(False)
plt.grid(axis="y")
#plt.savefig("C:\\Users\\linus\\OneDrive\\Dokumente\\Masterarbeit\\Graphen\\MDC QE vs SW.pdf", format="pdf", bbox_inches="tight")
plt.show()

subset = completed_data.dropna(subset=["QE_GR", "QE_HR", "LW_IN", "SW_IN", "SW_OUT_GR_Calculated", "SW_OUT_HR_Calculated"])

print(((completed_data["QE_GR"].mean()+completed_data["SW_OUT_GR_Calculated"].mean())/completed_data["SW_IN"].mean())*100)
print((completed_data["QE_GR"].mean()+completed_data["SW_OUT_GR_Calculated"].mean()))

print(((completed_data["QE_HR"].mean()+completed_data["SW_OUT_HR_Calculated"].mean())/completed_data["SW_IN"].mean())*100)
print((completed_data["QE_HR"].mean()+completed_data["SW_OUT_HR_Calculated"].mean()))


print(((subset["QE_GR"].mean()+subset["SW_OUT_GR_Calculated"].mean())/subset["SW_IN"].mean())*100)
print((subset["QE_GR"].mean()+subset["SW_OUT_GR_Calculated"].mean()))

print(((subset["QE_HR"].mean()+subset["SW_OUT_HR_Calculated"].mean())/subset["SW_IN"].mean())*100)
print((subset["QE_HR"].mean()+subset["SW_OUT_HR_Calculated"].mean()))

print((((completed_data[(completed_data["Datetime"].dt.hour >= 10) &
              (completed_data["Datetime"].dt.hour <= 16)]["QE_GR"]).mean()+ 
       completed_data[(completed_data["Datetime"].dt.hour >= 10) &
              (completed_data["Datetime"].dt.hour <= 16)]["SW_OUT_GR_Calculated"].mean())/
             completed_data[(completed_data["Datetime"].dt.hour >= 10) &
              (completed_data["Datetime"].dt.hour <= 16)]["SW_IN"].mean())*100)

print((((completed_data[(completed_data["Datetime"].dt.hour >= 10) &
              (completed_data["Datetime"].dt.hour <= 16)]["QE_HR"]).mean()+ 
       completed_data[(completed_data["Datetime"].dt.hour >= 10) &
              (completed_data["Datetime"].dt.hour <= 16)]["SW_OUT_HR_Calculated"].mean())/
             completed_data[(completed_data["Datetime"].dt.hour >= 10) &
              (completed_data["Datetime"].dt.hour <= 16)]["SW_IN"].mean())*100)

print((((subset[(subset["Datetime"].dt.hour >= 10) &
              (subset["Datetime"].dt.hour <= 16)]["QE_GR"]).mean()+ 
       subset[(subset["Datetime"].dt.hour >= 10) &
              (subset["Datetime"].dt.hour <= 16)]["SW_OUT_GR_Calculated"].mean())/
             subset[(subset["Datetime"].dt.hour >= 10) &
              (subset["Datetime"].dt.hour <= 16)]["SW_IN"].mean())*100)

print((((subset[(subset["Datetime"].dt.hour >= 10) &
              (subset["Datetime"].dt.hour <= 16)]["QE_HR"]).mean()+ 
       subset[(subset["Datetime"].dt.hour >= 10) &
              (subset["Datetime"].dt.hour <= 16)]["SW_OUT_HR_Calculated"].mean())/
             subset[(subset["Datetime"].dt.hour >= 10) &
              (subset["Datetime"].dt.hour <= 16)]["SW_IN"].mean())*100)

print(completed_data["SW_OUT_GR_Calculated"].mean())
print(completed_data["SW_OUT_HR_Calculated"].mean())
print(completed_data[(completed_data["Datetime"].dt.hour >= 10)&(completed_data["Datetime"].dt.hour <= 16)]["SW_OUT_GR_Calculated"].mean())
print(completed_data[(completed_data["Datetime"].dt.hour >= 10)&(completed_data["Datetime"].dt.hour <= 16)]["SW_OUT_HR_Calculated"].mean())

print(subset["SW_OUT_GR_Calculated"].mean())
print(subset["SW_OUT_HR_Calculated"].mean())
print(subset[(subset["Datetime"].dt.hour >= 10)&(subset["Datetime"].dt.hour <= 16)]["SW_OUT_GR_Calculated"].mean())
print(subset[(subset["Datetime"].dt.hour >= 10)&(subset["Datetime"].dt.hour <= 16)]["SW_OUT_HR_Calculated"].mean()) 

print(completed_data["QE_GR"].mean())
print(completed_data["QE_HR"].mean())
print(completed_data[(completed_data["Datetime"].dt.hour >= 10)&(completed_data["Datetime"].dt.hour <= 16)]["QE_GR"].mean())
print(completed_data[(completed_data["Datetime"].dt.hour >= 10)&(completed_data["Datetime"].dt.hour <= 16)]["QE_HR"].mean())

print(subset["QE_GR"].mean())
print(subset["QE_HR"].mean())
print(subset[(subset["Datetime"].dt.hour >= 10)&(subset["Datetime"].dt.hour <= 16)]["QE_GR"].mean())
print(subset[(subset["Datetime"].dt.hour >= 10)&(subset["Datetime"].dt.hour <= 16)]["QE_HR"].mean())

print(completed_data["QE_HR"].mean()/completed_data["QE_GR"].mean())
print(completed_data[(completed_data["Datetime"].dt.hour >= 10)&(completed_data["Datetime"].dt.hour <= 16)]["QE_HR"].mean()/completed_data[(completed_data["Datetime"].dt.hour >= 10)&(completed_data["Datetime"].dt.hour <= 16)]["QE_GR"].mean())
print(subset["QE_HR"].mean()/subset["QE_GR"].mean())
print(subset[(subset["Datetime"].dt.hour >= 10)&(subset["Datetime"].dt.hour <= 16)]["QE_HR"].mean()/subset[(subset["Datetime"].dt.hour >= 10)&(subset["Datetime"].dt.hour <= 16)]["QE_GR"].mean())

print(completed_data["SW_OUT_HR_Calculated"].mean()/completed_data["SW_OUT_GR_Calculated"].mean())
print(completed_data[(completed_data["Datetime"].dt.hour >= 10)&(completed_data["Datetime"].dt.hour <= 16)]["SW_OUT_HR_Calculated"].mean()/completed_data[(completed_data["Datetime"].dt.hour >= 10)&(completed_data["Datetime"].dt.hour <= 16)]["SW_OUT_GR_Calculated"].mean())
print(subset["SW_OUT_HR_Calculated"].mean()/subset["SW_OUT_GR_Calculated"].mean())
print(subset[(subset["Datetime"].dt.hour >= 10)&(subset["Datetime"].dt.hour <= 16)]["SW_OUT_HR_Calculated"].mean()/subset[(subset["Datetime"].dt.hour >= 10)&(subset["Datetime"].dt.hour <= 16)]["SW_OUT_GR_Calculated"].mean())

print(completed_data["QE_GR"].mean())
print(completed_data["QE_HR"].mean())
print(completed_data[(completed_data["Datetime"].dt.hour >= 10)&(completed_data["Datetime"].dt.hour <= 16)]["QE_GR"].mean())
print(completed_data[(completed_data["Datetime"].dt.hour >= 10)&(completed_data["Datetime"].dt.hour <= 16)]["QE_HR"].mean())

print(completed_data["TS_CS65X_1_1_1"].mean())
print(completed_data["TS_CS65X_1_1_2"].mean())
print(completed_data[(completed_data["Datetime"].dt.hour >= 10)&(completed_data["Datetime"].dt.hour <= 16)]["TS_CS65X_1_1_1"].mean())
print(completed_data[(completed_data["Datetime"].dt.hour >= 10)&(completed_data["Datetime"].dt.hour <= 16)]["TS_CS65X_1_1_2"].mean())

print(subset["TS_CS65X_1_1_1"].mean())
print(subset["TS_CS65X_1_1_2"].mean())
print(subset[(subset["Datetime"].dt.hour >= 10)&(subset["Datetime"].dt.hour <= 16)]["TS_CS65X_1_1_1"].mean())
print(subset[(subset["Datetime"].dt.hour >= 10)&(subset["Datetime"].dt.hour <= 16)]["TS_CS65X_1_1_2"].mean())

print((completed_data["TS_CS65X_1_1_1"]-completed_data["TS_CS65X_1_1_2"]).mean())
print((completed_data[(completed_data["Datetime"].dt.hour >= 10)&(completed_data["Datetime"].dt.hour <= 16)]["TS_CS65X_1_1_1"]-
       completed_data[(completed_data["Datetime"].dt.hour >= 10)&(completed_data["Datetime"].dt.hour <= 16)]["TS_CS65X_1_1_2"]).mean())
print((completed_data[(completed_data["Datetime"].dt.hour < 10)|(completed_data["Datetime"].dt.hour > 16)]["TS_CS65X_1_1_1"] - 
       completed_data[(completed_data["Datetime"].dt.hour < 10)|(completed_data["Datetime"].dt.hour > 16)]["TS_CS65X_1_1_2"]).mean())


print((KP_1["TS_CS65X_1_1_1"]-KP_1["TS_CS65X_1_1_2"]).mean())
print((KP_1[(KP_1["Datetime"].dt.hour >= 10)&(KP_1["Datetime"].dt.hour <= 16)]["TS_CS65X_1_1_1"]-KP_1[(KP_1["Datetime"].dt.hour >= 10)&(KP_1["Datetime"].dt.hour <= 16)]["TS_CS65X_1_1_2"]).mean())
print((KP_1[(KP_1["Datetime"].dt.hour < 10)|(KP_1["Datetime"].dt.hour > 16)]["TS_CS65X_1_1_1"].mean()-KP_1[(KP_1["Datetime"].dt.hour < 10)|(KP_1["Datetime"].dt.hour > 16)]["TS_CS65X_1_1_2"]).mean())

print((KP_2["TS_CS65X_1_1_1"]-KP_2["TS_CS65X_1_1_2"]).mean())
print((KP_2[(KP_2["Datetime"].dt.hour >= 10)&(KP_2["Datetime"].dt.hour <= 16)]["TS_CS65X_1_1_1"]-KP_2[(KP_2["Datetime"].dt.hour >= 10)&(KP_2["Datetime"].dt.hour <= 16)]["TS_CS65X_1_1_2"]).mean())
print((KP_2[(KP_2["Datetime"].dt.hour < 10)|(KP_2["Datetime"].dt.hour > 16)]["TS_CS65X_1_1_1"].mean()-KP_2[(KP_2["Datetime"].dt.hour < 10)|(KP_2["Datetime"].dt.hour > 16)]["TS_CS65X_1_1_2"]).mean())


print(subset["TS_CS65X_1_1_1"].std())
print(subset["TS_CS65X_1_1_2"].std())
print(subset[(subset["Datetime"].dt.hour >= 10)&(subset["Datetime"].dt.hour <= 16)]["TS_CS65X_1_1_1"].std())
print(subset[(subset["Datetime"].dt.hour >= 10)&(subset["Datetime"].dt.hour <= 16)]["TS_CS65X_1_1_2"].std())

print(completed_data["TS_CS65X_1_1_1"].std())
print(completed_data["TS_CS65X_1_1_2"].std())
print(completed_data[(completed_data["Datetime"].dt.hour >= 10)&(completed_data["Datetime"].dt.hour <= 16)]["TS_CS65X_1_1_1"].std())
print(completed_data[(completed_data["Datetime"].dt.hour >= 10)&(completed_data["Datetime"].dt.hour <= 16)]["TS_CS65X_1_1_2"].std())


print(KP_1["TS_CS65X_1_1_1"].std())
print(KP_1["TS_CS65X_1_1_2"].std())
print(KP_1[(KP_1["Datetime"].dt.hour >= 10)&(KP_1["Datetime"].dt.hour <= 16)]["TS_CS65X_1_1_1"].std())
print(KP_1[(KP_1["Datetime"].dt.hour >= 10)&(KP_1["Datetime"].dt.hour <= 16)]["TS_CS65X_1_1_2"].std())

print(KP_2["TS_CS65X_1_1_1"].std())
print(KP_2["TS_CS65X_1_1_2"].std())
print(KP_2[(KP_2["Datetime"].dt.hour >= 10)&(KP_2["Datetime"].dt.hour <= 16)]["TS_CS65X_1_1_1"].std())
print(KP_2[(KP_2["Datetime"].dt.hour >= 10)&(KP_2["Datetime"].dt.hour <= 16)]["TS_CS65X_1_1_2"].std())


# durchschnittliche Albedo von weißen Cool Roofs berechnen, Datenbank aus coolroofs.org (CRRC Cool Roofs)

cr_albedo = pd.read_csv("C:\\Users\\linus\\OneDrive\\Dokumente\\Publikation\\Directory_Roofs_20240809-0100.csv", sep = ",", low_memory=False)

print(cr_albedo.info())
cr_albedo = cr_albedo[cr_albedo["Color"].str.contains("White")==True]
cr_albedo["3 Year Solar Reflectance"] = cr_albedo["3 Year Solar Reflectance"].str.replace("*", "", regex=False)
cr_albedo["3 Year Solar Reflectance"] = pd.to_numeric(cr_albedo["3 Year Solar Reflectance"])
print(cr_albedo.info())
print(cr_albedo["Initial Solar Reflectance"].mean()) # 0.75
print(cr_albedo["3 Year Solar Reflectance"].mean()) # 0.66

# with removing all the rows that contain a * in 3 Year Solar Reflections and that contain NaN in said column
cr_albedo = cr_albedo[cr_albedo["3 Year Solar Reflectance"].isna()==False]
cr_albedo = cr_albedo[~cr_albedo["3 Year Solar Reflectance"].str.contains("*", regex=False)]
cr_albedo["3 Year Solar Reflectance"] = pd.to_numeric(cr_albedo["3 Year Solar Reflectance"])
print(cr_albedo["Initial Solar Reflectance"].mean()) # 0.77
print(cr_albedo["3 Year Solar Reflectance"].mean()) # 0.66

# mean Air Temp and Precipitation sums per month
# precipitation to follow
print(monthly_temperature)



# Surface Temp Plot for Paper (measured Surface Temp vs TA vs SW IN)
st_data = pd.read_csv("C:\\Users\\linus\\OneDrive\\Dokumente\\Masterarbeit\\Daten\\Oberflächentemperatur Messung\\st_data.csv", sep = ",", low_memory=False)
st_data["Datetime"] =  pd.to_datetime(st_data["Datetime"])
merged_data = pd.merge(completed_data[["Datetime", "TA_1_1_2", "SW_IN"]], st_data[["Datetime", "GD_MW", "HD_MW"]], on="Datetime", how="left")
completed_data["GD_MW"] = merged_data["GD_MW"]
completed_data["HD_MW"] = merged_data["HD_MW"]


plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 24
timelabels = (["00:00", "04:00", "08:00", "12:00", "16:00", "20:00"])
fig, (ax1, ax3, ax5) = plt.subplots(3, 1, figsize=(13,10))

ax1.plot(completed_data.loc[completed_data["Datetime"].dt.date == pd.to_datetime("2023-08-11").date(), "Time"],
        completed_data.loc[completed_data["Datetime"].dt.date == pd.to_datetime("2023-08-11").date(), "TA_1_1_2"], 
        label="$T_A$", color="C3")
ax1.scatter(completed_data.loc[completed_data["Datetime"].dt.date == pd.to_datetime("2023-08-11").date(), "Time"], 
            completed_data.loc[completed_data["Datetime"].dt.date == pd.to_datetime("2023-08-11").date(), "GD_MW"], 
            label="$T_S$ GR", color="C2")
ax1.scatter(completed_data.loc[completed_data["Datetime"].dt.date == pd.to_datetime("2023-08-11").date(), "Time"],
            completed_data.loc[completed_data["Datetime"].dt.date == pd.to_datetime("2023-08-11").date(), "HD_MW"], 
            label="$T_S$ HR", color="C0")
ax2 = ax1.twinx()
ax2.plot(completed_data.loc[completed_data["Datetime"].dt.date == pd.to_datetime("2023-08-11").date(), "Time"],
        completed_data.loc[completed_data["Datetime"].dt.date == pd.to_datetime("2023-08-11").date(), "SW_IN"], 
        color="gold", label="$SW \downarrow$")

ax3.plot(completed_data.loc[completed_data["Datetime"].dt.date == pd.to_datetime("2023-08-24").date(), "Time"],
        completed_data.loc[completed_data["Datetime"].dt.date == pd.to_datetime("2023-08-24").date(), "TA_1_1_2"], 
        label="$T_A$", color="C3")
ax3.scatter(completed_data.loc[completed_data["Datetime"].dt.date == pd.to_datetime("2023-08-24").date(), "Time"],
            completed_data.loc[completed_data["Datetime"].dt.date == pd.to_datetime("2023-08-24").date(), "GD_MW"], 
            label="$T_S$ GR", color="C2")
ax3.scatter(completed_data.loc[completed_data["Datetime"].dt.date == pd.to_datetime("2023-08-24").date(), "Time"],
            completed_data.loc[completed_data["Datetime"].dt.date == pd.to_datetime("2023-08-24").date(), "HD_MW"], 
            label="$T_S$ HR", color="C0")
ax4 = ax3.twinx()
ax4.plot(completed_data.loc[completed_data["Datetime"].dt.date == pd.to_datetime("2023-08-24").date(), "Time"],
        completed_data.loc[completed_data["Datetime"].dt.date == pd.to_datetime("2023-08-24").date(), "SW_IN"], 
        color="gold", label="$SW \downarrow$")

ax5.plot(completed_data.loc[completed_data["Datetime"].dt.date == pd.to_datetime("2024-01-29").date(), "Time"],
        completed_data.loc[completed_data["Datetime"].dt.date == pd.to_datetime("2024-01-29").date(), "TA_1_1_2"], 
        label="$T_A$", color="C3")
ax5.scatter(completed_data.loc[completed_data["Datetime"].dt.date == pd.to_datetime("2024-01-29").date(), "Time"],
            completed_data.loc[completed_data["Datetime"].dt.date == pd.to_datetime("2024-01-29").date(), "GD_MW"], 
            label="$T_S$ GR", color="C2")
ax5.scatter(completed_data.loc[completed_data["Datetime"].dt.date == pd.to_datetime("2024-01-29").date(), "Time"],
            completed_data.loc[completed_data["Datetime"].dt.date == pd.to_datetime("2024-01-29").date(), "HD_MW"], 
            label="$T_S$ HR", color="C0")

ax6 = ax5.twinx()
ax6.plot(completed_data.loc[completed_data["Datetime"].dt.date == pd.to_datetime("2024-01-29").date(), "Time"],
        completed_data.loc[completed_data["Datetime"].dt.date == pd.to_datetime("2024-01-29").date(), "SW_IN"], 
        color="gold", label="$SW \downarrow$")

ax5.legend(loc='center', bbox_to_anchor=(0.375, -0.7), ncol=3, frameon=False)
ax6.legend(loc='center', bbox_to_anchor=(0.85, -0.7), ncol=1, frameon=False)
ax3.set_ylabel("Temperature (°C)")
ax4.set_ylabel("SW $\downarrow$ ($W/m²$)")
ax5.set_xlabel("Time of Day")
ax1.set_xticks(completed_data.loc[completed_data["Datetime"].dt.date == pd.to_datetime("2024-01-29").date(), "Time"][::8], labels=[])
ax3.set_xticks(completed_data.loc[completed_data["Datetime"].dt.date == pd.to_datetime("2024-01-29").date(), "Time"][::8], labels=[])
ax5.set_xticks(completed_data.loc[completed_data["Datetime"].dt.date == pd.to_datetime("2024-01-29").date(), "Time"][::8], labels=timelabels)
ax1.set_xlim("00:00:00","23:30:00")
ax2.set_xlim("00:00:00","23:30:00")
ax3.set_xlim("00:00:00","23:30:00")
ax4.set_xlim("00:00:00","23:30:00")
ax5.set_xlim("00:00:00","23:30:00")
ax6.set_xlim("00:00:00","23:30:00")
ax1.set_title("11.08.2023", fontsize=24)
ax3.set_title("24.08.2023", fontsize=24)
ax5.set_title("29.01.2024", fontsize=24)
plt.tight_layout()
plt.savefig("C:\\Users\\linus\\OneDrive\\Dokumente\\Publikation\\Plots\\surface_temps.pdf", format="pdf", bbox_inches='tight')
plt.show()

# VWC compare and QE compare during a heat wave
fig, (ax1, ax2) = plt.subplots(2,1,figsize=(13,10))
ax1.plot(HP_2["Datetime"], HP_2["VWC_1_1_1"],linewidth = 2,color="C2")
ax1.plot(HP_2["Datetime"], HP_2["VWC_1_1_2"],linewidth = 2,color="C0")
ax1.set_ylabel("VWC ($m³/m³$)")
ax2.plot(HP_2["Datetime"] , HP_2["QE_GR"],linewidth = 2,color="C2", label="Green Roof")
ax2.plot(HP_2["Datetime"] , HP_2["QE_HR"],linewidth = 2,color="C0", label="Green Roof")
ax2.set_ylabel("Q$_E$ ($W/m²$)")
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%d. %b"))
ax1.set_xticklabels([])
ax2.legend(loc='center', bbox_to_anchor=(0.5, -0.25), ncol=2, frameon=False)
ax1.set_xlim(HP_2["Datetime"].min(), HP_2["Datetime"].max())
ax2.set_xlim(HP_2["Datetime"].min(), HP_2["Datetime"].max())
plt.tight_layout()
plt.savefig("C:\\Users\\linus\\OneDrive\\Dokumente\\Publikation\\Plots\\vwc_qe_timeseries.pdf", format="pdf", bbox_inches='tight')
plt.show()

# VWC timeseries

fig, ax1 = plt.subplots(figsize=(30,5))
ax1.bar(completed_data.groupby(completed_data["Datetime"].dt.date)["Datetime"].first(),
        completed_data.groupby(completed_data["Datetime"].dt.date)["Precipitation"].sum(), color="lightgray", edgecolor="dimgray", label="Precipitation", width=0.75)
ax2 = ax1.twinx()
ax2.plot(completed_data["Datetime"], completed_data["VWC_1_1_1"], color="C2", label="Green Roof")
ax2.plot(completed_data["Datetime"], completed_data["VWC_1_1_2"], color="C0", label="Hybrid Roof")
ax1.set_xlim(completed_data["Datetime"].min(), completed_data["Datetime"].max())
ax2.set_xlim(completed_data["Datetime"].min(), completed_data["Datetime"].max())
ax1.set_xticklabels(["Aug 2023", "Sep 2023", "Oct 2023", "Nov 2023", "Dec 2023", "Jan 2024", "Feb 2024", "Mar 2024"])
ax1.set_ylabel("Precipitation (mm)")
ax2.set_ylabel("VWC ($m³/m³$)")
ax1.legend(loc='center', bbox_to_anchor=(0.5, -0.4), ncol=1, frameon=False)
ax2.legend(loc='center', bbox_to_anchor=(0.5, -0.25), ncol=2, frameon=False)
#plt.tight_layout()
plt.savefig("C:\\Users\\linus\\OneDrive\\Dokumente\\Publikation\\Plots\\vwc_precipitation_timeseries_for_appendix.pdf", format="pdf", bbox_inches='tight')
plt.show()
