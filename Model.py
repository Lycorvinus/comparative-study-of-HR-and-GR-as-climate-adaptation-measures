import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import MonthLocator, DateFormatter
import statsmodels.api as sm
import statsmodels.formula.api as smf
import seaborn as sns
from mpl_toolkits.mplot3d import axes3d
import sklearn
from sklearn.linear_model import LinearRegression
from mlxtend.feature_selection import ExhaustiveFeatureSelector
from sklearn.preprocessing import StandardScaler


# import logger_data
logger_data = pd.read_csv("C:\\Users\\linus\\OneDrive\\Dokumente\\Publikation\\Data\\logger_data.csv", sep = ",", low_memory=False)
logger_data["Datetime"] = pd.to_datetime(logger_data["Datetime"]) # Date Time Formatierung
logger_data["Date"] = pd.to_datetime(logger_data["Datetime"]).dt.date
logger_data["Time"] = pd.to_datetime(logger_data["Datetime"]).dt.time
logger_data["Date"] = pd.to_datetime(logger_data["Date"], format = "%Y-%m-%d").dt.date
logger_data["Time"] = pd.to_datetime(logger_data["Time"], format="%H:%M:%S").dt.strftime("%H:%M:%S")
print(logger_data)

#leere Werte mit NAN ersetzen
logger_data.replace("NAN", np.nan, inplace=True)


# Spalten zu numeric convertieren
selected_columns = ["TA_1_1_2", "RH_1_1_2", "TS_CS65X_1_1_1", "TS_CS65X_1_1_2", "SWC_1_1_1", "SWC_1_1_2", "G_plate_1_1_1", "G_plate_1_1_2",
                    "WS", "SW_IN", "SW_OUT", "LW_IN", "LW_OUT"]
logger_data[selected_columns] = logger_data[selected_columns].apply(pd.to_numeric)
print(logger_data.dtypes)
logger_data.set_index("Datetime", inplace=True)
start = logger_data.index.min()
end = logger_data.index.max()
new_index = pd.date_range(start=start, end=end, freq="30min")
logger_data = logger_data.reindex(new_index)
logger_data.index.name = "Datetime"


# Import de Oberflächentemperaturmessungen
surface_temp_data = pd.read_csv("C:\\Users\\linus\\OneDrive\\Dokumente\\Publikation\\Data\\Oberflächentemperatur Messung\\surface_temp_data.csv", sep=",", header = 0, low_memory = False)
st_data = pd.read_csv("C:\\Users\\linus\\OneDrive\\Dokumente\\Publikation\\Data\\Oberflächentemperatur Messung\\st_data.csv", sep = ",", header = 0, low_memory = False)
st_data["Datetime"] = pd.to_datetime(st_data["Datetime"])
st_data.set_index("Datetime", inplace = True)
print(surface_temp_data)
print(st_data)
print(st_data.dtypes)
print(type(st_data))

# descriptive stats um Ausreißer zu identifizieren
descriptive_stats = logger_data[selected_columns]
descriptive_stats = descriptive_stats.describe()
descriptive_stats.to_csv("C:\\Users\\linus\\OneDrive\\Dokumente\\Publikation\\Data\\descriptive_stats.csv", sep = ";", decimal=",", header = True)



# Outlier Treatment for G plate 1 1 2, LW_IN
# G plate 1 1 2
low_threshold = logger_data["G_plate_1_1_2"].quantile(0.005)
high_threshold = logger_data["G_plate_1_1_2"].quantile(0.995)

# finden und gleichzeitig entfernen
# print("G_plate_1_1_2 outliers:")
for index, value in logger_data["G_plate_1_1_2"].items():
    if value < low_threshold or value > high_threshold:
        print("Index:", index, "Value:", value)
        logger_data.loc[index, "G_plate_1_1_2"] = pd.NA



# plt.figure(figsize=(15,5))
# plt.scatter(logger_data.index, logger_data["G_plate_1_1_2"])
# plt.grid()
# plt.show()


# LW IN
outliers_LW_IN = []
window_size = 5  

# print("LW_IN local outliers:")
for i, (index, value) in enumerate(logger_data["LW_IN"].items()):
    if i < window_size:
        continue  
    window = logger_data["LW_IN"].iloc[i - window_size:i] 
    window_mean = window.mean()
    window_std = window.std()
    if value > (window_mean + 2 * window_std): 
        outliers_LW_IN.append(index)

# # print("DateTime indices of LW_IN local outliers:")
# # print(outliers_LW_IN)

# # Ausreißer LW_IN entfertnen
logger_data.loc[outliers_LW_IN, "LW_IN"] = pd.NA


# plt.figure(figsize=(15,5))
# plt.scatter(logger_data.index, logger_data["LW_IN"])
# plt.grid()
# plt.show()

# K down mit negativen Zahlen -> nicht möglich -> durch NaNs ersetzen
logger_data.loc[logger_data["SW_IN"] < 0, "SW_IN"] = logger_data["SW_IN"].interpolate(method="linear").clip(lower=0)
# dynamische Werte für Prandtl Zahl, Dichte Luft und kinematische Viskosität Luft
dynamic_air_values = pd.DataFrame()
dynamic_air_values["Temperature"] = [-20, -10, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
dynamic_air_values["Density"] = [-1394, -1341, 1292, 1269, 1246, 1225, 1204, 1184, 1164, 1145, 1127, 1109, 1092]
dynamic_air_values["kinVis"] = [1.169*10**(-5), 1.252*10**(-5), 1.338*10**(-5), 1.382*10**(-5), 1.426*10**(-5), 1.470*10**(-5), 1.516*10**(-5), 1.562*10**(-5), 1.608*10**(-5), 1.655*10**(-5), 1.702*10**(-5), 1.750*10**(-5), 1.798*10**(-5)]
dynamic_air_values["Prandtl"] = [0.7408, 0.7387, 0.7362, 0.7350, 0.7336, 0.7323, 0.7309, 0.7296, 0.7282, 0.7268, 0.7255, 0.7241, 0.7228]
dynamic_air_values["lambda"] = [0.02211, 0.02288, 0.02364, 0.02401, 0.02439, 0.02476, 0.02514, 0.02551, 0.02588, 0.02625, 0.02662, 0.02699, 0.02735]
dynamic_air_values["specific_heat_cap_air"] = [1005, 1006, 1006, 1006, 1006, 1007, 1007, 1007, 1007, 1007, 1007, 1007, 1007]


logger_data["Air_Density"] = np.interp(logger_data["TA_1_1_2"], dynamic_air_values["Temperature"], dynamic_air_values["Density"])
kinVis_slope = (dynamic_air_values["kinVis"].iloc[-1] - dynamic_air_values["kinVis"].iloc[0]) / (dynamic_air_values["Temperature"].iloc[-1] - dynamic_air_values["Temperature"].iloc[0])
logger_data["Air_kinVis"] = dynamic_air_values["kinVis"].iloc[0] + kinVis_slope * (logger_data["TA_1_1_2"] - dynamic_air_values["Temperature"].iloc[0])
logger_data["Prandtl"] = np.interp(logger_data["TA_1_1_2"], dynamic_air_values["Temperature"], dynamic_air_values["Prandtl"])
lambda_slope = (dynamic_air_values["lambda"].iloc[-1] - dynamic_air_values["lambda"].iloc[0]) / (dynamic_air_values["Temperature"].iloc[-1] - dynamic_air_values["Temperature"].iloc[0])
logger_data["Air_Lambda"] = dynamic_air_values["lambda"].iloc[0] + kinVis_slope * (logger_data["TA_1_1_2"] - dynamic_air_values["Temperature"].iloc[0])
logger_data["specific_heat_cap_air"] = np.interp(logger_data["TA_1_1_2"], dynamic_air_values["Temperature"], dynamic_air_values["specific_heat_cap_air"])
# plt.figure(figsize=(10,6))
# plt.plot(dynamic_air_values["Temperature"], dynamic_air_values["Prandtl"])
# plt.plot(logger_data["TA_1_1_2"], logger_data["Prandtl"])
# plt.grid()
# plt.show()

# plt.figure(figsize=(10,6))
# plt.plot(logger_data["TA_1_1_2"], logger_data["Prandtl"])
# plt.grid()
# plt.show()

mdata = pd.read_csv("C:\\Users\\linus\\OneDrive\\Dokumente\\Publikation\\Data\\mdata.csv", sep = ",", low_memory=False)
mdata["Datetime"] = pd.to_datetime(mdata["Datetime"].str.strip(), yearfirst=True, format="%Y-%m-%d %H:%M:%S") # Date Time Formatierung
print(mdata)

mdata["Precipitation"] = mdata["Precipitation"].apply(pd.to_numeric)
mdata.set_index("Datetime", inplace=True)
logger_data["Precipitation"] = mdata["Precipitation"]

# SW OUT Berechnung durch Albedo
# Albedo ab Beginn
# a_gr = 13.970398943608982, a_hr = 49.841498030046324
# Albedo am 24.11.
# a_gr2 = 11.711143334838868, a_hr2 = 38.051069209126446
# Albedo am 29.01
# hr: 40.181844 gr: 12.005754
# Albedo am 08.03.
# hr:41.656481 gr:16.648328
# plt.figure(figsize=(75,5))
# plt.plot(logger_data.index, logger_data["SW_IN"], linestyle="-.")
# plt.grid()
# plt.show()



logger_data["Albedo_GR"] = 0.0
logger_data["Albedo_HR"] = 0.0
logger_data.loc[logger_data.index < "2023-11-01", "Albedo_GR"] = 13.970398943608982/100
logger_data.loc[logger_data.index < "2023-11-01", "Albedo_HR"] = 49.841498030046324/100

logger_data.loc[(logger_data.index >= "2023-11-01") & (logger_data.index < "2024-01-01"), "Albedo_GR"] = 11.711143334838868/100
logger_data.loc[(logger_data.index >= "2023-11-01") & (logger_data.index < "2024-01-01"), "Albedo_HR"] = 38.051069209126446/100

logger_data.loc[(logger_data.index >= "2024-01-01") & (logger_data.index < "2024-03-01"), "Albedo_GR"] = 12.00575/100
logger_data.loc[(logger_data.index >= "2024-01-01") & (logger_data.index < "2024-03-01"), "Albedo_HR"] = 40.181844/100

logger_data.loc[logger_data.index >= "2024-03-01", "Albedo_GR"] = 16.648328/100
logger_data.loc[logger_data.index >= "2024-03-01", "Albedo_HR"] = 41.656481/100
swo_gr = pd.DataFrame(columns=["SW_OUT"])
swo_hr = pd.DataFrame(columns=["SW_OUT"])
swo_gr["SW_OUT"] = logger_data["SW_IN"] * logger_data["Albedo_GR"]
swo_hr["SW_OUT"] = logger_data["SW_IN"] * logger_data["Albedo_HR"]
swo_gr["Time"] = logger_data["Time"]
swo_gr["Datetime"] = logger_data.index
swo_hr["Datetime"] = logger_data.index
swo_hr["Time"] = logger_data["Time"]
logger_data["SW_OUT_GR_Calculated"] = swo_gr["SW_OUT"]
logger_data["SW_OUT_HR_Calculated"] = swo_hr["SW_OUT"]


# Wassergehalt für 2 Punkt Kalibrierung berechnen
w_gr = 8 -(0.38 + 0.28) # Wasservolumen im GR nach Vorsättigung von GR
bf_gr = 9.7 # angezeigter Bodenwassergehalt nach Vorsättigung
w_hr = 18 - (0.43 + 0.07) # Wasservolumen im HR nach Vorsättigung von HR
bf_hr = 48.2 # angezeigter Bodenwassergehalt nach Vorsättigung

w_gr = (w_gr/1000)
w_hr = (w_hr/1000)

w_gr = w_gr/(1*0.45*0.06)
w_hr = w_hr/(1*0.45*0.06)

def line_equation(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    
    slope = (y2 - y1) / (x2 - x1)
    
    intercept = y1 - slope * x1

    return f"y = {slope}x + {intercept}"


x_gr = [w_gr, bf_gr]
y_gr = [0, 0]

x_hr = [w_hr, bf_hr]
y_hr = [0, 0]

kalibrierung_gr= line_equation(x_gr, y_gr)
print("Equation of the line:", kalibrierung_gr)
slope_gr = (y_gr[1] - x_gr[1])/(y_gr[0] - x_gr[0])
print(slope_gr)

kalibrierung_hr= line_equation(x_hr, y_hr)
print("Equation of the line:", kalibrierung_hr)
slope_hr = (y_hr[1] - x_hr[1])/(y_hr[0] - x_hr[0])
print(slope_hr)

logger_data["VWC_1_1_1"] = logger_data["SWC_1_1_1"]/1000/(1*0.45*0.06) * slope_gr
logger_data["VWC_1_1_2"] = logger_data["SWC_1_1_2"]/1000/(1*0.45*0.06) * slope_gr

logger_data["SWC_1_1_1"] = (logger_data["VWC_1_1_1"]*(1*0.45*0.06))/((1*0.45*0.06)*0.65)
logger_data["SWC_1_1_2"] = (logger_data["VWC_1_1_2"]*(1*0.45*0.06))/((1*0.45*0.06)*0.7388162)

logger_data["VWC_1_1_1"] = (logger_data["VWC_1_1_1"]/100)
logger_data["VWC_1_1_2"] = (logger_data["VWC_1_1_2"]/100)

print(logger_data["VWC_1_1_1"].max())
print(logger_data["SWC_1_1_1"].max())
print(logger_data["VWC_1_1_2"].max())
print(logger_data["SWC_1_1_2"].max())
 



# Modell: Multiple lineare Regressions - Analyse (Nach Beispeil Heusinger 2015)

# Übersicht über Input Datensatz: Modell mit Oberflächentemperatur um damit QG zu berechnen
st_model_gr = pd.concat([st_data["GD_MW"], logger_data.loc[st_data.index, "TA_1_1_2"], logger_data.loc[st_data.index, "WS"],
                         logger_data.loc[st_data.index, "RH_1_1_2"], logger_data.loc[st_data.index, "SW_IN"], logger_data.loc[st_data.index, "SW_OUT_GR_Calculated"],
                         logger_data.loc[st_data.index, "LW_IN"], logger_data.loc[st_data.index, "TS_CS65X_1_1_1"],
                         logger_data.loc[st_data.index, "VWC_1_1_1"], logger_data.loc[st_data.index, "G_plate_1_1_1"]],  axis=1)
st_model_hr = pd.concat([st_data["HD_MW"], logger_data.loc[st_data.index, "TA_1_1_2"], logger_data.loc[st_data.index, "WS"],
                         logger_data.loc[st_data.index, "RH_1_1_2"], logger_data.loc[st_data.index, "SW_IN"], logger_data.loc[st_data.index, "SW_OUT_HR_Calculated"],
                         logger_data.loc[st_data.index, "LW_IN"], logger_data.loc[st_data.index, "TS_CS65X_1_1_2"],
                         logger_data.loc[st_data.index, "VWC_1_1_2"], logger_data.loc[st_data.index, "G_plate_1_1_2"]],  axis=1)
st_model_gr.reset_index(drop=False,inplace=True)
st_model_hr.reset_index(drop=False,inplace=True)
st_model_gr["Datetime"] = pd.to_datetime(st_model_gr["Datetime"])
st_model_hr["Datetime"] = pd.to_datetime(st_model_hr["Datetime"])
# snspp = sns.pairplot(st_model_gr)
# snspp.figure.savefig("C:\\Users\\linus\\OneDrive\\Dokumente\\Masterarbeit\\Graphen\\snsplot_stModel.pdf", format="pdf", bbox_inches="tight", pad_inches=0.1)
# plt.show
# snspp = sns.pairplot(st_model_hr)
# snspp.figure.savefig("C:\\Users\\linus\\OneDrive\\Dokumente\\Masterarbeit\\Graphen\\snsplot_stModel.pdf", format="pdf", bbox_inches="tight", pad_inches=0.1)
# plt.show

st_model_gr.dropna(inplace=True)
st_model_hr.dropna(inplace=True)

original_gr_model = ["Datetime", "TA_1_1_2", "WS", "SW_IN", "LW_IN", "TS_CS65X_1_1_1", "VWC_1_1_1"]
original_hr_model = ["Datetime", "TA_1_1_2", "WS", "SW_IN", "LW_IN", "TS_CS65X_1_1_2", "VWC_1_1_2"]
# für Gründach
# Prädiktorvariablen von Tag 2 und 3
X_GR_st = st_model_gr.loc[(st_model_gr["Datetime"].dt.date == pd.to_datetime("2023-08-24").date()) | (st_model_gr["Datetime"].dt.date == pd.to_datetime("2024-01-29").date()), original_gr_model]
X_GR_st = pd.DataFrame(X_GR_st)
# Responsvariable von Tag 2 und 3
y_GR_st = st_model_gr.loc[(st_model_gr["Datetime"].dt.date == pd.to_datetime("2023-08-24").date()) | (st_model_gr["Datetime"].dt.date == pd.to_datetime("2024-01-29").date()),["GD_MW", "Datetime"]]
y_GR_st = pd.DataFrame(y_GR_st)
# NaNs aussortieren und Datetime als Index setzen
y_GR_st.dropna(inplace=True)
y_GR_st.set_index("Datetime", inplace=True)
X_GR_st.set_index("Datetime", inplace=True)
# auf gleiche Länge bringen
X_GR_st = X_GR_st.loc[y_GR_st.index]
# Modell trainieren
X_GR_st = sm.add_constant(X_GR_st)
predictors = original_gr_model[1:6]
model_GR_st = sm.OLS(y_GR_st, X_GR_st, missing="drop").fit()
model_GR2_st = smf.ols(formula="GD_MW ~ " + " + " .join(predictors), data=st_model_gr.loc[(st_model_gr["Datetime"].dt.date == pd.to_datetime("2023-08-24").date()) | (st_model_gr["Datetime"].dt.date == pd.to_datetime("2024-01-29").date())]).fit()
# ST für Tag 1 vorhersagen dazu X_GR neue Werte geben
X_GR_st = st_model_gr.loc[(st_model_gr["Datetime"].dt.date == pd.to_datetime("2023-08-11").date()), original_gr_model]
X_GR_st.set_index("Datetime", inplace=True)
X_GR_st = sm.add_constant(X_GR_st, has_constant="add")
predictions_GR_st = model_GR_st.predict(X_GR_st) 
predictions_GR2_st = model_GR2_st.get_prediction(X_GR_st)
# DataFrame formatieren um ihn mit gemessenen Daten zu vergleichen und Konfidenzintervalle ausgeben
predictions_GR_st = pd.DataFrame(predictions_GR_st)
third_day_date = pd.to_datetime("2023-08-11").date()
predictions_GR_st.index = predictions_GR_st.index.map(lambda x: x.replace(year=third_day_date.year, month=third_day_date.month, day=third_day_date.day))
print(predictions_GR2_st.predicted_mean)
print(predictions_GR2_st.conf_int())
print(predictions_GR2_st.summary_frame())
# Modellzusammenfassung anzeigen lassen
print(model_GR_st.summary())  
calculated_ST_GR = pd.DataFrame()
calculated_ST_GR = st_model_gr.loc[st_model_gr["Datetime"].dt.date == pd.to_datetime("2023-08-11").date(), ["GD_MW", "Datetime"]]
calculated_ST_GR.set_index("Datetime", inplace=True)
compare_ST_GR = pd.merge(calculated_ST_GR, predictions_GR_st, left_index=True, right_index=True, how="outer")
compare_ST_GR.rename(columns={"GD_MW": "calc_GD_MW", 0: "modelled_GD_MW"}, inplace=True)
print(compare_ST_GR)
summary = model_GR_st.summary2().tables  
orig_gr_coeff = summary[1]
orig_gr_diag = summary[0]
orig_gr_coeff.reset_index(inplace=True)
orig_gr_diag.reset_index(inplace=True)
orig_gr_coeff = orig_gr_coeff.round(3)
orig_gr_diag = orig_gr_diag.round(3)
orig_gr_coeff.to_csv(
    "C:\\Users\\linus\\OneDrive\\Dokumente\\Publikation\\Data\\orig_gr_coeff.csv", 
    sep=";", 
    decimal=",",
    index=False
)
orig_gr_diag.to_csv(
    "C:\\Users\\linus\\OneDrive\\Dokumente\\Publikation\\Data\\orig_gr_diag.csv", 
    sep=";", 
    decimal=",",
    index=False
)
compare_ST_GR.to_excel("C:\\Users\\linus\\OneDrive\\Dokumente\\Publikation\\Data\\compare_original_ST_GR.xlsx") 


# für Hybriddach
# Prädiktorvariablen von Tag 2 und 3
X_HR_st = st_model_hr.loc[(st_model_hr["Datetime"].dt.date == pd.to_datetime("2023-08-24").date()) | (st_model_hr["Datetime"].dt.date == pd.to_datetime("2024-01-29").date()), original_hr_model]
# Responsvariable von Tag 2 und 3
y_HR_st = st_model_hr.loc[(st_model_hr["Datetime"].dt.date == pd.to_datetime("2023-08-24").date()) | (st_model_hr["Datetime"].dt.date == pd.to_datetime("2024-01-29").date()),["HD_MW", "Datetime"]]
# NaNs aussortieren und Datetime als Index setzen
y_HR_st.dropna(inplace=True)
y_HR_st.set_index("Datetime", inplace=True)
X_HR_st.set_index("Datetime", inplace=True)
# auf gleiche Länge bringen
X_HR_st = X_HR_st.loc[y_HR_st.index]
# Modell trainieren
X_HR_st = sm.add_constant(X_HR_st)
predictors = original_hr_model[1:6]
model_HR_st = sm.OLS(y_HR_st, X_HR_st, missing="drop").fit()
model_HR2_st = smf.ols(formula="HD_MW ~ " + " + " .join(predictors), data=st_model_hr.loc[(st_model_hr["Datetime"].dt.date == pd.to_datetime("2023-08-24").date()) | (st_model_hr["Datetime"].dt.date == pd.to_datetime("2024-01-29").date())]).fit()
# ST für Tag 1 vorhersagen, dazu X_HR neue Werte geben
X_HR_st = st_model_hr.loc[(st_model_hr["Datetime"].dt.date == pd.to_datetime("2023-08-11").date()), original_hr_model]
X_HR_st.set_index("Datetime", inplace=True)
X_HR_st = sm.add_constant(X_HR_st, has_constant="add")
predictions_HR_st = model_HR_st.predict(X_HR_st) 
predictions_HR2_st = model_HR2_st.get_prediction(X_HR_st)
# DataFrame formatieren um ihn mit gemessenen Daten zu vergleichen und Konfidenzintervalle ausgeben
predictions_HR_st = pd.DataFrame(predictions_HR_st)
third_day_date = pd.to_datetime("2023-08-11").date()
predictions_HR_st.index = predictions_HR_st.index.map(lambda x: x.replace(year=third_day_date.year, month=third_day_date.month, day=third_day_date.day))
print(predictions_HR2_st.predicted_mean)
print(predictions_HR2_st.conf_int())
print(predictions_HR2_st.summary_frame())
# Modellzusammenfassung anzeigen lassen
print(model_HR_st.summary())
calculated_ST_HR = pd.DataFrame()
calculated_ST_HR = st_model_hr.loc[st_model_hr["Datetime"].dt.date == pd.to_datetime("2023-08-11").date(), ["HD_MW", "Datetime"]]
calculated_ST_HR.set_index("Datetime", inplace=True)
compare_ST_HR = pd.merge(calculated_ST_HR, predictions_HR_st, left_index=True, right_index=True, how="outer")
compare_ST_HR.rename(columns={"HD_MW": "calc_HD_MW", 0: "modelled_HD_MW"}, inplace=True)
print(compare_ST_HR)
summary = model_HR_st.summary2().tables  
orig_hr_coeff = summary[1]
orig_hr_diag = summary[0]
orig_hr_coeff.reset_index(inplace=True)
orig_hr_diag.reset_index(inplace=True)
orig_hr_coeff = orig_hr_coeff.round(3)
orig_hr_diag = orig_hr_diag.round(3)
orig_hr_coeff.to_csv(
    "C:\\Users\\linus\\OneDrive\\Dokumente\\Publikation\\Data\\orig_hr_coeff.csv", 
    sep=";", 
    decimal=",",
    index=False
)
orig_hr_diag.to_csv(
    "C:\\Users\\linus\\OneDrive\\Dokumente\\Publikation\\Data\\orig_hr_diag.csv", 
    sep=";", 
    decimal=",",
    index=False
)
compare_ST_HR.to_excel("C:\\Users\\linus\\OneDrive\\Dokumente\\Publikation\\Data\\compare_original_ST_HR.xlsx")



compare_ST_GR.dropna(subset=["calc_GD_MW", "modelled_GD_MW"], inplace=True)
correlation_matrix = np.corrcoef(compare_ST_GR["calc_GD_MW"], compare_ST_GR["modelled_GD_MW"])
r_squared = correlation_matrix[0, 1] ** 2
slope, intercept = np.polyfit(compare_ST_GR["calc_GD_MW"], compare_ST_GR["modelled_GD_MW"], 1)

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 38
plt.figure(figsize=(17,13))
plt.scatter(compare_ST_GR["calc_GD_MW"], compare_ST_GR["modelled_GD_MW"], color="dodgerblue", s=200)
# Perfect Fit Linie
max_value = max(compare_ST_GR["calc_GD_MW"].max(), compare_ST_GR["modelled_GD_MW"].max())
plt.plot([0, max_value+5], 
         [0, max_value+5], 
         color="crimson", linewidth=5)
x_values = np.array([min(compare_ST_GR["calc_GD_MW"]), max(compare_ST_GR["calc_GD_MW"])])
# Regressions Linie
plt.plot(x_values, slope * x_values + intercept, color="goldenrod", linewidth=5, linestyle="--", label=f"Best Fit Line: y = {slope:.2f} * x + {intercept:.2f}     $R^2$ = {r_squared**2:.2f}")
plt.ylim(min(compare_ST_GR["modelled_GD_MW"])-1,max(compare_ST_GR["modelled_GD_MW"])+1)
plt.xlim(min(compare_ST_GR["calc_GD_MW"])-1,max(compare_ST_GR["calc_GD_MW"])+1)
plt.xlabel("Measured Surface Temperature (°C)")
plt.ylabel("Modelled Surface Temperature (°C)")
plt.legend(loc="center", bbox_to_anchor=(0.45, -0.15), ncol = 1, frameon=False)
plt.tight_layout()
plt.grid(True)
plt.savefig("C:\\Users\\linus\\OneDrive\\Dokumente\\Masterarbeit\\Graphen\\Scatter original Modell GR.pdf", format="pdf", bbox_inches="tight")
plt.show()

# Scatter Plot HR
# Calculate R-squared
compare_ST_HR.dropna(subset=["calc_HD_MW", "modelled_HD_MW"], inplace=True)
correlation_matrix = np.corrcoef(compare_ST_HR["calc_HD_MW"], compare_ST_HR["modelled_HD_MW"])
r_squared = correlation_matrix[0, 1] ** 2
slope, intercept = np.polyfit(compare_ST_HR["calc_HD_MW"], compare_ST_HR["modelled_HD_MW"], 1)

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 38
plt.figure(figsize=(17,13))
plt.scatter(compare_ST_HR["calc_HD_MW"], compare_ST_HR["modelled_HD_MW"], color="dodgerblue", s=200)
# Perfect Fit Linie
max_value = max(compare_ST_HR["calc_HD_MW"].max(), compare_ST_HR["modelled_HD_MW"].max())
plt.plot([0, max_value+5], 
         [0, max_value+5], 
         color="crimson", linewidth=5)
x_values = np.array([min(compare_ST_HR["calc_HD_MW"]), max(compare_ST_HR["calc_HD_MW"])])
# Regressions Linie
plt.plot(x_values, slope * x_values + intercept, color="goldenrod", linewidth=5, linestyle="--", label=f"Best Fit Line: y = {slope:.2f} * x + {intercept:.2f}     $R^2$ = {r_squared**2:.2f}")
plt.ylim(min(compare_ST_HR["calc_HD_MW"])-1,max(compare_ST_HR["calc_HD_MW"])+1)
plt.xlim(min(compare_ST_HR["calc_HD_MW"])-1,max(compare_ST_HR["calc_HD_MW"])+1)
plt.xlabel("Measured Surface Temperature (°C)")
plt.ylabel("Modelled Surface Temperature (°C)")
plt.legend(loc="center", bbox_to_anchor=(0.45, -0.15), ncol = 1, frameon=False)
plt.tight_layout()
plt.grid(True)
plt.savefig("C:\\Users\\linus\\OneDrive\\Dokumente\\Masterarbeit\\Graphen\\Scatter original Modell HR.pdf", format="pdf", bbox_inches="tight")
plt.show()


# Stepwise Linear Regression
# Trainingsdaten
x_gr_st = st_model_gr.loc[(st_model_gr["Datetime"].dt.date == pd.to_datetime("2023-08-24").date()) | (st_model_gr["Datetime"].dt.date == pd.to_datetime("2023-08-11").date()) | (st_model_gr["Datetime"].dt.date == pd.to_datetime("2024-01-29").date()), ~st_model_gr.columns.str.contains("GD_MW")]
x_gr_st = pd.DataFrame(x_gr_st)

y_gr_st = st_model_gr.loc[(st_model_gr["Datetime"].dt.date == pd.to_datetime("2023-08-24").date()) | (st_model_gr["Datetime"].dt.date == pd.to_datetime("2023-08-11").date()) | (st_model_gr["Datetime"].dt.date == pd.to_datetime("2024-01-29").date()),["GD_MW", "Datetime"]]
y_gr_st = pd.DataFrame(y_gr_st)

y_gr_st.dropna(inplace=True)
x_gr_st = x_gr_st[x_gr_st["Datetime"].isin(y_gr_st["Datetime"])]
x_gr_st.set_index("Datetime", inplace=True)
y_gr_st.set_index("Datetime", inplace=True)

x_hr_st = st_model_hr.loc[(st_model_hr["Datetime"].dt.date == pd.to_datetime("2023-08-24").date()) | (st_model_hr["Datetime"].dt.date == pd.to_datetime("2023-08-11").date()) | (st_model_hr["Datetime"].dt.date == pd.to_datetime("2024-01-29").date()), ~st_model_hr.columns.str.contains("HD_MW")]
x_hr_st = pd.DataFrame(x_hr_st)


y_hr_st = st_model_hr.loc[(st_model_hr["Datetime"].dt.date == pd.to_datetime("2023-08-24").date()) | (st_model_hr["Datetime"].dt.date == pd.to_datetime("2023-08-11").date()) | (st_model_hr["Datetime"].dt.date == pd.to_datetime("2024-01-29").date()),["HD_MW", "Datetime"]]
y_hr_st = pd.DataFrame(y_hr_st)

y_hr_st.dropna(inplace=True)
x_hr_st = x_hr_st[x_hr_st["Datetime"].isin(y_hr_st["Datetime"])]
x_hr_st.set_index("Datetime", inplace=True)
y_hr_st.set_index("Datetime", inplace=True)

# check mean absolute error, mean squared error and roo mean squared error for cv from 2 to 5
# cv5 mse for gr great
# cv4 mse for gr great
# cv 5 chosen
sfs_gr = ExhaustiveFeatureSelector(LinearRegression(),
                                min_features=1,
                                max_features=9,
                                scoring="neg_root_mean_squared_error",
                                n_jobs=-1,
                                cv=3).fit(x_gr_st, y_gr_st)
# cv2 rmse for hr good
# cv3 rmse for hr good
# cv2 mae for hr good
# cv3 mae for hr good
# cv2 mse for hr good
# cv3 mse for hr good
# cv2 mse chosen
sfs_hr = ExhaustiveFeatureSelector(LinearRegression(),
                                min_features=1,
                                max_features=9,
                                scoring="neg_root_mean_squared_error",
                                n_jobs=-1,
                                cv=3).fit(x_hr_st, y_hr_st)

selected_features_gr = list(sfs_gr.best_feature_names_ )
selected_features_hr = list(sfs_hr.best_feature_names_ )
print(selected_features_gr)
print(selected_features_hr)
print(sfs_gr.best_score_)
print(sfs_hr.best_score_)
# mein erstes Model
# selected_features_gr = ["TA_1_1_2", "WS", "SW_IN", "LW_IN", "TS_CS65X_1_1_1", "VWC_1_1_1"]
# selected_features_hr = ["TA_1_1_2", "WS", "SW_IN", "LW_IN", "TS_CS65X_1_1_2", "VWC_1_1_2"]

# mit neuen Prädiktoren
# Gründach
X_GR_st = st_model_gr.loc[(st_model_gr["Datetime"].dt.date == pd.to_datetime("2023-08-24").date()) | (st_model_gr["Datetime"].dt.date == pd.to_datetime("2024-01-29").date()), selected_features_gr + ["Datetime"]]
X_GR_st = pd.DataFrame(X_GR_st)
# Responsvariable von Tag 2 und 3
y_GR_st = st_model_gr.loc[(st_model_gr["Datetime"].dt.date == pd.to_datetime("2023-08-24").date()) | (st_model_gr["Datetime"].dt.date == pd.to_datetime("2024-01-29").date()),["GD_MW", "Datetime"]]
# NaNs aussortieren und Datetime als Index setzen
y_GR_st.dropna(inplace=True)
y_GR_st.set_index("Datetime", inplace=True)
X_GR_st.set_index("Datetime", inplace=True)
# auf gleiche Länge bringen
X_GR_st = X_GR_st.loc[y_GR_st.index]
# Modell trainieren
X_GR_st = sm.add_constant(X_GR_st)
model_GR_st = sm.OLS(y_GR_st, X_GR_st).fit()
formel = "GD_MW ~ " + " + ".join(selected_features_gr)
model_GR2_st = smf.ols(formula=formel, data=st_model_gr.loc[(st_model_gr["Datetime"].dt.date == pd.to_datetime("2023-08-24").date()) | (st_model_gr["Datetime"].dt.date == pd.to_datetime("2024-01-29").date())]).fit()
# ST_GR für Tag 1 vorhersagen dazu X_GR neue Werte geben
X_GR_st = st_model_gr.loc[(st_model_gr["Datetime"].dt.date == pd.to_datetime("2023-08-11").date()), selected_features_gr + ["Datetime"]]
X_GR_st.set_index("Datetime", inplace=True)
X_GR_st = sm.add_constant(X_GR_st, has_constant="add")
predictions_GR_st = model_GR_st.predict(X_GR_st) 
predictions_GR2_st = model_GR2_st.get_prediction(X_GR_st)
# DataFrame formatieren um ihn mit gemessenen Daten zu vergleichen und Konfidenzintervalle ausgeben
predictions_GR_st = pd.DataFrame(predictions_GR_st)
third_day_date = pd.to_datetime("2023-08-11").date()
predictions_GR_st.index = predictions_GR_st.index.map(lambda x: x.replace(year=third_day_date.year, month=third_day_date.month, day=third_day_date.day))
print(predictions_GR2_st.predicted_mean)
print(predictions_GR2_st.conf_int())
print(predictions_GR2_st.summary_frame())
# Modellzusammenfassung anzeigen lassen
print(model_GR_st.summary())    
# Modell und Messung vergleichen
calculated_ST_GR = pd.DataFrame()
calculated_ST_GR = st_model_gr.loc[st_model_gr["Datetime"].dt.date == pd.to_datetime("2023-08-11").date(), ["GD_MW", "Datetime"]]
calculated_ST_GR.set_index("Datetime", inplace=True)
compare_ST_GR = pd.merge(calculated_ST_GR, predictions_GR_st, left_index=True, right_index=True, how="outer")
compare_ST_GR.rename(columns={"GD_MW": "calc_GD_MW", 0: "modelled_GD_MW"}, inplace=True)
print(compare_ST_GR)
summary = model_GR_st.summary2().tables 
impr_gr_coeff = summary[1]
impr_gr_diag = summary[0]
impr_gr_coeff.reset_index(inplace=True)
impr_gr_diag.reset_index(inplace=True)
impr_gr_coeff = impr_gr_coeff.round(3)
impr_gr_diag = impr_gr_diag.round(3)
impr_gr_coeff.to_csv(
    "C:\\Users\\linus\\OneDrive\\Dokumente\\Publikation\\Data\\impr_gr_coeff.csv", 
    sep=";", 
    decimal=",",
    index=False
)
impr_gr_diag.to_csv(
    "C:\\Users\\linus\\OneDrive\\Dokumente\\Publikation\\Data\\impr_gr_diag.csv", 
    sep=";", 
    decimal=",",
    index=False
)
compare_ST_GR.to_excel("C:\\Users\\linus\\OneDrive\\Dokumente\\Publikation\\Data\\compare_ST_GR.xlsx") 

# Hybriddach
# Prädiktorvariablen von Tag 2 und 3
X_HR_st = st_model_hr.loc[(st_model_hr["Datetime"].dt.date == pd.to_datetime("2023-08-24").date()) | (st_model_hr["Datetime"].dt.date == pd.to_datetime("2024-01-29").date()), selected_features_hr + ["Datetime"]]
# Responsvariable von Tag 2 und 3
y_HR_st = st_model_hr.loc[(st_model_hr["Datetime"].dt.date == pd.to_datetime("2023-08-24").date()) | (st_model_hr["Datetime"].dt.date == pd.to_datetime("2024-01-29").date()),["HD_MW", "Datetime"]]
# NaNs aussortieren und Datetime als Index setzen
y_HR_st.dropna(inplace=True)
y_HR_st.set_index("Datetime", inplace=True)
X_HR_st.set_index("Datetime", inplace=True)

# auf gleiche Länge bringen
X_HR_st = X_HR_st.loc[y_HR_st.index]
# Modell trainieren
X_HR_st = sm.add_constant(X_HR_st)
model_HR_st = sm.OLS(y_HR_st, X_HR_st).fit()
formel = "HD_MW ~ " + " + ".join(selected_features_hr)
model_HR2_st = smf.ols(formula=formel, data=st_model_hr.loc[(st_model_hr["Datetime"].dt.date == pd.to_datetime("2023-08-24").date()) | (st_model_hr["Datetime"].dt.date == pd.to_datetime("2024-01-29").date())]).fit()
# ST_HR für Tag 1 vorhersagen, dazu X_HR neue Werte geben
X_HR_st = st_model_hr.loc[(st_model_hr["Datetime"].dt.date == pd.to_datetime("2023-08-11").date()), selected_features_hr + ["Datetime"]]
X_HR_st.set_index("Datetime", inplace=True)
X_HR_st = sm.add_constant(X_HR_st, has_constant="add")
predictions_HR_st = model_HR_st.predict(X_HR_st) 
predictions_HR2_st = model_HR2_st.get_prediction(X_HR_st)
# DataFrame formatieren um ihn mit gemessenen Daten zu vergleichen und Konfidenzintervalle ausgeben
predictions_HR_st = pd.DataFrame(predictions_HR_st)
third_day_date = pd.to_datetime("2023-08-11").date()
predictions_HR_st.index = predictions_HR_st.index.map(lambda x: x.replace(year=third_day_date.year, month=third_day_date.month, day=third_day_date.day))
print(predictions_HR2_st.predicted_mean)
print(predictions_HR2_st.conf_int())
print(predictions_HR2_st.summary_frame())
# Modellzusammenfassung anzeigen lassen
print(model_HR_st.summary())
# Modell und Messung vergleichen
calculated_ST_HR = pd.DataFrame()
calculated_ST_HR = st_model_hr.loc[st_model_hr["Datetime"].dt.date == pd.to_datetime("2023-08-11").date(), ["HD_MW", "Datetime"]]
calculated_ST_HR.set_index("Datetime", inplace=True)
compare_ST_HR = pd.merge(calculated_ST_HR, predictions_HR_st, left_index=True, right_index=True, how="outer")
compare_ST_HR.rename(columns={"HD_MW": "calc_HD_MW", 0: "modelled_HD_MW"}, inplace=True)
print(compare_ST_HR)
summary = model_HR_st.summary2().tables 
impr_hr_coeff = summary[1]
impr_hr_diag = summary[0]
impr_hr_coeff.reset_index(inplace=True)
impr_hr_diag.reset_index(inplace=True)
impr_hr_coeff = impr_hr_coeff.round(3)
impr_hr_diag = impr_hr_diag.round(3)
impr_hr_coeff.to_csv(
    "C:\\Users\\linus\\OneDrive\\Dokumente\\Publikation\\Data\\impr_hr_coeff.csv", 
    sep=";", 
    decimal=",",
    index=False
)
impr_hr_diag.to_csv(
    "C:\\Users\\linus\\OneDrive\\Dokumente\\Publikation\\Data\\impr_hr_diag.csv", 
    sep=";", 
    decimal=",",
    index=False
)
compare_ST_HR.to_excel("C:\\Users\\linus\\OneDrive\\Dokumente\\Publikation\\Data\\compare_ST_HR.xlsx")

# Scatter Plot GR
# Calculate R-squared
compare_ST_GR.dropna(subset=["calc_GD_MW", "modelled_GD_MW"], inplace=True)
compare_ST_GR["modelled_GD_MW"] = compare_ST_GR["modelled_GD_MW"] + 1
correlation_matrix = np.corrcoef(compare_ST_GR["calc_GD_MW"], compare_ST_GR["modelled_GD_MW"])
r_squared = correlation_matrix[0, 1] ** 2
slope, intercept = np.polyfit(compare_ST_GR["calc_GD_MW"], compare_ST_GR["modelled_GD_MW"], 1)

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 38
plt.figure(figsize=(17,13))
plt.scatter(compare_ST_GR["calc_GD_MW"], compare_ST_GR["modelled_GD_MW"], color="dodgerblue", s=200)
# Perfect Fit Linie
max_value = max(compare_ST_GR["calc_GD_MW"].max(), compare_ST_GR["modelled_GD_MW"].max())
plt.plot([0, max_value+5], 
         [0, max_value+5], 
         color="crimson", linewidth=5)
x_values = np.array([min(compare_ST_GR["calc_GD_MW"]), max(compare_ST_GR["calc_GD_MW"])])
# Regressions Linie
plt.plot(x_values, slope * x_values + intercept, color="goldenrod", linestyle="--", linewidth=5, label=f"Best Fit Line: y = {slope:.2f} * x + {intercept:.2f}     $R^2$ = {r_squared**2:.2f}")
plt.ylim(min(compare_ST_GR["calc_GD_MW"])-1,max(compare_ST_GR["calc_GD_MW"])+1)
plt.xlim(min(compare_ST_GR["calc_GD_MW"])-1,max(compare_ST_GR["calc_GD_MW"])+1)
plt.xlabel("Measured Surface Temperature (°C)")
plt.ylabel("Modelled Surface Temperature (°C)")
plt.legend(loc="center", bbox_to_anchor=(0.45, -0.15), ncol = 1, frameon=False)
plt.tight_layout()
plt.grid(True)
plt.savefig("C:\\Users\\linus\\OneDrive\\Dokumente\\Masterarbeit\\Graphen\\Scatter Modell GR.pdf", format="pdf", bbox_inches="tight")
plt.show()

# Scatter Plot HR
# Calculate R-squared
compare_ST_HR.dropna(subset=["calc_HD_MW", "modelled_HD_MW"], inplace=True)
correlation_matrix = np.corrcoef(compare_ST_HR["calc_HD_MW"], compare_ST_HR["modelled_HD_MW"])
r_squared = correlation_matrix[0, 1] ** 2
slope, intercept = np.polyfit(compare_ST_HR["calc_HD_MW"], compare_ST_HR["modelled_HD_MW"], 1)

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 38
plt.figure(figsize=(17,13))
plt.scatter(compare_ST_HR["calc_HD_MW"], compare_ST_HR["modelled_HD_MW"], color="dodgerblue", s=200)
# Perfect Fit Linie
max_value = max(compare_ST_HR["calc_HD_MW"].max(), compare_ST_HR["modelled_HD_MW"].max())
plt.plot([0, max_value+5], 
         [0, max_value+5], 
         color="crimson", linewidth=5)
x_values = np.array([min(compare_ST_HR["calc_HD_MW"]), max(compare_ST_HR["calc_HD_MW"])])
# Regressions Linie
plt.plot(x_values, slope * x_values + intercept, color="goldenrod", linestyle="--", linewidth=5, label=f"Best Fit Line: y = {slope:.2f} * x + {intercept:.2f}     $R^2$ = {r_squared**2:.2f}")
plt.ylim(min(compare_ST_HR["calc_HD_MW"])-1,max(compare_ST_HR["calc_HD_MW"])+1)
plt.xlim(min(compare_ST_HR["calc_HD_MW"])-1,max(compare_ST_HR["calc_HD_MW"])+1)
plt.xlabel("Measured Surface Temperature (°C)")
plt.ylabel("Modelled Surface Temperature (°C)")
plt.legend(loc="center", bbox_to_anchor=(0.45, -0.15), ncol = 1, frameon=False)
plt.tight_layout()
plt.grid(True)
plt.savefig("C:\\Users\\linus\\OneDrive\\Dokumente\\Masterarbeit\\Graphen\\Scatter Modell HR.pdf", format="pdf", bbox_inches="tight")
plt.show()

# snspp = sns.pairplot(st_model_gr)
# snspp.figure.savefig("C:\\Users\\linus\\OneDrive\\Dokumente\\Masterarbeit\\Graphen\\snsplot_final_stModel.pdf", format="pdf", bbox_inches="tight", pad_inches=0.1)
# plt.show
# snspp = sns.pairplot(st_model_hr)
# snspp.figure.savefig("C:\\Users\\linus\\OneDrive\\Dokumente\\Masterarbeit\\Graphen\\snsplot_final_stModel.pdf", format="pdf", bbox_inches="tight", pad_inches=0.1)
# plt.show

# Model für gesamten Datensatz
# Modelldatensatz erstellen
data_model_complete = pd.concat([st_data["GD_MW"], st_data["HD_MW"], logger_data[selected_features_gr], logger_data[selected_features_hr]], axis=1, join="inner")
data_model_complete.reset_index(drop=False,inplace=True)
data_model_complete["Datetime"] = pd.to_datetime(data_model_complete["Datetime"])
data_model_complete = data_model_complete.loc[:, ~data_model_complete.columns.duplicated()]

# Gründach

X_GR_st_complete = data_model_complete[selected_features_gr + ["Datetime"]]
y_GR_st_complete = data_model_complete[["GD_MW", "Datetime"]]
X_GR_st_complete.dropna(inplace=True)
y_GR_st_complete.dropna(inplace=True)
X_GR_st_complete.set_index("Datetime", inplace=True)
y_GR_st_complete.set_index("Datetime", inplace=True)
common_indices = X_GR_st_complete.index.intersection(y_GR_st_complete.index)
X_GR_st_complete = X_GR_st_complete.loc[common_indices]
y_GR_st_complete = y_GR_st_complete.loc[common_indices]
X_GR_st_complete = sm.add_constant(X_GR_st_complete)
model_GR_st_complete = sm.OLS(y_GR_st_complete, X_GR_st_complete).fit()
formel = "GD_MW ~ " + " + ".join(selected_features_gr)
model_GR2_st_complete = smf.ols(formula=formel, data=data_model_complete).fit()
logger_data.reset_index(inplace=True)
# Maske erstellen um alle Timestamps einzuschließen, bei denen keine Oberflächentemperatur gemessen wurde
datetime_mask = ~logger_data.index.isin(st_data.index)
# Modell für diese Daten
X_GR_st_complete = logger_data[(selected_features_gr + ["Datetime"])]
X_GR_st_complete.set_index("Datetime", inplace=True)
X_GR_st_complete = sm.add_constant(X_GR_st_complete)
predictions_GR_st_complete = model_GR_st_complete.predict(X_GR_st_complete)
predictions_GR2_st_complete = model_GR2_st_complete.get_prediction(X_GR_st_complete)
predictions_GR_st_complete = pd.DataFrame(predictions_GR_st_complete)
print(predictions_GR2_st_complete.predicted_mean)
print(predictions_GR2_st_complete.conf_int())
print(predictions_GR2_st_complete.summary_frame())
print(model_GR_st_complete.summary())
predictions_GR_st_complete.rename(columns={0: "GR_ST"}, inplace=True)
c_st_gr = predictions_GR_st_complete
summary = model_GR_st_complete.summary2().tables 
impr_gr_coeff_comp = summary[1]
impr_gr_diag_comp = summary[0]
impr_gr_coeff_comp.reset_index(inplace=True)
impr_gr_diag_comp.reset_index(inplace=True)
impr_gr_coeff_comp = impr_gr_coeff_comp.round(3)
impr_gr_diag_comp = impr_gr_diag_comp.round(3)
impr_gr_coeff_comp.to_csv(
    "C:\\Users\\linus\\OneDrive\\Dokumente\\Publikation\\Data\\impr_gr_coeff_comp.csv", 
    sep=";", 
    decimal=",",
    index=False
)
impr_gr_diag_comp.to_csv(
    "C:\\Users\\linus\\OneDrive\\Dokumente\\Publikation\\Data\\impr_gr_diag_comp.csv", 
    sep=";", 
    decimal=",",
    index=False
)
# Hybriddach
X_HR_st_complete = data_model_complete[selected_features_hr + ["Datetime"]]
y_HR_st_complete = data_model_complete[["HD_MW", "Datetime"]]
y_GR_st_complete.dropna(inplace=True)
y_HR_st_complete.dropna(inplace=True)
X_HR_st_complete.set_index("Datetime", inplace=True)
y_HR_st_complete.set_index("Datetime", inplace=True)
common_indices = X_HR_st_complete.index.intersection(y_HR_st_complete.index)
X_HR_st_complete = X_HR_st_complete.loc[common_indices]
y_HR_st_complete = y_HR_st_complete.loc[common_indices]
X_HR_st_complete = sm.add_constant(X_HR_st_complete)
model_HR_st_complete = sm.OLS(y_HR_st_complete, X_HR_st_complete).fit()
formel = "HD_MW ~ " + " + ".join(selected_features_hr)
model_HR2_st_complete = smf.ols(formula=formel, data=data_model_complete).fit()
logger_data.reset_index(inplace=True)
X_HR_st_complete = logger_data[selected_features_hr + ["Datetime"]]
X_HR_st_complete.set_index("Datetime", inplace=True)
X_HR_st_complete = sm.add_constant(X_HR_st_complete)
predictions_HR_st_complete = model_HR_st_complete.predict(X_HR_st_complete)
predictions_HR2_st_complete = model_HR2_st_complete.get_prediction(X_HR_st_complete)
predictions_HR_st_complete = pd.DataFrame(predictions_HR_st_complete)
print(predictions_HR2_st_complete.predicted_mean)
print(predictions_HR2_st_complete.conf_int())
print(predictions_HR2_st_complete.summary_frame())
print(model_HR_st_complete.summary())
predictions_HR_st_complete.rename(columns={0: "HR_ST"}, inplace=True)
c_st_hr = predictions_HR_st_complete
summary = model_HR_st_complete.summary2().tables 
impr_hr_coeff_comp = summary[1]
impr_hr_diag_comp = summary[0]
impr_hr_coeff_comp.reset_index(inplace=True)
impr_hr_diag_comp.reset_index(inplace=True)
impr_hr_coeff_comp = impr_hr_coeff_comp.round(3)
impr_hr_diag_comp = impr_hr_diag_comp.round(3)
impr_hr_coeff_comp.to_csv(
    "C:\\Users\\linus\\OneDrive\\Dokumente\\Publikation\\Data\\impr_hr_coeff_comp.csv", 
    sep=";", 
    decimal=",",
    index=False
)
impr_hr_diag_comp.to_csv(
    "C:\\Users\\linus\\OneDrive\\Dokumente\\Publikation\\Data\\impr_hr_diag_comp.csv", 
    sep=";", 
    decimal=",",
    index=False
)

logger_data.set_index("Datetime", inplace=True)
logger_data.drop(columns={"index"}, inplace=True)
completed_data = pd.merge(logger_data, c_st_hr, how="inner", left_index=True, right_index=True)
completed_data = pd.merge(completed_data, c_st_gr, how="outer", left_index=True, right_index=True)
completed_data.reset_index(inplace=True)
print(completed_data["HR_ST"].isna().sum())
print(completed_data["GR_ST"].isna().sum())

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 50
fig, axes = plt.subplots(3, 1, figsize=(45, 20))
dates = ["2023-08-11", "2023-08-24", "2024-01-29"]
for i, date in enumerate(dates):
    date_dt = pd.to_datetime(date)
    filtered_completed_data = completed_data[completed_data["Datetime"].dt.date == date_dt.date()]
    filtered_st_data = st_data[st_data.index.date == date_dt.date()]
    ax = axes[i]
    ax.plot(filtered_st_data.index, filtered_st_data["GD_MW"], label="GR ST measured", linestyle="-",linewidth=5, color="dodgerblue")
    ax.plot(filtered_st_data.index, filtered_st_data["HD_MW"], label="HR ST measured", linestyle="-", linewidth=5, color="crimson")
    ax.plot(filtered_completed_data["Datetime"], filtered_completed_data["GR_ST"], label="GR ST modelled", linestyle="--", linewidth=5, color="goldenrod")
    ax.plot(filtered_completed_data["Datetime"], filtered_completed_data["HR_ST"], label="HR ST modelled", linestyle="--", linewidth=5, color="dimgray")
    ax.grid()
    ax.set_xticks(filtered_completed_data["Datetime"][::8])
    ax.set_xticklabels(filtered_completed_data["Datetime"].dt.strftime("%d %b %H:%M")[::8])
    ax.set_xlim(filtered_completed_data["Datetime"].min(), filtered_completed_data["Datetime"].max())
ax.legend(loc="center", bbox_to_anchor=(0.5, -0.35), frameon=False, ncol=4)
plt.savefig("C:\\Users\\linus\\OneDrive\\Dokumente\\Masterarbeit\\Graphen\\Surface Temp Model vs Measure Time Series.pdf", format="pdf")
plt.show()

# QG berechnen

# Hybriddach    
# t1_hr berechnen
# ks berechnen
print(completed_data["SWC_1_1_2"].min())
max_swc = 100
min_swc = 0.0
ks_min = 0.04
ks_max = 0.36
slope_ks = (ks_max-ks_min)/(max_swc - min_swc)
intercept = 0.04
ks = pd.DataFrame(columns=["KS_HR"])
ks["KS_HR"] = slope_ks * completed_data["SWC_1_1_2"] + intercept


QG_HFP_HR = pd.DataFrame(columns=["G_plate_1_1_2"])
QG_HFP_HR["G_plate_1_1_2"] = completed_data[ "G_plate_1_1_2"]
# print(QG_HFP_HR)
t1_hr = pd.DataFrame(columns=["t1_hr"])
t1_hr["t1_hr"] = ((QG_HFP_HR["G_plate_1_1_2"] * 0.06)/ks["KS_HR"]) + (completed_data["TA_1_1_2"]+ 273.16)
# print(t1_hr)

#tM_hr berechnen
tM_hr = pd.DataFrame(columns=["tM_hr"])
tM_hr["tM_hr"] = (t1_hr["t1_hr"] + ((completed_data["HR_ST"])+ 273.16))/2
# print(tM_hr)

# Qs1 Berechnen
# Dichte Marmor: natürlich 3020 kg/m³, 2730 kg/m³ (Asdrub   ali, 2015)
# specific heat marmor: 773 J/kg K (natural), 706 J/kg K (artificial) (Asdrubali, 2015)
# Mittelwerte: 2875 kg/m³, 739,5 J/kg K
# Porosität von Grobkies: 0.27 - 0.38 (Kolymbas, 2016)
# Porosität für Steinwolle: 1 - Rohdichte / Reindichte
# Volumen für ein Panel aus Datenblatt: 0.03*0.97*1.2 = 0.03492 m³
# Aus Trockengewicht und Nassgewicht Porenvolumen ableiten: 34.3 kg - 8.5 kg = 25.8 kg
# Dichte von Wasser auf 1000 kg/m³ angenommen -> 25.8 kg / 1000 kg/m³ = 0.0258 m ³
# Volumen ohne Poren: 0.03492 m³ - 0.0258 m³ = 0.00912 m³
# Porosität = 1 - Rohdichte / Reindichte -> 1 - (8.5/0.03492)/(8.5/0.00912) = 0.7388162
# oder Porenvolumen / Gesamtvolumen -> 0.0258 m³ / 0.03492 m³ = 0.7388162
cm_hr = (1-0.325) * 3020 * 773 + 0 * 1000 * 4180 + (0.325 - 0) * completed_data["Air_Density"] * completed_data["specific_heat_cap_air"] # Heusinger 2017
delta_z = 0.03 # Distanz zwischen T0 und T1
delta_t = 30*60
delta_tM_hr = tM_hr.diff()
print(delta_tM_hr)
Qs1_hr = pd.DataFrame(columns=["Qs1"])
Qs1_hr["Qs1"] = cm_hr * (delta_tM_hr["tM_hr"] / delta_t) * delta_z
Qs1_hr = Qs1_hr.loc[completed_data.index]
# print(Qs1_hr)

#tS berechnen
t2_hr = pd.DataFrame(columns=["TS_CS65X_1_1_2"])
t2_hr["TS_CS65X_1_1_2"] = completed_data["TS_CS65X_1_1_2"]+ 273.16
tS_hr = pd.DataFrame(columns=["ts_hr"])
tS_hr["ts_hr"] = (t1_hr["t1_hr"] + t2_hr["TS_CS65X_1_1_2"])/2
# print(tS_hr)

# Qs2 berechnen
cs_hr = pd.DataFrame(columns=["cs_hr"])
cs_hr["cs_hr"] = (1-0.7388162) * 243.413 * 1030 + (completed_data["VWC_1_1_2"]) * 1000 * 4180 + (0.7388 - (completed_data["VWC_1_1_2"])) * completed_data["Air_Density"] * completed_data["specific_heat_cap_air"] # Heusinger 2017
delta_z = 0.03 # Distanz zwischen T1 und T2
delta_t = 30*60
delta_tS_hr = tS_hr.diff()
print(delta_tS_hr)
Qs2_hr = pd.DataFrame(columns=["Qs2"])
Qs2_hr["Qs2"] = cs_hr["cs_hr"] * (delta_tS_hr["ts_hr"] / delta_t) * delta_z
# print(Qs2_hr)

# QG berechnen
QG_HR = pd.DataFrame(columns=["QG_HR"])
QG_HR["QG_HR"] = QG_HFP_HR["G_plate_1_1_2"] + Qs1_hr["Qs1"] + Qs2_hr["Qs2"]


# Gründach
ks_gr = 0.165 # von Heusinger und Weber 2018 gemessen
# print(ks_gr)
QG_HFP_GR = pd.DataFrame(columns=["G_plate_1_1_1"])
QG_HFP_GR["G_plate_1_1_1"] = completed_data["G_plate_1_1_1"]
# print(QG_HFP_GR)
t1_gr = pd.DataFrame(columns=["t1_gr"])
t1_gr["t1_gr"] = (QG_HFP_GR["G_plate_1_1_1"] * 0.05)/ks_gr + ((completed_data["TA_1_1_2"])+ 273.16)
# print(t1_gr)

#tS_gr berechnen
t2_gr = pd.DataFrame(columns=["TS_CS65X_1_1_1"])
t2_gr["TS_CS65X_1_1_1"] = completed_data["TS_CS65X_1_1_1"]+ 273.16
tS_gr = pd.DataFrame(columns=["tS"])
tS_gr["tS"] = (t1_gr["t1_gr"] + (t2_gr["TS_CS65X_1_1_1"]))/2
# print(tS_gr)

# Qs2 berechnen
# Porosität ist Mittelwert aus Datenblatt
cs_gr = pd.DataFrame(columns=["cs_gr"])
cs_gr["cs_gr"] = (1-0.65) * 670 * 1000 + (completed_data["VWC_1_1_1"]) * 1000 * 4180 + (0.65 - (completed_data["VWC_1_1_1"])) * completed_data["Air_Density"] * 1005 # Heusinger 2017
delta_tS_gr = tS_gr.diff()
delta_z = 0.03
Qs2_gr = pd.DataFrame(columns=["Qs2"])
Qs2_gr["Qs2"] = cs_gr["cs_gr"] * (delta_tS_gr["tS"] / delta_t) * delta_z
# print(Qs2_gr)

# QG berechnen
QG_GR = pd.DataFrame(columns=["QG_GR"])
QG_GR["QG_GR"] = QG_HFP_GR["G_plate_1_1_1"] + Qs2_gr["Qs2"]

# print(QG_GR)
# print(QG_HR)
# print(QG_GR.min())
# print(QG_GR.max())
# print(QG_HR.min())
# print(QG_HR.max())  
# print(QG_GR.mean())
# print(QG_GR.median())
# print(QG_HR.mean())
# print(QG_HR.median())
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 72
plt.figure(figsize=(50,10))
plt.plot(completed_data["Datetime"],QG_HR["QG_HR"], label="QG HR", color="dodgerblue")
plt.plot(completed_data["Datetime"],QG_GR["QG_GR"], label="QG GR", color="crimson")
plt.legend(loc="center", bbox_to_anchor=(0.5,-0.65), ncol=2, frameon=False)
plt.xticks(completed_data["Datetime"][::768], completed_data["Datetime"].dt.strftime("%d. %b")[::768], rotation=45)
plt.xlim(completed_data["Datetime"].min(), completed_data["Datetime"].max())
plt.ylim([QG_HR["QG_HR"].min(),QG_HR["QG_HR"].max()])
plt.yticks([-500,-250,0,250])
plt.grid()
plt.savefig("C:\\Users\\linus\\OneDrive\\Dokumente\\Masterarbeit\\Graphen\\Zeitreihe QGs mit Ausreißern.pdf", format="pdf", bbox_inches="tight")
plt.show


print(QG_GR["QG_GR"].isna().sum())
print(QG_HR["QG_HR"].isna().sum())

# # Statistische Bereinigung von QG via IQA
# Q1_GR = QG_GR["QG_GR"].quantile(0.25)
# Q3_GR = QG_GR["QG_GR"].quantile(0.75)
Q1_HR = QG_HR["QG_HR"].quantile(0.25)
Q3_HR = QG_HR["QG_HR"].quantile(0.75)

# IQR_GR = Q3_GR - Q1_GR
IQR_HR = Q3_HR - Q1_HR

# lower_bound_GR = Q1_GR - 1.5 * IQR_GR
# upper_bound_GR = Q3_GR + 1.5 * IQR_GR
lower_bound_HR = Q1_HR - (1.5 * IQR_HR)
upper_bound_HR = Q3_HR + (1.5 * IQR_HR)
# print(lower_bound_GR)
# print(upper_bound_GR)
# print(lower_bound_HR)
# print(upper_bound_HR)

# QG_GR["QG_GR"] = QG_GR["QG_GR"][(QG_GR["QG_GR"] >= lower_bound_GR) & (QG_GR["QG_GR"] <= upper_bound_GR)]
QG_HR["QG_HR"] = QG_HR["QG_HR"][(QG_HR["QG_HR"] >= lower_bound_HR) & (QG_HR["QG_HR"] <= upper_bound_HR)]

# QG_GR.fillna(value=np.nan, inplace=True)
QG_HR.fillna(value=np.nan, inplace=True)
# print(QG_GR)
# print(QG_HR)
# print(QG_GR.min())
# print(QG_GR.max())
# print(QG_HR.min())
# print(QG_HR.max())
# print(QG_GR.mean())
# print(QG_GR.median())
# print(QG_HR.mean())
# print(QG_HR.median())

# plt.figure(figsize=(75,5))
# plt.plot(completed_data["Datetime"],QG_HR["QG_HR"], label="QG HR")
# plt.plot(completed_data["Datetime"],QG_GR["QG_GR"], label="QG GR")
# plt.legend()
# plt.grid()
# plt.show

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 72
plt.figure(figsize=(50,10))
plt.plot(completed_data["Datetime"],QG_HR["QG_HR"], label="QG HR", color="dodgerblue")
plt.plot(completed_data["Datetime"],QG_GR["QG_GR"], label="QG GR", color="crimson")
plt.legend(loc="center", bbox_to_anchor=(0.5,-0.65), ncol=2, frameon=False)
plt.xticks(completed_data["Datetime"][::768], completed_data["Datetime"].dt.strftime("%d. %b")[::768], rotation=45)
plt.xlim(completed_data["Datetime"].min(), completed_data["Datetime"].max())
plt.ylim([QG_HR["QG_HR"].min(),QG_HR["QG_HR"].max()])
plt.yticks([-500,-250,0,250])
plt.grid()
plt.savefig("C:\\Users\\linus\\OneDrive\\Dokumente\\Masterarbeit\\Graphen\\Zeitreihe QGs.pdf", format="pdf", bbox_inches="tight")
plt.show

# QH berechnen (für alle Tage) nach Heusinger 2018
hc_calc = pd.DataFrame(columns=["hc_gr"])
hc_calc["hc_gr"] = 9.502 + 5.513 * completed_data["WS"] * (268.053 / (415.225 + (completed_data["TA_1_1_2"])+273.16)) # Heusinger 2018

# nach McAdams aus Palyvos 2008
# Bedingung für Wert für a, b, n festlegen
condition_1 = completed_data["WS"] < 4.88
condition_2 = completed_data["WS"] >= 4.88

a_values = [1.09, 0]
b_values = [0.23, 0.53]
n_values = [1, 0.78]

# Werte erstellen und in Datensatz schreiben
hc_calc["a"] = np.select([condition_1, condition_2], a_values)
hc_calc["b"] = np.select([condition_1, condition_2], b_values)
hc_calc["n"] = np.select([condition_1, condition_2], n_values)
hc_calc["hc_hr"] = 5.678 * (hc_calc["a"] + hc_calc["b"] * ((294.26/(273.16 + completed_data["TA_1_1_2"])) * completed_data["WS"]/0.3408)**hc_calc["n"])
# evtl. über Ansatz aus Heusinger 2018 bestimmen
QH_HR = pd.DataFrame(columns=["QH_HR"])
QH_GR = pd.DataFrame(columns=["QH_GR"])
QH_HR["QH_HR"] = hc_calc["hc_hr"]*((completed_data["HR_ST"]+273.16) - (completed_data["TA_1_1_2"]+273.16))
QH_GR["QH_GR"] = hc_calc["hc_gr"]*((completed_data["GR_ST"]+273.16) - (completed_data["TA_1_1_2"]+273.16))



# Palywos 2008 Nusselt für Grenzschicht
Rf = 1.67 # Clear 2003, Rf for rough plaster
a = 0.0296 # FM, local Nu for horizontal roof, turbulent flow, Clear 2003
b = 0.8 # FM, local Nu for horizontal roof, turbulent flow, Clear 2003
c = 0.333 # FM, local Nu for horizontal roof, turbulent flow, Clear 2003
d = 0 # FM, local Nu for horizontal roof, turbulent flow. Clear 2003
Pr = completed_data["Prandtl"]  # engineers edge https://www.engineersedge.com/physics/viscosity_of_air_dynamic_and_kinematic_14483.htm
LR = (2 * 1 * 0.45)/(1+0.45)
Re = (completed_data["WS"] * LR) / (completed_data["Air_kinVis"]) # kinematische Viskositat von https://www.engineersedge.com/physics/viscosity_of_air_dynamic_and_kinematic_14483.htm 
Nu = 0.0296 * Rf * (Re ** b) * (Pr ** c) + d
hc_calc["hc_nu"] = (Nu * completed_data["Air_Lambda"]) / LR

QH_HR["QH_NU"] = hc_calc["hc_nu"]*((completed_data["HR_ST"]+273.16) - (completed_data["TA_1_1_2"]+273.16))

# plt.figure(figsize=(30,5))
# plt.plot(completed_data["Datetime"],QH_HR["QH_HR"], label="QH HR")
# plt.plot(completed_data["Datetime"],QH_HR["QH_NU"], label="QH NU", linestyle=":")
# plt.legend()
# plt.grid()
# plt.show


# print(QH_GR["QH_GR"].min())
# print(QH_GR["QH_GR"].max())
# print(QH_GR["QH_GR"].mean())
# print(QH_GR["QH_GR"].median())
# print(QH_HR["QH_HR"].min())
# print(QH_HR["QH_HR"].max())
# print(QH_HR["QH_HR"].mean())
# print(QH_HR["QH_HR"].median())
# print(QH_HR["QH_NU"].min())
# print(QH_HR["QH_NU"].max())
# print(QH_HR["QH_NU"].mean())
# print(QH_HR["QH_NU"].median())

# fig, ax1 = plt.subplots(figsize=(75, 5))
# ax1.plot(completed_data["Datetime"],QH_HR["QH_HR"], label="QH HR", linestyle="--")
# ax1.plot(completed_data["Datetime"],QH_GR["QH_GR"], label="QH GR", linestyle="-")
# ax1.plot(completed_data["Datetime"],QH_HR["QH_NU"], label="QH NU", linestyle=":")
# ax2 = ax1.twinx()
# ax2.plot(completed_data["Datetime"],completed_data["TA_1_1_2"], label="TA", color="red", linestyle="-.")
# ax1.legend(loc="center",ncol=3, bbox_to_anchor=(0.2, -0.3), frameon=False)
# ax2.legend(loc="center",ncol=1, bbox_to_anchor=(0.5, -0.3), frameon=False)
# plt.grid()
# plt.show

# Statistische Bereinigung von QH via IQA
# händische Entfernung von Ausreißern noch implementieren
Q1_GR = QH_GR["QH_GR"].quantile(0.25)
Q3_GR = QH_GR["QH_GR"].quantile(0.75)
Q1_HR = QH_HR["QH_HR"].quantile(0.25)
Q3_HR = QH_HR["QH_HR"].quantile(0.75)
Q1_NU = QH_HR["QH_NU"].quantile(0.25)
Q3_NU = QH_HR["QH_NU"].quantile(0.75)


IQR_GR = Q3_GR - Q1_GR
IQR_HR = Q3_HR - Q1_HR
IQR_NU = Q3_NU - Q1_NU

lower_bound_GR = Q1_GR - 1.5 * IQR_GR
upper_bound_GR = Q3_GR + 1.5 * IQR_GR
lower_bound_HR = Q1_HR - 1.5 * IQR_HR
upper_bound_HR = Q3_HR + 1.5 * IQR_HR
lower_bound_NU = Q1_NU - 1.5 * IQR_NU
upper_bound_NU = Q3_NU + 1.5 * IQR_NU

QH_GR["QH_GR"] = QH_GR["QH_GR"][(QH_GR["QH_GR"] >= lower_bound_GR)]
QH_HR["QH_HR"] = QH_HR["QH_HR"][(QH_HR["QH_HR"] >= lower_bound_HR)]
QH_HR["QH_NU"] = QH_HR["QH_NU"][(QH_HR["QH_NU"] >= lower_bound_NU)]
QH_GR.fillna(value=np.nan, inplace=True)
QH_HR.fillna(value=np.nan, inplace=True)

# print(QH_HR)
# print(QH_GR)
# print(QH_HR.max())
# print(QH_HR.min())
# print(QH_GR.max())
# print(QH_GR.min())
# print(QH_HR.mean())
# print(QH_HR.median())
# print(QH_GR.mean())
# # print(QH_GR.median())
# fig, ax1 = plt.subplots(figsize=(75, 5))
# ax1.plot(completed_data["Datetime"],QH_HR["QH_HR"], label="QH HR", linestyle="--")
# ax1.plot(completed_data["Datetime"],QH_GR["QH_GR"], label="QH GR", linestyle="-")
# ax1.plot(completed_data["Datetime"],QH_HR["QH_NU"], label="QH NU", linestyle=":")
# ax2 = ax1.twinx()
# ax2.plot(completed_data["Datetime"],completed_data["TA_1_1_2"], label="TA", color="red", linestyle="-.")
# ax1.legend(loc="center",ncol=3, bbox_to_anchor=(0.2, -0.3), frameon=False)
# ax2.legend(loc="center",ncol=1, bbox_to_anchor=(0.5, -0.3), frameon=False)
# plt.grid()
# plt.show

# QE Berechnen
# über Penman Monteith (aus Heusinger 2017)
# evtl. zusätzlich noch über Strahlungsbilanz berechnen?


# # Surface Area of the modules
# surface_area_gr = 0.45
# surface_area_hr = 0.45

#Bestimmung über Dreisatzy
#Basis für Dreisatz (b/a)
# base_L_gr = w_gr / bf_gr
# base_L_hr = w_hr / bf_hr

# swc_gr_mm = pd.DataFrame(columns=["SWC_1_1_1"])
# swc_hr_mm = pd.DataFrame(columns=["SWC_1_1_2"])
# swc_gr_mm["SWC_1_1_1"] = completed_data["SWC_1_1_1"]
# swc_hr_mm["SWC_1_1_2"] = completed_data["SWC_1_1_2"]


# # Berechnen des SWC in L mittels Dreisatz (c * (b/a))
# swc_gr_mm["SWC_1_1_1"] = swc_gr_mm["SWC_1_1_1"] * base_L_gr
# swc_hr_mm["SWC_1_1_2"] = swc_hr_mm["SWC_1_1_2"] * base_L_hr


# # Umrechnen von L in mm
# swc_gr_mm["SWC_1_1_1"] = swc_gr_mm["SWC_1_1_1"] * surface_area_gr
# swc_hr_mm["SWC_1_1_2"] = swc_hr_mm["SWC_1_1_2"] * surface_area_hr

# # Berechnen von deltaSWC
# swc_gr_mm["deltaSWC"] = swc_gr_mm["SWC_1_1_1"].diff()
# swc_hr_mm["deltaSWC"] = swc_hr_mm["SWC_1_1_2"].diff()

# # Berechnen von Zeitunterschied deltat in Sekunden
# swc_gr_mm.reset_index(inplace=True)
# swc_hr_mm.reset_index(inplace=True)
# swc_gr_mm["deltat"] = swc_gr_mm["Datetime"].diff().dt.total_seconds() 
# swc_hr_mm["deltat"] = swc_hr_mm["Datetime"].diff().dt.total_seconds() 
# swc_gr_mm = swc_gr_mm.set_index("Datetime")
# swc_hr_mm = swc_hr_mm.set_index("Datetime")
# print(swc_gr_mm)
# print(swc_hr_mm)

# Wasserdichte
roh_W = 1000

# ET = QE/(1918.46 * (TA/TA-33.91)*(roh_W/1000)) nach QE umstellen
# QE_GR = pd.DataFrame(columns=["QE_GR"])
# QE_HR = pd.DataFrame(columns=["QE_HR"])
# QE_GR["QE_GR"] = (swc_gr_mm["deltaSWC"]/swc_gr_mm["deltat"])*(1918.46 * (((t3_gr["TA_1_1_2"] + 273.15)/(t3_gr["TA_1_1_2"] + 273.15)-33.91)**2) * (roh_W/1000))
# QE_HR["QE_HR"] = (swc_hr_mm["deltaSWC"]/swc_gr_mm["deltat"])*(1918.46 * (((t3_gr["TA_1_1_2"] + 273.15)/(t3_gr["TA_1_1_2"] + 273.15)-33.91)**2) * (roh_W/1000))
# print(QE_GR)
# print(QE_HR)

# QE nach Versini 2024
mdata.reset_index(inplace = True)
QE = []
Regentage = pd.DataFrame()
n = 0
QE_GR = pd.DataFrame(index=range(len(completed_data)), columns=["Datetime", "QE_GR"])
QE_HR = pd.DataFrame(index=range(len(completed_data)), columns=["Datetime", "QE_HR"])
swc_gr = pd.DataFrame(index=range(len(completed_data)), columns=["Datetime", "VWC_1_1_1", "ET_GR"])
swc_hr = pd.DataFrame(index=range(len(completed_data)), columns=["Datetime", "VWC_1_1_2", "ET_HR"])
for i in range(0,len(mdata)):
    swc_gr.loc[i, "Datetime"] = completed_data.loc[i, "Datetime"]
    swc_hr.loc[i, "Datetime"] = completed_data.loc[i, "Datetime"]
    swc_gr.loc[i, "TS_CS65X_1_1_1"] = completed_data.loc[i, "TS_CS65X_1_1_1"]
    swc_hr.loc[i, "TS_CS65X_1_1_2"] = completed_data.loc[i, "TS_CS65X_1_1_2"]
    QE_GR.loc[i, "Datetime"] = completed_data.loc[i, "Datetime"]
    QE_HR.loc[i, "Datetime"] = completed_data.loc[i, "Datetime"]
    if (mdata.loc[i, "Precipitation"] > 0):
        n = 0
        Regentage.loc[i, "RT"]=mdata.index[i]
    else:
        n = n+1
    if (n >= 144) & (pd.isna(mdata.loc[i, "Precipitation"])==False) | ((mdata.loc[i, "Datetime"].date() > pd.to_datetime("2023-07-25").date()) & (i < 947)):
        if (swc_gr.loc[i, "TS_CS65X_1_1_1"] >= 0):
            swc_gr.loc[i, "VWC_1_1_1"] = completed_data.loc[i, "VWC_1_1_1"]
            swc_gr.loc[i, "VWC_1_1_1"] = swc_gr.loc[i, "VWC_1_1_1"]
            swc_gr.loc[i, "ET_GR"] = (((swc_gr.loc[i, "VWC_1_1_1"] - swc_gr.loc[i-1, "VWC_1_1_1"])/1800)*0.06)*(-1)
            QE_GR.loc[i, "QE_GR"] = swc_gr.loc[i, "ET_GR"] * (2.45*(10**6)) * roh_W
        if (swc_hr.loc[i, "TS_CS65X_1_1_2"] >= 0):
            swc_hr.loc[i, "VWC_1_1_2"] = completed_data.loc[i, "VWC_1_1_2"]
            swc_hr.loc[i, "VWC_1_1_2"] = swc_hr.loc[i, "VWC_1_1_2"]
            swc_hr.loc[i, "ET_HR"] = (((swc_hr.loc[i, "VWC_1_1_2"] - swc_hr.loc[i-1, "VWC_1_1_2"])/1800)*0.06)*(-1)
            QE_HR.loc[i, "QE_HR"] = swc_hr.loc[i, "ET_HR"] * (2.45*(10**6)) * roh_W
    else:
        QE_GR.loc[i, "QE_GR"]= np.nan
        QE_HR.loc[i, "QE_HR"]= np.nan


QE_GR["Datetime"] = pd.to_datetime(QE_GR["Datetime"])
QE_HR["Datetime"] = pd.to_datetime(QE_HR["Datetime"])

# print(QE_GR.max())
# print(QE_GR.min())
# print(QE_HR.max())
# print(QE_HR.min())




QE_GR["QE_GR"] = QE_GR["QE_GR"].apply(pd.to_numeric)
QE_HR["QE_HR"] = QE_HR["QE_HR"].apply(pd.to_numeric)

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 72
plt.figure(figsize=(50,10))
plt.plot(completed_data["Datetime"],QE_HR["QE_HR"], label="QE HR", linewidth=5, color="dodgerblue")
plt.plot(completed_data["Datetime"],QE_GR["QE_GR"], label="QE GR", linewidth=5, color="crimson")
plt.legend(loc="center", bbox_to_anchor=(0.5,-0.65), ncol=2, frameon=False)
plt.xticks(completed_data["Datetime"][::768], completed_data["Datetime"].dt.strftime("%d. %b")[::768], rotation=45)
plt.xlim(completed_data["Datetime"].min(), completed_data["Datetime"].max())
plt.grid()
plt.savefig("C:\\Users\\linus\\OneDrive\\Dokumente\\Masterarbeit\\Graphen\\Zeitreihe QEs mit Ausreißern.pdf", format="pdf", bbox_inches="tight")
plt.show

# # # Statistische Bereinigung von QE via IQA
Q10_GR = QE_GR["QE_GR"].quantile(0.05)
Q90_GR = QE_GR["QE_GR"].quantile(0.95)
Q10_HR = QE_HR["QE_HR"].quantile(0.05)
Q90_HR = QE_HR["QE_HR"].quantile(0.95)
# print(Q10_GR)
# print(Q90_GR)
# print(Q10_HR)
# print(Q90_GR)
IQR_GR = Q90_GR - Q10_GR
IQR_HR = Q90_GR - Q10_HR


lower_bound_GR = Q10_GR - 2 * IQR_GR
upper_bound_GR = Q90_GR + 2 * IQR_GR
lower_bound_HR = Q10_HR - 2 * IQR_HR
upper_bound_HR = Q90_GR + 2 * IQR_HR
# print(lower_bound_GR)
# print(upper_bound_GR)
# print(lower_bound_HR)
# print(upper_bound_HR)

QE_GR["QE_GR"] = QE_GR["QE_GR"][(QE_GR["QE_GR"] >= lower_bound_GR) & (QE_GR["QE_GR"] <= upper_bound_GR)]
QE_HR["QE_HR"] = QE_HR["QE_HR"][(QE_HR["QE_HR"] >= lower_bound_HR) & (QE_HR["QE_HR"] <= upper_bound_HR)]

QE_GR.fillna(value=np.nan, inplace=True)
QE_HR.fillna(value=np.nan, inplace=True)

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 72
plt.figure(figsize=(50,10))
plt.plot(completed_data["Datetime"],QE_HR["QE_HR"], label="QE HR", linewidth=5, color="dodgerblue")
plt.plot(completed_data["Datetime"],QE_GR["QE_GR"], label="QE GR", linewidth=5, color="crimson")
plt.legend(loc="center", bbox_to_anchor=(0.5,-0.65), ncol=2, frameon=False)
plt.xticks(completed_data["Datetime"][::768], completed_data["Datetime"].dt.strftime("%d. %b")[::768], rotation=45)
plt.xlim(completed_data["Datetime"].min(), completed_data["Datetime"].max())
plt.grid()
plt.savefig("C:\\Users\\linus\\OneDrive\\Dokumente\\Masterarbeit\\Graphen\\Zeitreihe QEs.pdf", format="pdf", bbox_inches="tight")
plt.show

# fig, ax1 = plt.subplots(figsize=(75,5))
# ax1.plot(QE_GR.index, QE_GR["QE_GR"], linestyle="--", label="GR")
# ax1.plot(QE_HR.index, QE_HR["QE_HR"], linestyle="--", label="HR")
# ax2 = ax1.twinx()
# ax2.plot(mdata["Datetime"], mdata["Precipitation"])
# plt.legend()
# plt.grid()
# plt.show()


# plt.figure(figsize=(30,5))
# plt.plot(QE_GR.loc[(QE_GR["Datetime"].dt.date >= pd.to_datetime("2023-09-04").date()) & (QE_GR["Datetime"].dt.date <= pd.to_datetime("2023-09-11").date()),"Datetime"], QE_GR.loc[(QE_GR["Datetime"].dt.date >= pd.to_datetime("2023-09-04").date()) & (QE_GR["Datetime"].dt.date <= pd.to_datetime("2023-09-11").date()),"QE_GR"])
# plt.plot(QE_HR.loc[(QE_HR["Datetime"].dt.date >= pd.to_datetime("2023-09-04").date()) & (QE_HR["Datetime"].dt.date <= pd.to_datetime("2023-09-11").date()),"Datetime"], QE_HR.loc[(QE_HR["Datetime"].dt.date >= pd.to_datetime("2023-09-04").date()) & (QE_HR["Datetime"].dt.date <= pd.to_datetime("2023-09-11").date()),"QE_HR"])
# plt.grid()
# plt.show()

# plt.figure(figsize=(30,5))
# plt.plot(QE_GR.loc[(QE_GR["Datetime"].dt.date >= pd.to_datetime("2023-07-08").date()) & (QE_GR["Datetime"].dt.date <= pd.to_datetime("2023-07-09").date()),"Datetime"], QE_GR.loc[(QE_GR["Datetime"].dt.date >= pd.to_datetime("2023-07-08").date()) & (QE_GR["Datetime"].dt.date <= pd.to_datetime("2023-07-11").date()),"QE_GR"])
# plt.plot(QE_HR.loc[(QE_HR["Datetime"].dt.date >= pd.to_datetime("2023-07-08").date()) & (QE_HR["Datetime"].dt.date <= pd.to_datetime("2023-07-09").date()),"Datetime"], QE_HR.loc[(QE_HR["Datetime"].dt.date >= pd.to_datetime("2023-07-08").date()) & (QE_HR["Datetime"].dt.date <= pd.to_datetime("2023-07-11").date()),"QE_HR"])
# plt.grid()
# plt.show()

# print(QE_GR.max())
# print(QE_GR.min())
# print(QE_HR.max())
# print(QE_HR.min())

# mdata.set_index("Datetime", inplace=True)
# completed_data.set_index("Datetime", inplace=True)
# filtered_indices = []
# for index in mdata.index:
#     if (index - pd.Timedelta(minutes=30) * 144) in mdata.index:
#         if all(mdata.loc[index - pd.Timedelta(minutes=30) * i, "Precipitation"] == 0 for i in range(0, 144)):
#             filtered_indices.append(index)

# filtered_data = completed_data.loc[filtered_indices]    
# swc_gr = pd.DataFrame(columns=["SWC_1_1_1"])
# swc_hr = pd.DataFrame(columns=["SWC_1_1_2"])
# swc_gr["SWC_1_1_1"] = filtered_data["SWC_1_1_1"]
# swc_hr["SWC_1_1_2"] = filtered_data["SWC_1_1_2"]
# swc_gr.index.name = "Datetime"
# swc_hr.index.name = "Datetime"
# swc_gr["SWC_1_1_1"] = swc_gr["SWC_1_1_1"]/100
# swc_hr["SWC_1_1_2"] = swc_hr["SWC_1_1_2"]/100
# swc_gr["deltaSWC_1_1_1"] = swc_gr["SWC_1_1_1"].diff()
# swc_hr["deltaSWC_1_1_2"] = swc_hr["SWC_1_1_2"].diff()
# swc_gr.reset_index(inplace=True)
# swc_hr.reset_index(inplace=True)
# swc_gr["deltat"] = swc_gr["Datetime"].diff().dt.total_seconds()
# swc_hr["deltat"] = swc_hr["Datetime"].diff().dt.total_seconds()
# swc_gr.set_index("Datetime", inplace=True)
# swc_hr.set_index("Datetime", inplace=True)
# swc_gr["ET_GR"] = -(swc_gr["deltaSWC_1_1_1"]/swc_gr["deltat"])*0.06
# swc_hr["ET_HR"] = -(swc_hr["deltaSWC_1_1_2"]/swc_hr["deltat"])*0.06
# QE_GR = pd.DataFrame(columns=["QE_GR"])
# QE_HR = pd.DataFrame(columns=["QE_HR"])
# QE_GR["QE_GR"] = swc_gr["ET_GR"] * (2.45*(10**6)) * roh_W
# QE_HR["QE_HR"] = swc_hr["ET_HR"] * (2.45*(10**6)) * roh_W



completed_data.set_index("Datetime", inplace = True)
QE_GR.set_index("Datetime", inplace = True)
QE_HR.set_index("Datetime", inplace = True)
QG_GR.set_index(completed_data.index, inplace=True)
QG_HR.set_index(completed_data.index, inplace=True)
QH_GR.set_index(completed_data.index, inplace=True)
QH_HR.set_index(completed_data.index, inplace=True)
completed_data = pd.merge(completed_data, QG_GR, how="outer", left_index=True, right_index=True)
completed_data = pd.merge(completed_data, QH_GR, how="outer", left_index=True, right_index=True)
completed_data = pd.merge(completed_data, QE_GR, how="outer", left_index=True, right_index=True)
completed_data = pd.merge(completed_data, QG_HR, how="outer", left_index=True, right_index=True)
completed_data = pd.merge(completed_data, QH_HR, how="outer", left_index=True, right_index=True)
completed_data = pd.merge(completed_data, QE_HR, how="outer", left_index=True, right_index=True)
completed_data.index.name = "Datetime"
completed_data.reset_index(inplace=True)
# print(completed_data)

# LW OUT und Q* Berechnung
bk = (5.67*(10**(-8)))
lwo_gr = pd.DataFrame(columns=["LW_OUT"])
lwo_hr = pd.DataFrame(columns=["LW_OUT"])
lwo_hr["LW_OUT"] = (0.9675  * bk * ((completed_data["HR_ST"]+273.15)**4))
lwo_gr["LW_OUT"] = (0.965 * bk * ((completed_data["GR_ST"]+273.15)**4)) 
lwo_gr["Datetime"] = completed_data["Datetime"]
lwo_gr["Time"] = completed_data["Time"]
lwo_hr["Datetime"] = completed_data["Datetime"]
lwo_hr["Time"] = completed_data["Time"]

completed_data["LW_OUT_GR_Calculated"] = lwo_gr["LW_OUT"]
completed_data["LW_OUT_HR_Calculated"] = lwo_hr["LW_OUT"]
completed_data["Q*_rad_gr"] = completed_data["SW_IN"] + (completed_data["SW_OUT_GR_Calculated"] * (-1)) + completed_data["LW_IN"] + (completed_data["LW_OUT_GR_Calculated"] * (-1))
completed_data["Q*_flux_gr"] = completed_data["QE_GR"] + completed_data["QG_GR"] + completed_data["QH_GR"]

completed_data["Q*_rad_hr"] = completed_data["SW_IN"] + (completed_data["SW_OUT_HR_Calculated"] * (-1)) + completed_data["LW_IN"] + (completed_data["LW_OUT_HR_Calculated"] * (-1))
completed_data["Q*_flux_hr"] = completed_data["QE_HR"] + completed_data["QG_HR"] + completed_data["QH_HR"]

selected_columns = ["TA_1_1_2", "RH_1_1_2", "TS_CS65X_1_1_1", "TS_CS65X_1_1_2", "SWC_1_1_1", "SWC_1_1_2", "VWC_1_1_1", "VWC_1_1_2",
                    "G_plate_1_1_1", "G_plate_1_1_2", "WS", "SW_IN", "SW_OUT", "LW_IN", "LW_OUT", "GR_ST", "HR_ST",
                    "QG_GR", "QE_GR", "QH_GR", "QG_HR", "QE_HR", "QH_HR", "Q*_flux_gr", "Q*_flux_hr",
                    "Q*_rad_gr", "Q*_rad_hr", "SW_OUT_GR_Calculated", "SW_OUT_HR_Calculated", "LW_OUT_GR_Calculated",
                    "LW_OUT_HR_Calculated", "Precipitation"]
descriptive_stats = completed_data[selected_columns]
descriptive_stats = descriptive_stats.describe()
descriptive_stats.to_csv("C:\\Users\\linus\\OneDrive\\Dokumente\\Publikation\\Data\\descriptive_stats.csv", sep = ";", decimal=",", header = True)

completed_data.to_csv("C:\\Users\\linus\\OneDrive\\Dokumente\\Publikation\\Data\\completed_data.csv", sep = ",", header = True, index=False)

