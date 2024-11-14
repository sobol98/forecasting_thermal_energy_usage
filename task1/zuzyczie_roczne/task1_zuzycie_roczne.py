import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

XLSX_FILE = './Pogoda_gdańsk_wykonanie.xlsx'
CSV_FILE = './LicznikCiepla.csv'

XLSX_columns = [
    'Data', 'Temperatura_Świbno', 'Prędkość_wiatru_Świbno', 'Nasłonecznienie_Świbno',
    'Temperatura_Port_Rebiechowo', 'Prędkość_wiatru_Port_Rebiechowo', 'Nasłonecznienie_Port_Rebiechowo',
    'Średnia_godzinowa'
]


pogoda_df=pd.read_excel(XLSX_FILE, skiprows=3, header=None, names=XLSX_columns)
cieplo_df=pd.read_csv(CSV_FILE, sep='\t', encoding='windows-1250', header=0)  #ANSI = windows-1250 or ISO-8859-1

# print(pogoda_df.head())
# print(cieplo_df.head())

cieplo_df['datetime'] = pd.to_datetime(cieplo_df.iloc[:, 0]) + pd.Timedelta(minutes=5)
pogoda_df['datetime'] = pd.to_datetime(pogoda_df.iloc[:, 0])  # Zakładamy, że data jest w kolumnie o nazwie 'datetime'


cieplo_df = cieplo_df.sort_values(by='datetime')
pogoda_df = pogoda_df.sort_values(by='datetime')

print(cieplo_df['datetime'])
print(pogoda_df['datetime'])





merged_df = pd.merge(pogoda_df[['datetime', 'Średnia_godzinowa']], cieplo_df[['datetime', 'StanLicznikaCiepła']], on='datetime', how='inner')

merged_df['cieplo_diff'] = merged_df['StanLicznikaCiepła'].diff()

# merged_df['time_diff'] = (merged_df['datetime'] - merged_df['datetime'].shift(1)).dt.total_seconds() / 3600
# merged_df['cieplo_diff'] = merged_df['cieplo_diff'] / merged_df['time_diff']


merged_df = merged_df[(merged_df['cieplo_diff'] > 0) & (merged_df['cieplo_diff'] < 500)] 

merged_df['year'] = merged_df['datetime'].dt.year
merged_df_2020=merged_df[merged_df['year']==2020]

merged_df_2020 = merged_df_2020.copy() 
merged_df_2020.loc[:,'month'] = merged_df_2020['datetime'].dt.month

monthly_agg_df = merged_df_2020.groupby('month').agg(
    total_cieplo_diff=('cieplo_diff', 'sum')
).reset_index()

monthly_agg_df['month_name'] = monthly_agg_df['month'].apply(lambda x: pd.to_datetime(f'2020-{x:02d}-01').strftime('%B'))

monthly_agg_df.to_excel('monthly_aggregated_data_2020.xlsx', index=False)


# # ------------------------------
# # Wyświetlanie dat i wartości, gdzie zużycie ciepła przekracza 3 GJ
# threshold_h = 3  # Próg zużycia ciepła
# threshold_l = 0
# # Filtrowanie na podstawie wartości 'cieplo_diff'
# high_consumption_dates = merged_df[merged_df['cieplo_diff'] > threshold_h][['datetime', 'cieplo_diff']]
# low_consumption_dates = merged_df[merged_df['cieplo_diff'] < threshold_l][['datetime', 'cieplo_diff']]

# # Wypisanie dat i wartości przekraczających próg
# print("Daty, dla których zużycie ciepła jest większe niż", threshold_h, "GJ:")
# print(high_consumption_dates)
# print("Daty, dla których zużycie ciepła jest mniejsze niż", threshold_l, "GJ:")
# print(low_consumption_dates)
# ------------------------------


x = monthly_agg_df['month_name']
y = monthly_agg_df['total_cieplo_diff']



# x = x[1:]
# y = y[1:]

plt.figure(figsize=(10, 6))
# sns.scatterplot(x=x, y=y)

sns.barplot(x=monthly_agg_df['month_name'], y=monthly_agg_df['total_cieplo_diff'], palette='viridis')
plt.title("Sumaryczne zużycie ciepła w każdym miesiącu 2020 roku")
plt.xlabel("Miesiąc")
plt.ylabel("Sumaryczne zużycie ciepła (GJ)")

plt.ylim(0, y.max() * 1.1)
plt.xticks(rotation=45)
plt.tight_layout() 

plt.grid(True)
plt.show()