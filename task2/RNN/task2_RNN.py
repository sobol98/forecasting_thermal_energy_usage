import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import datetime

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import onnx
import onnxruntime as ort


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
pogoda_df['datetime'] = pd.to_datetime(pogoda_df.iloc[:, 0]) 


cieplo_df = cieplo_df.sort_values(by='datetime')
pogoda_df = pogoda_df.sort_values(by='datetime')

# print(cieplo_df['datetime'])
# print(pogoda_df['datetime'])


merged_df = pd.merge(pogoda_df[['datetime', 'Średnia_godzinowa']], cieplo_df[['datetime', 'StanLicznikaCiepła']], on='datetime', how='inner')

merged_df['cieplo_diff'] = merged_df['StanLicznikaCiepła'].diff()

merged_df['time_diff'] = (merged_df['datetime'] - merged_df['datetime'].shift(1)).dt.total_seconds() / 3600  # w godzinach
merged_df['cieplo_diff'] = merged_df['cieplo_diff'] / merged_df['time_diff']


merged_df = merged_df[(merged_df['cieplo_diff'] > 0) & (merged_df['cieplo_diff'] < 500)]  # Dostosuj zakres, jeśli 500 GJ jest zbyt duże


merged_df['date'] = merged_df['datetime'].dt.date

daily_agg_df = merged_df.groupby('date').agg(
    mean_temperature=('Średnia_godzinowa', 'mean'),
    mean_cieplo_diff=('cieplo_diff', 'mean')
).reset_index()

# daily_agg_df.to_excel('daily_aggregated_data.xlsx', index=False)


# --------------------------  ML ------------------------------------

# RNN model

# Data
X = daily_agg_df['mean_temperature'].values.reshape(-1, 1)  # temperatura zewnętrzna
y = daily_agg_df['mean_cieplo_diff'].values  # zużycie ciepła 


# Normalizacja danych
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X = scaler_X.fit_transform(X)
y = scaler_y.fit_transform(y.reshape(-1, 1))


window_size = 20
X_train, y_train = [], []
for i in range(len(X) - window_size):
    X_train.append(X[i:i+window_size])
    y_train.append(y[i + window_size])

X_train, y_train = np.array(X_train), np.array(y_train)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

# Stworzenie DataLoadera
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)



# Definicja modelu RNN
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])  # używamy wyjścia z ostatniego kroku czasowego
        return out

# Ustawienia modelu
input_size = 1
hidden_size = 50
output_size = 1
num_epochs = 30
learning_rate = 0.001

model = RNNModel(input_size, hidden_size, output_size)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

train_losses = []
# Trenowanie modelu
for epoch in range(num_epochs):
    epoch_loss = 0
    for X_batch, y_batch in train_loader:
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        
    avg_epoch_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_epoch_loss)    
    
            
    if (epoch+1) % 1 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')



current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
filename = f"training_loss_{current_time}.png" 


plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs+1), train_losses, label="Błąd treningowy (MSE)")
plt.xlabel("Epoka")
plt.ylabel("Błąd (MSE)")
plt.title("Wykres uczenia modelu RNN")
plt.legend()
plt.grid(True)

plt.savefig(filename)
print(f"Wykres zapisany jako {filename}")


torch.save(model.state_dict(), f'rnn_model_{current_time}_.pth')
print("Model został zapisany.")

# Prognozy
model.eval()
with torch.no_grad():
    predictions = model(X_train_tensor).numpy()
    predictions = scaler_y.inverse_transform(predictions)
    y_train_actual = scaler_y.inverse_transform(y_train)

# Wykres rzeczywistego zużycia ciepła i prognoz

dates_for_plot = np.array(daily_agg_df['date'][window_size:])

# Upewnienie się, że y_train_actual również jest jednowymiarowe
y_train_actual = y_train_actual.ravel()


plt.figure(figsize=(12, 6))
plt.plot(dates_for_plot, y_train_actual, label='Rzeczywiste zużycie ciepła', color='blue')
plt.plot(dates_for_plot, predictions, label='Prognoza zużycia ciepła', color='red')
plt.xlabel("Data")
plt.ylabel("Zużycie ciepła (GJ)")
plt.title("Rzeczywiste vs Prognozowane Zużycie Ciepła - Model RNN (PyTorch)")
plt.legend()
plt.grid(True)
plt.show()


