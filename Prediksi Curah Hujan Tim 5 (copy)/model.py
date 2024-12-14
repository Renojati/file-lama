import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle

# Load Dataset
df_cuaca = pd.read_excel("C:\\Users\\ACER\\Downloads\\Coba\\Coba\\dataset\\Data_Set_ Stasiun_Meteorologi_Citeko_2013-2023.xlsx")

# Preprocessing
df_cuaca.drop(['station_id', 'ddd_car'], axis=1, inplace=True)
df_cuaca['Tanggal'] = pd.to_datetime(df_cuaca['Tanggal'], format='%d-%m-%Y')
df_cuaca = df_cuaca.dropna()
df_cuaca = df_cuaca.fillna(df_cuaca.mean())

# Feature and Target Definition
fitur = df_cuaca[['Tn', 'Tx', 'Tavg', 'RH_avg', 'ss', 'ff_avg']].astype(float)
target = (df_cuaca['RR'] > 0).astype(int)  # Binary target: 1 for rain, 0 for no rain

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(fitur, target, test_size=0.4, random_state=100)

# Initialize and Train Model
model = GaussianNB()
model.fit(X_train, y_train)

# Evaluate Model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy * 100:.2f}%")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(classification_rep)

# Save Model
with open('naive_bayes_model_fix.pkl', 'wb') as file:
    pickle.dump(model, file)

# Prediction Function
def predict_rain(Tn, Tx, Tavg, RH_avg, ss, ff_avg):
    """
    Predict rain based on input features.
    Returns "Ya" if rain is predicted, "Tidak" otherwise.
    """
    data = pd.DataFrame({'Tn': [Tn], 'Tx': [Tx], 'Tavg': [Tavg], 'RH_avg': [RH_avg], 'ss': [ss], 'ff_avg': [ff_avg]})
    prediction = model.predict(data)
    return "Ya" if prediction[0] == 1 else "Tidak"

# Example Usage
if __name__ == "__main__":
    print("Example Prediction:")
    print(predict_rain(20, 30, 25, 80, 5, 3))
