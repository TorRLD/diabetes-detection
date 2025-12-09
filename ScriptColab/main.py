import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, classification_report # <--- ADICIONADO
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Carregar Dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
columns = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
df = pd.read_csv(url, names=columns)

X = df.iloc[:, 0:8].values.astype(np.float32)
y = df.iloc[:, 8].values.astype(np.int32)

# 2. Normalizar e Dividir
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 3. Treinar Modelo
model = keras.Sequential([
    keras.layers.Dense(12, activation='relu', input_shape=(8,)),
    keras.layers.Dense(8, activation='relu'),
    keras.layers.Dense(2, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=250, batch_size=16, verbose=0)

# --- AQUI ESTÁ O QUE FALTAVA (Itens 2.i.3 e 2.i.4) ---
print("\n" + "="*40)
print("RELATÓRIO DE DESEMPENHO (MATRIZ E MÉTRICAS)")
print("="*40)

# Fazer predições no conjunto de teste
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

# (3) Matriz de Confusão
cm = confusion_matrix(y_test, y_pred)
print("\n>>> Matriz de Confusão:")
print(cm)
print("(Linha=Real, Coluna=Predito | [VN FP] / [FN VP])")

# (4) Relatório de Classificação (Precision, Recall, F1)
print("\n>>> Relatório de Classificação:")
print(classification_report(y_test, y_pred, target_names=['Negativo', 'Positivo']))
print("="*40)
# -----------------------------------------------------

# 4. Converter para TFLite e Gerar Header (Igual ao anterior)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

def hex_to_c_array(data, var_name):
    c_str = '#ifndef MODEL_DATA_H\n#define MODEL_DATA_H\n\n#include <stdint.h>\n\n'
    c_str += f'const unsigned char {var_name}[] __attribute__((aligned(8))) = {{\n'
    for i, val in enumerate(data):
        c_str += f'0x{val:02x}, '
        if (i + 1) % 12 == 0: c_str += '\n'
    c_str += '\n};\n\n'
    c_str += f'const int {var_name}_len = {len(data)};\n\n#endif // MODEL_DATA_H\n'
    return c_str

with open('model_data.h', 'w') as f:
    f.write(hex_to_c_array(tflite_model, 'model_data'))

print("\nArquivo 'model_data.h' atualizado.")