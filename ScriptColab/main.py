import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# 1. Load Dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
columns = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
df = pd.read_csv(url, names=columns)

# Separate inputs (X) and outputs (y)
X = df.iloc[:, 0:8].values.astype(np.float32)
y = df.iloc[:, 8].values.astype(np.int32)

# 2. Normalize Data (Using MinMaxScaler as used in your C code)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 3. Create Model
model = keras.Sequential([
    keras.layers.Dense(12, activation='relu', input_shape=(8,)),
    keras.layers.Dense(8, activation='relu'),
    keras.layers.Dense(2, activation='softmax') # Output: [Prob_Negative, Prob_Positive]
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
print("Training model...")
model.fit(X_train, y_train, epochs=250, batch_size=16, verbose=0)
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Final Accuracy: {acc*100:.2f}%")

# 4. Convert to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# 5. Generate Header File (.h)
def hex_to_c_array(data, var_name):
    c_str = ''
    c_str += '// model_data.h\n'
    c_str += '#ifndef MODEL_DATA_H\n'
    c_str += '#define MODEL_DATA_H\n\n'
    c_str += '#include <stdint.h>\n\n'
    c_str += '// Note que aqui colocamos o "unsigned char" e os dados juntos\n'
    c_str += f'const unsigned char {var_name}[] __attribute__((aligned(8))) = {{\n'
    
    for i, val in enumerate(data):
        c_str += f'0x{val:02x}, '
        if (i + 1) % 12 == 0:
            c_str += '\n'
            
    c_str += '\n};\n\n'
    c_str += f'const int {var_name}_len = {len(data)};\n\n'
    c_str += '#endif // MODEL_DATA_H\n'
    return c_str

# Write to file
with open('model_data.h', 'w') as f:
    f.write(hex_to_c_array(tflite_model, 'model_data'))

print("-" * 40)
print("File 'model_data.h' generated successfully!")
print("-" * 40)

# 6. Print Normalization Values (To update your C code)
min_vals = scaler.data_min_
max_vals = scaler.data_max_

def format_array_cpp(arr):
    return "{ " + ", ".join(f"{x:.4f}f" for x in arr) + " }"

print("Update these lines in your C code (diabetes-detection.c):")
print(f"const float MIN_VALS[] = {format_array_cpp(min_vals)};")
print(f"const float MAX_VALS[] = {format_array_cpp(max_vals)};")