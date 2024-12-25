import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dense, Concatenate
import pickle

# Load Month-Wise Data
file_path = '/content/input data file.xlsx'  # Replace with your file path
data = pd.read_excel(file_path)

# Rename columns to match expected variables
data = data.rename(columns={
    'Date': 'Date',
    'SKU_Id': 'SKU_ID',
    'Price': 'Price',
    'Company': 'Company',
    'Dosage': 'Dosage_mg',
    'Tabs': 'Tabs',
    'Launch_year': 'Launch_Year',
    'Curr_mkt_share': 'Market_Share'
})

# Clean and preprocess the data
## Convert Date to a consistent datetime format
data['Date'] = pd.to_datetime(data['Date'], format='%b\'%y', errors='coerce')

# Encode categorical variables
le_company = LabelEncoder()
data['Company'] = le_company.fit_transform(data['Company'])

# Normalize numerical features
scaler = MinMaxScaler()
data[['Price', 'Market_Share']] = scaler.fit_transform(data[['Price', 'Market_Share']])

# Normalize market share within each date group
for date in data['Date'].unique():
    group_data = data[data['Date'] == date]
    group_data['Market_Share'] /= group_data['Market_Share'].sum()

# Define a function to generate sequences with SKU embeddings
def create_sequences(data, n_steps):
    sequences, static_features, targets, sku_ids = [], [], [], []
    for sku_id in data['SKU_ID'].unique():
        sku_data = data[data['SKU_ID'] == sku_id].reset_index(drop=True)
        for i in range(len(sku_data) - n_steps):
            # Time-Series Features: Price
            seq = sku_data.iloc[i:i + n_steps][['Price']].values
            # Static Features: Company, Dosage_mg, Tabs, Launch_Year
            static = sku_data.iloc[i][['Company', 'Dosage_mg', 'Tabs', 'Launch_Year']].values
            # Target: Market_Share at Time t + n_steps
            target = sku_data.iloc[i + n_steps]['Market_Share']
            sequences.append(seq)
            static_features.append(static)
            targets.append(target)
            sku_ids.append(sku_id)
    return np.array(sequences), np.array(static_features), np.array(targets), np.array(sku_ids)

# Parameters
n_steps = 12  # Sequence length (e.g., 12 months)
n_sku = len(data['SKU_ID'].unique())  # Number of unique SKUs
embedding_dim = 8  # Dimensionality of SKU embeddings

# Generate sequences
X_seq, X_static, y, sku_ids = create_sequences(data, n_steps)

# Convert arrays to float32 for TensorFlow compatibility
X_seq = X_seq.astype('float32')
X_static = X_static.astype('float32')
y = y.astype('float32')

# Impute missing values if necessary
imputer = SimpleImputer(strategy='mean')
X_seq = imputer.fit_transform(X_seq.reshape(-1, 1)).reshape(X_seq.shape)
X_static = imputer.fit_transform(X_static)

# Encode SKU_ID using LabelEncoder instead of OneHotEncoder
sku_encoder = LabelEncoder()  # Change here
sku_indices = sku_encoder.fit_transform(sku_ids) # Change here
n_categories = len(sku_encoder.classes_) # Change here

# Keep track of original indices before splitting
original_indices = np.arange(len(sku_ids))  # Use length of sku_ids

# Split the data into training and testing sets
X_seq_train, X_seq_test, X_static_train, X_static_test, y_train, y_test, sku_indices_train, sku_indices_test = train_test_split(
    X_seq, X_static, y, sku_indices, test_size=0.2, random_state=42 # Changed here
)

def create_model(n_embedding_dim):
  sequence_input = tf.keras.Input(shape=(n_steps, 1), name="seq_input")
  x = tf.keras.layers.LSTM(64, return_sequences=False)(sequence_input)

  sku_input = tf.keras.Input(shape=(1,), name="sku_input")
  # Use n_embedding_dim for output_dim in Embedding
  sku_embedding = tf.keras.layers.Embedding(input_dim=n_categories, output_dim=n_embedding_dim)(sku_input)  # change to n_categories
  sku_embedding_reshaped = tf.keras.layers.Reshape((n_embedding_dim,))(sku_embedding)


  static_input = tf.keras.Input(shape=(X_static.shape[1],), name="static_input")
  y = tf.keras.layers.Dense(32, activation='relu')(static_input)

  merged = tf.keras.layers.Concatenate()([x, sku_embedding_reshaped, y])
  z = tf.keras.layers.Dense(64, activation='relu')(merged)
  output = tf.keras.layers.Dense(1, activation='sigmoid', name="output")(z)

  model = tf.keras.Model(inputs=[sequence_input, sku_input, static_input], outputs=output)
  model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['mae'])
  return model

model = create_model(10)

# Train Model
history = model.fit(
    [X_seq_train, sku_indices_train, X_static_train], y_train,
    validation_split=0.2,
    epochs=10,
    batch_size=32,
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)]
)

model.save('market_share_model.h5') # Save the model
with open('sku_encoder.pkl', 'wb') as f: # Save the SKU encoder
    pickle.dump(sku_encoder, f)
with open('price_scaler.pkl', 'wb') as f: # Save the price scaler
    pickle.dump(scaler, f)

# Evaluate Model
loss, mae = model.evaluate([X_seq_test, sku_indices_test, X_static_test], y_test)
print(f"Test Loss: {loss}, Test MAE: {mae}")

# Predict Market Share Ratio
y_pred_ratio = model.predict([X_seq_test, sku_indices_test, X_static_test])


# Calculate Actual Market Share from Ratio
def calculate_actual_market_share(y_pred_ratio, y_test, data):
  actual_market_shares = []
  for i, ratio in enumerate(y_pred_ratio):
    date = data.iloc[i]['Date']
    total_market_share = data[data['Date'] == date]['Market_Share'].sum()
    # Normalize predicted ratio by total market share
    #actual_market_share = ratio / y_pred_ratio.sum() * total_market_share
    actual_market_share = ratio * total_market_share
    actual_market_shares.append(actual_market_share)
  return actual_market_shares

y_pred = calculate_actual_market_share(y_pred_ratio, y_test, data)

# Create a DataFrame for Actual vs Predicted Market Share
output_data = pd.DataFrame({
    'Actual Market Share': y_test.flatten(),
    'Predicted Market Share': y_pred
})

# Display the first 10 results
print(output_data.head(10))

# Save results to a CSV file
output_data.to_csv('predicted_market_share.csv', index=False)

import matplotlib.pyplot as plt

# Assuming 'output_data' DataFrame from your code
plt.figure(figsize=(10, 6))
plt.plot(output_data['Actual Market Share'], label='Actual')
plt.plot(output_data['Predicted Market Share'], label='Predicted')
plt.title('Actual vs. Predicted Market Share')
plt.xlabel('Data Point')
plt.ylabel('Market Share')
plt.legend()
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(output_data['Actual Market Share'], output_data['Predicted Market Share'])
plt.title('Actual vs. Predicted Market Share (Scatter Plot)')
plt.xlabel('Actual Market Share')
plt.ylabel('Predicted Market Share')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')  # Ideal prediction line
plt.show()

errors = output_data['Actual Market Share'] - output_data['Predicted Market Share']
plt.figure(figsize=(8, 6))
plt.hist(errors, bins=20)
plt.title('Distribution of Prediction Errors')
plt.xlabel('Prediction Error')
plt.ylabel('Frequency')
plt.show()


# Assuming 'output_data' has a 'Date' column
output_data['Date'] = data['Date'][:len(output_data)]  # Align dates

plt.figure(figsize=(12, 6))
plt.plot(output_data['Date'], output_data['Actual Market Share'], label='Actual')
plt.plot(output_data['Date'], output_data['Predicted Market Share'], label='Predicted')
plt.title('Actual vs. Predicted Market Share Over Time')
plt.xlabel('Date')
plt.ylabel('Market Share')
plt.legend()
plt.show()

from sklearn.metrics import r2_score, mean_squared_error
r2 = r2_score(y_test, y_pred)  # Assuming y_test and y_pred are your actual and predicted values
print(f"R-squared: {r2}")
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
