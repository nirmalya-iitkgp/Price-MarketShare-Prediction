import pandas as pd
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_squared_error

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('path_to_your_model.h5')

# Load and preprocess data (use the same preprocessing as in your training script)
def preprocess_data(price, company, dosage, tabs, launch_year):
    # Example of creating a DataFrame with the input data
    data = pd.DataFrame({
        'Price': [price],
        'Company': [company],
        'Dosage_mg': [dosage],
        'Tabs': [tabs],
        'Launch_Year': [launch_year]
    })

    # Encode categorical variables
    le_company = LabelEncoder()
    data['Company'] = le_company.fit_transform(data['Company'])

    # Normalize numerical features
    scaler = MinMaxScaler()
    data[['Price']] = scaler.fit_transform(data[['Price']])
    
    return data

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract features from form
    price = float(request.form['price'])
    company = request.form['company']
    dosage = float(request.form['dosage'])
    tabs = int(request.form['tabs'])
    launch_year = int(request.form['launch_year'])

    # Preprocess the data
    data = preprocess_data(price, company, dosage, tabs, launch_year)

    # Assuming the model expects sequences and static features
    X_seq = np.array([data['Price'].values]).reshape((1, -1, 1))
    X_static = data[['Company', 'Dosage_mg', 'Tabs', 'Launch_Year']].values
    sku_id = 0  # Dummy SKU_ID for embedding input

    # Predict market share
    y_pred_ratio = model.predict([X_seq, np.array([sku_id]), X_static])
    y_pred = y_pred_ratio[0][0]  # Assuming the model's output needs to be processed further

    # Calculate key metrics (use dummy actual values for demonstration)
    y_test = np.array([0.25])  # Replace with actual values
    r2 = r2_score(y_test, y_pred_ratio)
    mse = mean_squared_error(y_test, y_pred_ratio)

    return render_template('index.html', prediction_text=f'Predicted Market Share: {y_pred}',
                           r2_score=f'R-squared: {r2}', mse=f'Mean Squared Error: {mse}')

if __name__ == "__main__":
    app.run(debug=True)
