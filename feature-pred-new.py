# 1. Import Libraries
import pandas as pd
import joblib
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
from google.colab import drive


drive.mount('/content/drive')

def load_data(file_path):
    """Load data from an Excel file and display the first 10 rows."""
    try:
        data = pd.read_excel(file_path, engine='openpyxl')
        print("Data loaded successfully.")
        display(data.head(10))
        return data
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

file_path = '<path>data/input_data.xlsx'  # Update this to the correct path
data = load_data(file_path)


def categorize_data(data):
    """Categorize data into good and bad based on predefined ranges."""
    # Define expected ranges
    ranges = {
        'Vibration Frequency': (1490, 1510),
        'Vibration Amplitude': (0.04, 0.06),
        'Bearing Temperature': (60, 80),
        'Motor Temperature': (80, 100),
        'Belt Load': (1.0, 1.4),
        'Torque': (280, 320),
        'Noise Levels': (55, 65),
        'Current and Voltage': (14, 16),
        'Hydraulic Pressure': (375, 385),
        'Belt Thickness': (1.5, 1.7),
        'Roller Condition': (65, 100),
    }

    # Convert relevant columns to numeric, coercing errors to NaN
    for feature in ranges.keys():
        data[feature] = pd.to_numeric(data[feature], errors='coerce')

    # Forward fill NaN values to avoid losing too much data
    data.fillna(method='ffill', inplace=True)

    status = []
    description = []

    for index, row in data.iterrows():
        is_good = True
        desc = []
        for feature, (min_val, max_val) in ranges.items():
            value = row[feature]
            if pd.isna(value) or value < min_val or value > max_val:
                is_good = False
                desc.append(f"{feature} out of range ({value}): {min_val} - {max_val}")
        
        status.append("Good" if is_good else "Bad")
        description.append("; ".join(desc) if desc else "All values within range.")

    data['Status'] = status
    data['Description'] = description

    # Check for any remaining NaN values
    print("NaN values after categorization:")
    print(data.isnull().sum())

    return data


# Process the data
processed_data = categorize_data(data)

# 4. Define Model
def build_model(input_shape):
    """Build a simple neural network model."""
    inputs = Input(shape=(input_shape,))
    x = Dense(128, activation='relu')(inputs)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(1, activation='linear')(x)  # Assuming a regression task
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# 5. Train Model
# Ensure target_columns is defined before training
target_columns = [
    'Vibration Frequency',
    'Vibration Amplitude',
    'Bearing Temperature',
    'Motor Temperature',
    'Belt Load',
    'Torque',
    'Noise Levels',
    'Current and Voltage',
    'Hydraulic Pressure',
    'Belt Thickness',
    'Roller Condition'
]

# 5. Train Model
def train_model(data, target_column):
    """Train a neural network model on the provided data for a specific target column."""
    # Drop unnecessary columns and prepare the feature set and target variable
    X = data.drop(columns=['Status', 'Description', target_column, 'Name', 'Timestamp'])
    y = data[target_column]

    # Handle NaN values in the target variable
    if y.isnull().any():
        print(f"Warning: Target column '{target_column}' contains NaN values.")
        y.ffill(inplace=True)  # Forward fill as an example; adjust as needed

    # Handle NaN values in the features
    if X.isnull().any().any():
        print("Warning: Feature set contains NaN values. Filling with forward fill.")
        X.ffill(inplace=True)  # Forward fill as an example; adjust as needed

    # Check if there are still any NaN values after filling
    if X.isnull().any().any() or y.isnull().any():
        raise ValueError("There are still NaN values in the data after filling.")

    # Scale the features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)  # Scale all features

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Build and train the model
    model = build_model(X_train.shape[1])  # Build model based on input shape

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

    # Save the model and scaler
    model.save(f'models/{target_column}_model.h5')
    joblib.dump(scaler, f'models/{target_column}_scaler.pkl')  # Save the scaler

    print(f"Model for {target_column} trained and saved successfully.")
    return model, scaler, history
# Train models for all target columns
models = {}
scalers = {}
histories = {}

for target in target_columns:
    print(f"Training model for {target}...")
    model, scaler, history = train_model(processed_data, target)
    models[target] = model
    scalers[target] = scaler
    histories[target] = history

# 6. Display Training History
def plot_history(history, target):
    """Plot the training and validation loss over epochs for each model."""
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'Training and Validation Loss for {target}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.show()

# Plot history for all trained models
for target in target_columns:
    print(f"Plotting training history for {target}...")
    plot_history(histories[target], target)

def predict_last_rows(models, data, scalers, n=10):
    """Predict the last n rows of the training data and display original vs. predicted values for specified targets in a structured table."""
    
    def requires_maintenance(value, target):
        """Determine if the given value requires maintenance based on predefined criteria."""
        maintenance_criteria = {
            'Vibration Frequency': (1490, 1510),
            'Vibration Amplitude': (0.04, 0.06),
            'Bearing Temperature': (60, 80),
            'Motor Temperature': (80, 100),
            'Belt Load': (1.0, 1.4),
            'Torque': (10, 50),
            'Noise Levels': (50, 70)  # Added Noise Levels criteria
        }
        
        if target in maintenance_criteria:
            min_val, max_val = maintenance_criteria[target]
            if not (min_val <= value <= max_val):
                return f"Out of range ({value}): {min_val} - {max_val}"
        
        return False  # Default to not requiring maintenance if criteria not defined

    # Get the last n rows from the original data
    last_n_rows = data.tail(n)
    results = []

    # Process Vibration Frequency
    target_freq = 'Vibration Frequency'
    model_freq = models[target_freq]
    scaler_freq = scalers[target_freq]

    # Prepare features for prediction
    X_last_freq = last_n_rows.drop(columns=['Status', 'Description', target_freq], errors='ignore')
    X_last_freq = X_last_freq.select_dtypes(include=[np.number])  # Select only numeric types

    # Fill NaN values if any
    X_last_freq.fillna(method='ffill', inplace=True)
    X_last_freq.fillna(method='bfill', inplace=True)  # Add this line


    # Scale features
    X_last_freq_scaled = scaler_freq.transform(X_last_freq)

    # Make predictions
    predictions_freq = model_freq.predict(X_last_freq_scaled)

    # Add original and predicted values for Vibration Frequency
    for i in range(n):
        original_value_freq = last_n_rows[target_freq].iloc[i]
        maintenance_message_freq = requires_maintenance(original_value_freq, target_freq)
        result_original_freq = {
            'Timestamp': last_n_rows['Timestamp'].iloc[i],
            'Name': last_n_rows['Name'].iloc[i],
            'Description': maintenance_message_freq if maintenance_message_freq else '-',
            'Type': 'Original',
            'Vibration Frequency': original_value_freq,
            'Vibration Amplitude': None,  # Placeholder for Vibration Amplitude
            'Bearing Temperature': None,  # Placeholder for Bearing Temperature
            'Motor Temperature': None,  # Placeholder for Motor Temperature
            'Belt Load': None,  # Placeholder for Belt Load
            'Torque': None,  # Placeholder for Torque
            'Noise Levels': None,  # Placeholder for Noise Levels
            'Status': 'Maintenance Required' if maintenance_message_freq else '-'
        }
        results.append(result_original_freq)

        # Predicted values
        predicted_value_freq = predictions_freq[i] if predictions_freq[i] is not None and not np.isnan(predictions_freq[i]) else None
        maintenance_message_predicted = requires_maintenance(predicted_value_freq, target_freq)
        result_predicted_freq = {
            'Timestamp': last_n_rows['Timestamp'].iloc[i],
            'Name': last_n_rows['Name'].iloc[i],
            'Description': maintenance_message_predicted if maintenance_message_predicted else '-',
            'Type': 'Predicted',
            'Vibration Frequency': predicted_value_freq,
            'Vibration Amplitude': None,  # Placeholder for Vibration Amplitude
            'Bearing Temperature': None,  # Placeholder for Bearing Temperature
            'Motor Temperature': None,  # Placeholder for Motor Temperature
            'Belt Load': None,  # Placeholder for Belt Load
            'Torque': None,  # Placeholder for Torque
            'Noise Levels': None,  # Placeholder for Noise Levels
            'Status': 'Maintenance Required' if maintenance_message_predicted else '-'
        }
        results.append(result_predicted_freq)

    # Process Vibration Amplitude
    target_amp = 'Vibration Amplitude'
    model_amp = models[target_amp]
    scaler_amp = scalers[target_amp]

    # Prepare features for prediction
    X_last_amp = last_n_rows.drop(columns=['Status', 'Description', target_amp], errors='ignore')
    X_last_amp = X_last_amp.select_dtypes(include=[np.number])  # Select only numeric types

    # Fill NaN values if any
    X_last_amp.ffill(inplace=True)  # Forward fill to handle NaNs

    # Scale features
    X_last_amp_scaled = scaler_amp.transform(X_last_amp)

    # Make predictions
    predictions_amp = model_amp.predict(X_last_amp_scaled)

    # Add original and predicted values for Vibration Amplitude
    for i in range(n):
        original_value_amp = last_n_rows[target_amp].iloc[i]
        results[2*i]['Vibration Amplitude'] = original_value_amp  # Fill original amplitude
        results[2*i+1]['Vibration Amplitude'] = predictions_amp[i][0]  # Fill predicted amplitude

    # Process Bearing Temperature
    target_temp = 'Bearing Temperature'
    model_temp = models[target_temp]
    scaler_temp = scalers[target_temp]

    # Prepare features for prediction
    X_last_temp = last_n_rows.drop(columns=['Status', 'Description', target_temp], errors='ignore')
    X_last_temp = X_last_temp.select_dtypes(include=[np.number])  # Select only numeric types

    # Fill NaN values if any
    X_last_temp.ffill(inplace=True)  # Forward fill to handle NaNs

    # Scale features
    X_last_temp_scaled = scaler_temp.transform(X_last_temp)

    # Make predictions
    predictions_temp = model_temp.predict(X_last_temp_scaled)

    # Add original and predicted values for Bearing Temperature
    for i in range(n):
        original_value_temp = last_n_rows[target_temp].iloc[i]
        results[2*i]['Bearing Temperature'] = original_value_temp  # Fill original bearing temperature
        results[2*i+1]['Bearing Temperature'] = predictions_temp[i][0]  # Fill predicted bearing temperature

    # Process Motor Temperature
    target_motor_temp = 'Motor Temperature'
    model_motor_temp = models[target_motor_temp]
    scaler_motor_temp = scalers[target_motor_temp]

    # Prepare features for prediction
    X_last_motor_temp = last_n_rows.drop(columns=['Status', 'Description', target_motor_temp], errors='ignore')
    X_last_motor_temp = X_last_motor_temp.select_dtypes(include=[np.number])  # Select only numeric types

    # Fill NaN values if any
    X_last_motor_temp.ffill(inplace=True)  # Forward fill to handle NaNs

    # Scale features
    X_last_motor_temp_scaled = scaler_motor_temp.transform(X_last_motor_temp)

    # Make predictions
    predictions_motor_temp = model_motor_temp.predict(X_last_motor_temp_scaled)

    # Add original and predicted values for Motor Temperature
    for i in range(n):
        original_value_motor_temp = last_n_rows[target_motor_temp].iloc[i]
        results[2*i]['Motor Temperature'] = original_value_motor_temp  # Fill original motor temperature
        results[2*i+1]['Motor Temperature'] = predictions_motor_temp[i][0]  # Fill predicted motor temperature

    # Process Noise Levels
    target_noise = 'Noise Levels'
    model_noise = models[target_noise]
    scaler_noise = scalers[target_noise]

    # Prepare features for prediction
    X_last_noise = last_n_rows.drop(columns=['Status', 'Description', target_noise], errors='ignore')
    X_last_noise = X_last_noise.select_dtypes(include=[np.number])  # Select only numeric types

    # Fill NaN values if any
    X_last_noise.ffill(inplace=True)  # Forward fill to handle NaNs

    # Scale features
    X_last_noise_scaled = scaler_noise.transform(X_last_noise)

    # Make predictions
    predictions_noise = model_noise.predict(X_last_noise_scaled)

    # Add original and predicted values for Noise Levels
    for i in range(n):
        original_value_noise = last_n_rows[target_noise].iloc[i]
        results[2*i]['Noise Levels'] = original_value_noise  # Fill original noise level
        results[2*i+1]['Noise Levels'] = predictions_noise[i][0]  # Fill predicted noise level

    # Process Current and Voltage
    target_current_voltage = 'Current and Voltage'
    model_current_voltage = models[target_current_voltage]
    scaler_current_voltage = scalers[target_current_voltage]

    # Prepare features for prediction
    X_last_current_voltage = last_n_rows.drop(columns=['Status', 'Description', target_current_voltage], errors='ignore')
    X_last_current_voltage = X_last_current_voltage.select_dtypes(include=[np.number])  # Select only numeric types

    # Fill NaN values if any
    X_last_current_voltage.ffill(inplace=True)  # Forward fill to handle NaNs

    # Scale features
    X_last_current_voltage_scaled = scaler_current_voltage.transform(X_last_current_voltage)

    # Make predictions
    predictions_current_voltage = model_current_voltage.predict(X_last_current_voltage_scaled)

    # Add original and predicted values for Current and Voltage
    for i in range(n):
        original_value_current_voltage = last_n_rows[target_current_voltage].iloc[i]
        results[2*i]['Current and Voltage'] = original_value_current_voltage  # Fill original current and voltage
        results[2*i+1]['Current and Voltage'] = predictions_current_voltage[i][0]  # Fill predicted current and voltage

    # Process Hydraulic Pressure
    target_hydraulic_pressure = 'Hydraulic Pressure'
    model_hydraulic_pressure = models[target_hydraulic_pressure]
    scaler_hydraulic_pressure = scalers[target_hydraulic_pressure]

    # Prepare features for prediction
    X_last_hydraulic_pressure = last_n_rows.drop(columns=['Status', 'Description', target_hydraulic_pressure], errors='ignore')
    X_last_hydraulic_pressure = X_last_hydraulic_pressure.select_dtypes(include=[np.number])  # Select only numeric types

    # Fill NaN values if any
    X_last_hydraulic_pressure.ffill(inplace=True)  # Forward fill to handle NaNs

    # Scale features
    X_last_hydraulic_pressure_scaled = scaler_hydraulic_pressure.transform(X_last_hydraulic_pressure)

    # Make predictions
    predictions_hydraulic_pressure = model_hydraulic_pressure.predict(X_last_hydraulic_pressure_scaled)

    # Add original and predicted values for Hydraulic Pressure
    for i in range(n):
        original_value_hydraulic_pressure = last_n_rows[target_hydraulic_pressure].iloc[i]
        results[2*i]['Hydraulic Pressure'] = original_value_hydraulic_pressure  # Fill original hydraulic pressure
        results[2*i+1]['Hydraulic Pressure'] = predictions_hydraulic_pressure[i][0]  # Fill predicted hydraulic pressure

    # Process Belt Thickness
    target_belt_thickness = 'Belt Thickness'
    model_belt_thickness = models[target_belt_thickness]
    scaler_belt_thickness = scalers[target_belt_thickness]

    # Prepare features for prediction
    X_last_belt_thickness = last_n_rows.drop(columns=['Status', 'Description', target_belt_thickness], errors='ignore')
    X_last_belt_thickness = X_last_belt_thickness.select_dtypes(include=[np.number])  # Select only numeric types

    # Fill NaN values if any
    X_last_belt_thickness.ffill(inplace=True)  # Forward fill to handle NaNs

    # Scale features
    X_last_belt_thickness_scaled = scaler_belt_thickness.transform(X_last_belt_thickness)

    # Make predictions
    predictions_belt_thickness = model_belt_thickness.predict(X_last_belt_thickness_scaled)

    # Add original and predicted values for Belt Thickness
    for i in range(n):
        original_value_belt_thickness = last_n_rows[target_belt_thickness].iloc[i]
        results[2*i]['Belt Thickness'] = original_value_belt_thickness  # Fill original belt thickness
        results[2*i+1]['Belt Thickness'] = predictions_belt_thickness[i][0]  # Fill predicted belt thickness

    # Process Roller Condition
    target_roller_condition = 'Roller Condition'
    model_roller_condition = models[target_roller_condition]
    scaler_roller_condition = scalers[target_roller_condition]

    # Prepare features for prediction
    X_last_roller_condition = last_n_rows.drop(columns=['Status', 'Description', target_roller_condition], errors='ignore')
    X_last_roller_condition = X_last_roller_condition.select_dtypes(include=[np.number])  # Select only numeric types

    # Fill NaN values if any
    X_last_roller_condition.ffill(inplace=True)  # Forward fill to handle NaNs

    # Scale features
    X_last_roller_condition_scaled = scaler_roller_condition.transform(X_last_roller_condition)

    # Make predictions
    predictions_roller_condition = model_roller_condition.predict(X_last_roller_condition_scaled)

    # Add original and predicted values for Roller Condition
    for i in range(n):
        original_value_roller_condition = last_n_rows[target_roller_condition].iloc[i]
        results[2*i]['Roller Condition'] = original_value_roller_condition  # Fill original roller condition
        results[2*i+1]['Roller Condition'] = predictions_roller_condition[i][0]  # Fill predicted roller condition

    # Create a DataFrame from the results
    results_df = pd.DataFrame(results)

    # Ensure the output shows only the last 20 rows (10 original and 10 predicted)
    results_df = results_df.tail(n * 2)

    # Ensure the columns are in the correct order, including all relevant columns
    column_order = ['Type', 'Timestamp', 'Name', 'Status', 'Description', 
                    'Vibration Frequency', 'Vibration Amplitude', 
                    'Bearing Temperature', 'Motor Temperature', 
                    'Belt Load', 'Torque', 'Noise Levels',
                    'Current and Voltage', 'Hydraulic Pressure', 
                    'Belt Thickness', 'Roller Condition']  # Include all columns
    results_df = results_df[column_order]

    # Display the final DataFrame
    display(results_df)

# Call the function to predict and display results for the last 10 rows
predict_last_rows(models, processed_data, scalers, n=10)
