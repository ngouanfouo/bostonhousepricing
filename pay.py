import pickle
from flask import Flask, request, jsonify, render_template, url_for
import numpy as np
import pandas as pd


# Load the pickled model (assuming regmodel.pkl is in the same directory)
def load_model():
    global regmodel
    try:
        with open('regmodel.pkl', 'rb') as f:
            regmodel = pickle.load(f)
        print("Model loaded successfully!")
    except FileNotFoundError:
        print("Model file 'regmodel.pkl' not found. Please ensure it exists in the same directory.")
        raise  # Re-raise the exception to halt execution

# Load the model at application startup
load_model()


app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')  # Assuming a home.html template exists


@app.route('/predict_api', methods=['POST'])
def predict_api():
    try:
        # Get the data from the request
        data = request.get_json(force=True)  # Ensure JSON data is parsed correctly

        # Convert data to a NumPy array and reshape
        new_data = np.array(list(data.values())).reshape(1, -1)

        # Handle potential errors during data conversion or prediction
        try:
            # Check if the scaler object is defined
            if not hasattr(app, 'scaler'):
                print("Scaler object not found. Please ensure it's defined or data doesn't require scaling.")
                return jsonify({'error': 'Scaler object not found'})

            # Perform scaling if a scaler is available
            new_data = app.scaler.transform(new_data)

            # Make prediction using the loaded model
            output = regmodel.predict(new_data)
            return jsonify({'prediction': output[0]})

        except Exception as e:
            print(f"Error during prediction: {e}")
            return jsonify({'error': 'An error occurred during prediction'})

    except Exception as e:
        print(f"Error in request processing: {e}")
        return jsonify({'error': 'An error occurred while processing the request'})

if __name__ == "__main__":
    app.run(debug=True)
