from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib

app = Flask(__name__)

# Load the trained Random Forest model
model = joblib.load('random_forest_model.pkl')

@app.route('/')
def index():
    return render_template('index.html')  # Ensure index.html is in the 'templates' folder

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the request (expecting JSON data)
        data = request.get_json()

        # Extract the feature values
        item_weight = data['Item_Weight']
        item_fat_content = data['Item_Fat_Content']
        item_visibility = data['Item_Visibility']
        item_mrp = data['Item_MRP']
        outlet_size = data['Outlet_Size']
        outlet_location_type = data['Outlet_Location_Type']
        outlet_type = data['Outlet_Type']
        outlet_age = data['Outlet_Age']
        item_type_encoded = data['Item_Type_Encoded']
        item_identifier_categories_encoded = data['Item_Identifier_Categories_Encoded']

        # Convert the input into a numpy array (make sure to match the order of your features)
        input_data = np.array([[item_weight, item_fat_content, item_visibility, item_mrp,
                                outlet_size, outlet_location_type, outlet_type,
                                outlet_age, item_type_encoded, item_identifier_categories_encoded]])

        # Make prediction
        prediction = model.predict(input_data)

        # Return the prediction result as JSON
        return jsonify({
            'prediction': prediction[0]
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e)
        })

if __name__ == '__main__':
    app.run(debug=True)
