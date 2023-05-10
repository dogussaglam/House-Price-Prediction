import json
import pickle

from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd



app=Flask(__name__, static_url_path='/static')
## Load the model
regmodel=pickle.load(open('regmodel.pkl','rb'))
scaler = pickle.load(open('scaling.pkl','rb'))



@app.route('/')
def home():
    return render_template('home.html')

def check_op(x):
    if x == '1H OCEAN':
        return [1,0,0,0,0]
    elif x == 'INLAND':
        return [0,1,0,0,0]
    elif x == 'ISLAND':
        return [0,0,1,0,0]
    elif x == 'NEAR BAY':
        return [0,0,0,1,0]
    elif x == 'NEAR OCEAN':
        return [0,0,0,0,1]
    else:
        return 0

@app.route('/predict_api', methods=['POST'])
def predict_api():
    # Extract the input features from the request.form.get data
    data = request.json['data']
    longitude = float(data['Longitude'])
    print(longitude)
    latitude = float(data['Latitude'])
    print(latitude)
    housing_median_age = float(data['Housing Median Age'])
    print(housing_median_age)
    total_rooms = float(data['Total Rooms'])
    print(total_rooms)
    total_bedrooms = float(data['Total Bedrooms'])
    print(total_bedrooms)
    population = float(data['Population'])
    print(population)
    households = float(data['Households'])
    print(households)
    median_income = float(data['Median Income'])
    print(median_income)
    ocean_proximity = data['Ocean Proximity']
    print(ocean_proximity)


    bedroom_ratio = total_bedrooms / total_rooms
    print(bedroom_ratio)
    households_rooms = total_rooms / households
    print(households_rooms)

    # Concatenate the input features
    X = np.concatenate(([[longitude, latitude, housing_median_age, total_rooms,
                           total_bedrooms, population, households, median_income]],
                         [check_op(ocean_proximity)],
                         [[bedroom_ratio, households_rooms]]), axis=1)

    # Preprocess the input features (e.g., scale them)
    scaled_X= scaler.transform(X)

    # Make a prediction using the loaded model
    y_pred = regmodel.predict(scaled_X)

    # Return the prediction as a JSON response
    return jsonify(int(y_pred[0]))




@app.route('/predict', methods=['POST'])
def predict():
    # Extract the input features from the request.form.get data
    longitude = float(request.form.get('longitude'))
    latitude = float(request.form.get('latitude'))
    housing_median_age = float(request.form.get('housing_median_age'))
    total_rooms = float(request.form.get('total_rooms'))
    total_bedrooms = float(request.form.get('total_bedrooms'))
    population = float(request.form.get('population'))
    households = float(request.form.get('households'))
    median_income = float(request.form.get('median_income'))
    ocean_proximity = request.form.get('ocean_proximity')

    bedroom_ratio = total_bedrooms / total_rooms
    households_rooms = total_rooms / households


    # Concatenate the input features
    X = np.concatenate(([[longitude, latitude, housing_median_age, total_rooms,
                           total_bedrooms, population, households, median_income]],
                         [check_op(ocean_proximity)],
                         [[bedroom_ratio, households_rooms]]), axis=1)
    
    # Preprocess the input features (e.g., scale them)
    scaled_X = scaler.transform(X)

    # Make a prediction using the loaded model
    y_pred = regmodel.predict(scaled_X)
    print('Predicted house price: ${:,.2f}'.format(y_pred[0]))
    # Return the prediction as a JSON response
    return render_template('home.html', prediction_text='Predicted house price: ${:,.2f}'.format(y_pred[0]))


if __name__=="__main__":
    app.run(debug=True)