#importing libraries
import pickle
import pandas as pd
import streamlit as slt
import numpy as np

# page setting
slt.set_page_config(layout="wide")
slt.header(':blue[Cardheko-Price Prediction 🚗]')

# Load data
df = pd.read_csv("final_df.csv")
print(df.columns)

# Streamlit interface
col1, col2 = slt.columns(2)
with col1:
    Ft = slt.selectbox("Fuel type", ['Petrol', 'Diesel', 'Lpg', 'Cng', 'Electric'])

    Bt = slt.selectbox("Body type", ['Hatchback', 'SUV', 'Sedan', 'MUV', 'Coupe', 'Minivans',
                                     'Convertibles', 'Hybrids', 'Wagon', 'Pickup Trucks'])
    Tr = slt.selectbox("Transmission", ['Manual', 'Automatic'])

    Owner = slt.selectbox("Owner", [0, 1, 2, 3, 4, 5])

    Brand = slt.selectbox("Brand", options=df['Brand'].unique())

    filtered_models = df[(df['Brand'] == Brand) & (df['body type'] == Bt) & (df['Fuel type'] == Ft)]['model'].unique()


    Model=slt.selectbox("Model",options=filtered_models)

    Model_year = slt.selectbox("Model Year", options=sorted(df['modelYear'].unique()))
    
    
    IV = slt.selectbox("Insurance Validity", ['Third Party insurance', 'Comprehensive', 'Third Party',
                                              'Zero Dep', '2', '1', 'Not Available'])
    
    Km = slt.slider("Kilometers Driven", min_value=100, max_value=100000, step=1000)

    ML = slt.number_input("Mileage", min_value=5, max_value=50, step=1)  

    seats = slt.selectbox("Seats", options=sorted(df['Seats'].unique()))
    
    color = slt.selectbox("Color", df['Color'].unique())

    city = slt.selectbox("City", options=df['City'].unique())

with col2:
    Submit = slt.button("Predict")

    if Submit:
    
       # load the model,scaler and encoder
       with open('pipeline.pkl','rb') as files:
        pipeline=pickle.load(files)

        # input data
        new_df=pd.DataFrame({
        'Fuel type': Ft,
        'body type':Bt,
        'transmission':Tr,
        'ownerNo':Owner,
        'Brand':Brand,
        "model":Model,
        'modelYear':Model_year,
        'Insurance Validity':IV,
        'Kms Driven':Km,
        'Mileage':ML,
        'Seats':seats,
        'Color':color,
        'City': city},index=[0])
        
        # Display the selected details
        data = [Ft, Bt, Tr, Owner,Brand, Model,Model_year, IV, Km, ML, seats, color, city]

        slt.write(data)

        # FINAL MODEL PREDICTION 
        prediction=pipeline.predict(new_df)
        slt.write(f"The price of the {new_df['Brand'].iloc[0]} car is: {round(prediction[0],2)} lakhs")


