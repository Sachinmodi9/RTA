import streamlit as st
import numpy as np
import pandas as pd
import joblib
import pickle

st.set_page_config(page_title="Accident Severity Prediction App",
                   page_icon="🚧", layout="wide")

#creating option list for dropdown menu
options_day = ['Sunday', "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
options_number_vehicles_involved = [1,2,3,4,5,6,7]
options_hours = [i for i in range(0,25)]
options_lanes = ['Two-way (divided with broken lines road marking)', 'Undivided Two way',
       'other', 'Double carriageway (median)', 'One way',
       'Two-way (divided with solid lines road marking)', 'Unknown']
options_type_of_vehicle = ['Automobile','Public (> 45 seats)','Lorry (41?100Q)','Public (13?45 seats)','Lorry (11?40Q)','Long lorry','Public (12 seats)','Taxi','Pick up upto 10Q','Stationwagen','Ridden horse','Other','Bajaj','Turbo','Motorcycle','Special vehicle','Bicycle']
options_junction_type = ['Y Shape','No junction','Crossing','Unknown','O Shape','T Shape','X Shape']
options_cause = ['No distancing', 'Changing lane to the right',
       'Changing lane to the left', 'Driving carelessly',
       'No priority to vehicle', 'Moving Backward',
       'No priority to pedestrian', 'Other', 'Overtaking',
       'Driving under the influence of drugs', 'Driving to the left',
       'Getting off the vehicle improperly', 'Driving at high speed',
       'Overturning', 'Turnover', 'Overspeed', 'Overloading', 'Drunk driving',
       'Unknown', 'Improper parking']


st.markdown("<h1 style='text-align: center;'>Accident Severity Prediction App 🚧</h1>", unsafe_allow_html=True)

def main():
    with st.form('prediction_form'):

       st.subheader("Enter the input for following features:")
       day_of_week = st.selectbox("Select Day of the Week: ", options=options_day)
       hour = st.slider("Pickup Hour: ", 0, 23, value=0, format="%d")
       lanes = st.selectbox("Select Lanes: ", options=options_lanes)
       junction = st.selectbox("Select Junction: ", options=options_junction_type )
       vehiclesType = st.selectbox("Select vehicles type: ",options=options_type_of_vehicle )
       accident_cause = st.selectbox("Accident Cause: ", options=options_cause)
       vehicles_involved = st.slider("Vehicles Involved: ", 1, 7, value=1, format="%d")
      
       submit = st.form_submit_button("Predict")

       if submit:
              prediction = Predict(vehicles_involved,hour,day_of_week,vehiclesType,lanes,junction,accident_cause)
              st.write("Prediction:", prediction)


def Predict(vehicles_involved,hour,day_of_week,vehiclesType,lanes,junction,accident_cause):
       
       model = joblib.load('model/pipeline_v1.pkl')

       input_dict = {
       'Number_of_vehicles_involved':[vehicles_involved],
       'hour':[hour], 
       'Day_of_week':[day_of_week], 
       'Type_of_vehicle':[vehiclesType],
       'Lanes_or_Medians':[lanes], 
       'Types_of_Junction':[junction], 
       'Cause_of_accident':[accident_cause]
    }

    # Create a DataFrame from the dictionary
       input_df = pd.DataFrame(input_dict)

       prediction = model.predict(input_df)
       
       return prediction

if __name__ == '__main__':
    main()