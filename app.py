import json
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel,EmailStr,AnyUrl,Field,computed_field,ConfigDict
from typing import List,Dict,Optional,Annotated,Literal
import pickle
import pandas as pd


with open('model.pkl','rb') as f:
    model = pickle.load(f)

app=FastAPI()

tier_1_cities = [
    "Mumbai",
    "Delhi",
    "Bengaluru",
    "Chennai",
    "Kolkata",

    "Hyderabad",
    "Pune",
    "Ahmedabad"
]

tier_2_cities = [
    "Jaipur",
    "Lucknow",
    "Indore",
    "Bhopal",
    "Chandigarh",
    "Coimbatore",
    "Kochi",
    "Trivandrum",
    "Nagpur",
    "Vadodara"
]


class UserInput(BaseModel):
   

    age: Annotated[int, Field(..., gt=0, lt=120)]
    weight: Annotated[float, Field(..., gt=0)]
    height: Annotated[float, Field(..., gt=0)]
    income_lpa: Annotated[float, Field(..., gt=0)]
    smoker: Annotated[bool, Field(..., example=False)]
    city: Annotated[str, Field(...)]
    occupation: Annotated[Literal['retired','freelancer','student','government_job','business_owner','unemployed','private_job'],Field(...,)
    ]

# Computed field to find the bmi of the user
    @computed_field(return_type=float, repr=False)
    @property
    def bmi(self):
        return round(self.weight / (self.height ** 2), 2)


# Computed field to find the lifestyle risk of the user
    @computed_field(return_type=str, repr=False)
    @property
    def lifestyle_risk(self):
        if self.smoker and self.bmi > 30:
            return 'high'
        elif self.smoker and self.bmi > 45:
            return 'medium'
        return 'low'


# Computed field to find the age group of the user
    @computed_field(return_type=str, repr=False)
    @property
    def age_group(self):
        if self.age < 25:
            return 'young'
        elif self.age < 45:
            return 'adult'
        elif self.age < 60:
            return 'middle_aged'
        return 'senior'


# Computed field to find the City tier of the user
    @computed_field(return_type=int, repr=False)
    @property
    def city_tier(self):
        if self.city in tier_1_cities:
            return 1
        elif self.city in tier_2_cities:
            return 2
        return 3
       


def load_data():
    with open('patients.json','r') as f:
        data = json.load(f)
        
    return data



@app.post('/predict')
def predict_premium(data : UserInput):
    
    input_df = pd.DataFrame([{
        'bmi':data.bmi,
        'age_group':data.age_group,
        'lifestyle_risk':data.lifestyle_risk,
        'city_tier':data.city_tier,
        'income_lpa':data.income_lpa,
        'occupation':data.occupation,
    }])
    
    prediction = model.predict(input_df)[0]
   
    return JSONResponse(status_code = 200,content={'predicted_category':prediction})


