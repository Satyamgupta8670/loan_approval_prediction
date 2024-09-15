
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

model = joblib.load('model.joblib')

class LoanApplication(BaseModel):
    Gender: int
    Married: int
    Dependents: int
    Education: int
    Self_Employed: int
    ApplicantIncome: float
    CoapplicantIncome: float
    LoanAmount: float
    Loan_Amount_Term: float
    Credit_History: int
    Property_Area: str

@app.post("/predict")
async def predict(loan_application: LoanApplication):
    # Convert the input data to a Pandas DataFrame
    df = pd.DataFrame([loan_application.dict()])

    # One-hot encode the Property_Area column
    df = pd.get_dummies(data=df, columns=["Property_Area"])

    # Make a prediction using the trained model
    prediction = model.predict(df)

    # Return the predicted loan status as a JSON response
    return {"predicted_loan_status": prediction[0]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
