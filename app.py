from fastapi import FastAPI, HTTPException, Request
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from pathlib import Path
import pandas as pd
import joblib

app = FastAPI(title="Graduate Salary Prediction API")

BASE_DIR = Path(__file__).resolve().parent

templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

model = joblib.load(BASE_DIR / "rf_model.pkl")
scaler = joblib.load(BASE_DIR / "scaler.pkl")
feature_columns = joblib.load(BASE_DIR / "feature_columns.pkl")

print("All models loaded successfully!")

columns = [
    "10percentage", "12graduation", "12percentage",
    "10board", "12board", "CollegeTier", "collegeGPA",
    "CollegeCityTier", "GraduationYear", "English",
    "Logical", "Quant", "Domain", "ComputerProgramming",
    "ElectronicsAndSemicon", "ComputerScience",
    "MechanicalEngg", "ElectricalEngg", "TelecomEngg",
    "CivilEngg", "conscientiousness", "agreeableness",
    "extraversion", "nueroticism", "openess_to_experience",
    "Gender", "Degree", "Specialization", "CollegeState"
]

categorical = [
    "10board", "12board", "Gender",
    "Degree", "Specialization", "CollegeState"
]


class Student(BaseModel):
    features: dict


@app.get("/")
def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={}
    )


@app.get("/health")
def health():
    return {"status": "healthy", "model": "Random Forest"}


@app.post("/predict")
def predict(student: Student):
    try:
        data = pd.DataFrame([student.features])

        missing = set(columns) - set(data.columns)

        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"Missing features: {list(missing)}"
            )

        data = data[columns]

        for col in categorical:
            data[col] = data[col].astype(str).str.strip()

        numeric = data.select_dtypes(include="number").columns

        data[numeric] = data[numeric].replace(-1, float("nan"))

        if data[numeric].isnull().any().any():
            raise HTTPException(
                status_code=400,
                detail="Numeric features cannot be missing or -1"
            )

        data = pd.get_dummies(data, drop_first=True)

        data = data.reindex(
            columns=feature_columns,
            fill_value=0
        )

        scaled_data = scaler.transform(data)

        prediction = model.predict(scaled_data)[0]

        return {
            "predicted_salary": round(float(prediction), 2),
            "currency": "INR",
            "model": "Random Forest"
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))