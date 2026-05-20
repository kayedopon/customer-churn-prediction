from enum import Enum
from pydantic import BaseModel


class Gender(str, Enum):
    male = "Male"
    female = "Female"


class Contract(str, Enum):
    month_to_month = "Month-to-month"
    one_year = "One year"
    two_year = "Two year"


class PaymentMethod(str, Enum):
    electronic_check = "Electronic check"
    mailed_check = "Mailed check"
    bank_transfer = "Bank transfer"
    credit_card = "Credit card"


class CustomerData(BaseModel):
    gender: Gender
    senior_citizen: int
    tenure: int
    monthly_charges: float
    total_charges: float
    contract: Contract
    payment_method: PaymentMethod


class PredictionResponse(BaseModel):
    churn_probability: float
    prediction: int
