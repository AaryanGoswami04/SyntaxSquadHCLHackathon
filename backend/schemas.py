from pydantic import BaseModel, Field
from typing import Literal


class CustomerFeatures(BaseModel):
    recency: int = Field(..., ge=0, description="Days since last purchase")
    frequency: int = Field(..., ge=1, description="Count of unique past orders")
    monetary: float = Field(..., ge=0.0, description="Total past spend in USD")
    is_uk: int = Field(0, ge=0, le=1, description="1 if customer is in UK, else 0")
    horizon: Literal[30, 60, 90] = Field(
        30, description="Prediction window in days (30 / 60 / 90)"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "recency": 14,
                "frequency": 5,
                "monetary": 320.50,
                "is_uk": 1,
                "horizon": 30,
            }
        }
    }


class PredictionResponse(BaseModel):
    predicted_spend: float
    horizon_days: int
    label: str           # e.g. "Predicted 30-Day Spend"