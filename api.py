"""FastAPI service: POST /predict -> will this visitor buy?

Run locally:   uvicorn api:app --reload
Docs (Swagger): http://127.0.0.1:8000/docs
"""
from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel, Field

import predictor

app = FastAPI(
    title="Online Shoppers Purchase Intention API",
    description="Predicts whether an online store visitor will make a purchase.",
    version="1.0.0",
)


class Visitor(BaseModel):
    Administrative: int = Field(0, ge=0, description="Number of account/administrative pages visited")
    Administrative_Duration: float = Field(0, ge=0, description="Seconds spent on administrative pages")
    Informational: int = Field(0, ge=0, description="Number of informational pages visited")
    Informational_Duration: float = Field(0, ge=0, description="Seconds spent on informational pages")
    ProductRelated: int = Field(1, ge=0, description="Number of product pages visited")
    ProductRelated_Duration: float = Field(0, ge=0, description="Seconds spent on product pages")
    BounceRates: float = Field(0, ge=0, le=1, description="Average bounce rate")
    ExitRates: float = Field(0, ge=0, le=1, description="Average exit rate")
    PageValues: float = Field(0, ge=0, description="Average page value before a purchase")
    SpecialDay: float = Field(0, ge=0, le=1, description="Closeness to a special day (0 to 1)")
    Month: Literal["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"] = "Nov"
    OperatingSystems: int = Field(2, ge=1, description="Operating system code")
    Browser: int = Field(2, ge=1, description="Browser code")
    Region: int = Field(1, ge=1, description="Region code")
    TrafficType: int = Field(2, ge=1, description="Traffic source code")
    VisitorType: Literal["Returning_Visitor", "New_Visitor", "Other"] = "Returning_Visitor"
    Weekend: bool = False


@app.get("/")
def home():
    return {"status": "ok", "message": "Use POST /predict or open /docs"}


@app.post("/predict")
def predict(visitor: Visitor):
    data = visitor.model_dump() if hasattr(visitor, "model_dump") else visitor.dict()
    return predictor.predict(data)
