from typing import Optional, List
import json
from fastapi import APIRouter, Depends, Query, Response
from pydantic import BaseModel
from services.config_.object_relations import getsession, MainTable
from services.api.V1.endpoints.auth import verify_credentials
from sqlalchemy.orm import Session
from sqlalchemy import select

router = APIRouter()


class SalesRecord(BaseModel):
    transaction_id: str
    customer_id: str
    category: str
    item: Optional[str] = None
    price_per_unit: Optional[float] = None
    quantity: Optional[float] = None
    total_spent: Optional[float] = None
    payment_method: str
    location: str
    transaction_date: Optional[str] = None
    discount_applied: Optional[bool] = None


@router.get(
    "/maindata"
)
def get_data(
    _: bool = Depends(verify_credentials),
    session: Session = Depends(getsession)
):

    result = session.execute(select(MainTable)).scalars().all()
    return result