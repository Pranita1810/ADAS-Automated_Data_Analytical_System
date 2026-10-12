from sqlalchemy.orm import Mapped, mapped_column, DeclarativeBaseNoMeta, Session
from sqlalchemy import String, Integer, Float, Boolean, Date, create_engine
from dotenv import load_dotenv
import os
from pathlib import Path
from datetime import date

_env_path = Path(__file__).resolve().parent / ".env"
if _env_path.is_file():
    load_dotenv(dotenv_path=_env_path)
else:
    load_dotenv()


# ------- MAIN ORM ENGINE -------
class Base(DeclarativeBaseNoMeta):
    pass

# main table
class MainTable(Base):
    __tablename__ = "sales"
    transaction_id : Mapped[str] = mapped_column(String, primary_key= True)
    customer_id : Mapped[str] = mapped_column(String)
    category : Mapped[str] = mapped_column(String)
    item : Mapped[str] = mapped_column(String)
    price_per_unit : Mapped[float] = mapped_column(Float)
    quantity : Mapped[float] = mapped_column(Float)
    total_spent : Mapped[float] = mapped_column(Float)
    payment_method : Mapped[str] = mapped_column(String)
    location : Mapped[str] = mapped_column(String)
    transaction_date : Mapped[date] = mapped_column(Date)
    discount_applied : Mapped[bool] = mapped_column(Boolean)

# main engine 
engine = create_engine(os.getenv("DATABASE_URL"))

#session
def getsession():
    with Session(engine) as session:
        yield session


