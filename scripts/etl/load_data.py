"""
ETL: Generate synthetic billing claims, inject 50 anomalies, write to Postgres.
"""

import asyncio
import random
import uuid
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from billing_anomaly_detector.infrastructure.config import get_settings
from billing_anomaly_detector.infrastructure.persistence.models import InvoiceModel

SYNPUF_PATH = Path("data/DE1_0_2008_to_2010_Carrier_Claims_Sample_1A.csv")
RANDOM_SEED = 42

NORMAL_CODES = {
    "99213": (1.0, 1.8),
    "99214": (1.0, 2.0),
    "93000": (1.0, 1.5),
    "71046": (1.0, 1.6),
    "80053": (1.0, 1.4),
    "99232": (1.0, 1.7),
    "99283": (1.0, 1.9),
    "43239": (1.0, 2.2),
}

ANOMALY_CODES = {
    "99215": (6.0, 12.0),
    "27447": (5.0, 9.0),
    "33533": (7.0, 15.0),
}


def generate_normal_claims(n: int = 5000) -> pd.DataFrame:
    random.seed(RANDOM_SEED)
    rows = []
    codes = list(NORMAL_CODES.keys())
    for _ in range(n):
        code = random.choice(codes)
        lo, hi = NORMAL_CODES[code]
        allowed = round(random.uniform(50.0, 800.0), 2)
        ratio = round(random.uniform(lo, hi), 2)
        billed = round(allowed * ratio, 2)
        rows.append({
            "member_id": f"MBR{random.randint(100000, 999999)}",
            "claim_code": code,
            "provider_npi": f"{random.randint(1000000000, 9999999999)}",
            "billed_amount": billed,
            "allowed_amount": allowed,
            "service_date": date(2008, 1, 1) + timedelta(days=random.randint(0, 730)),
        })
    return pd.DataFrame(rows)


def load_synpuf(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    return pd.DataFrame({
        "member_id":      df["DESYNPUF_ID"],
        "claim_code":     df["HCPCS_CD_1"].fillna("99213"),
        "provider_npi":   df["PRF_PHYSN_NPI_1"].fillna("0000000000"),
        "billed_amount":  pd.to_numeric(df["LINE_ALOWD_CHRG_AMT"], errors="coerce").fillna(0),
        "allowed_amount": pd.to_numeric(df["LINE_NCH_PMT_AMT"], errors="coerce").fillna(0),
        "service_date":   pd.to_datetime(df["CLM_FROM_DT"], format="%Y%m%d").dt.date,
    })


def _make_row(
    code: str,
    billed: float,
    allowed: float,
    member_id: str | None = None,
    srv_date: date | None = None,
) -> dict:
    return {
        "member_id":      member_id or f"MBR{random.randint(100000, 999999)}",
        "claim_code":     code,
        "provider_npi":   f"{random.randint(1000000000, 9999999999)}",
        "billed_amount":  billed,
        "allowed_amount": allowed,
        "service_date":   srv_date or (date(2008, 1, 1) + timedelta(days=random.randint(0, 730))),
    }


def inject_anomalies(df: pd.DataFrame, count: int = 50) -> pd.DataFrame:
    random.seed(RANDOM_SEED + 1)
    anomalies = []
    codes = list(ANOMALY_CODES.keys())

    for _ in range(10):
        code = random.choice(codes)
        lo, hi = ANOMALY_CODES[code]
        allowed = round(random.uniform(500.0, 3000.0), 2)
        billed = round(allowed * random.uniform(lo, hi), 2)
        anomalies.append(_make_row(code, billed, allowed))

    for _ in range(10):
        allowed = round(random.uniform(100.0, 1000.0), 2)
        billed = round(allowed * random.uniform(4.0, 8.0) / 500) * 500
        anomalies.append(_make_row("99215", float(billed), allowed))

    for _ in range(10):
        billed = round(random.uniform(500.0, 5000.0), 2)
        allowed = round(random.uniform(0.01, 5.0), 2)
        anomalies.append(_make_row("99214", billed, allowed))

    member_id = f"MBR{random.randint(100000, 999999)}"
    srv_date = date(2009, 6, 15)
    for _ in range(10):
        allowed = round(random.uniform(200.0, 800.0), 2)
        billed = round(allowed * random.uniform(5.0, 9.0), 2)
        anomalies.append(_make_row("27447", billed, allowed, member_id, srv_date))

    for _ in range(10):
        allowed = round(random.uniform(100.0, 400.0), 2)
        billed = round(allowed * random.uniform(6.0, 11.0), 2)
        anomalies.append(_make_row("33533", billed, allowed))

    anomaly_df = pd.DataFrame(anomalies)
    print(f"  Injected {len(anomaly_df)} anomalies across 5 types")
    return pd.concat([df, anomaly_df], ignore_index=True)


def prepare_data() -> pd.DataFrame:
    """
    Synchronous data preparation — loads, enriches, and injects anomalies.
    Kept separate from async DB writes so filesystem calls don't block the event loop.
    """
    print("Loading claim data...")
    if SYNPUF_PATH.exists():
        df = load_synpuf(SYNPUF_PATH)
        print(f"  Loaded {len(df):,} rows from SynPUF")
    else:
        print("  SynPUF file not found — generating 5,000 synthetic claims")
        df = generate_normal_claims(5000)

    print("Injecting anomalies...")
    df = inject_anomalies(df)

    before = len(df)
    df = df[df["allowed_amount"] > 0]
    print(f"  Dropped {before - len(df)} rows with zero allowed_amount")
    return df


async def write_to_db(
    df: pd.DataFrame,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    batch_size = 500
    total = len(df)
    written = 0
    for start in range(0, total, batch_size):
        batch = df.iloc[start : start + batch_size]
        async with session_factory() as session:
            models = [
                InvoiceModel(
                    id=uuid.uuid4(),
                    member_id=str(row.member_id),
                    claim_code=str(row.claim_code),
                    provider_npi=str(row.provider_npi),
                    billed_amount=Decimal(str(row.billed_amount)),
                    billed_currency="USD",
                    allowed_amount=Decimal(str(row.allowed_amount)),
                    allowed_currency="USD",
                    service_date=row.service_date,
                )
                for row in batch.itertuples(index=False)
            ]
            session.add_all(models)
            await session.commit()
        written += len(batch)
        print(f"  Written {written}/{total} rows...")


async def main() -> None:
    settings = get_settings()
    engine = create_async_engine(settings.database_url, echo=False)
    session_factory = async_sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )

    # prepare_data() is synchronous — safe to call directly from async context
    # since it completes before any awaitable operations begin
    df = prepare_data()

    print(f"Writing {len(df):,} rows to NAS database...")
    await write_to_db(df, session_factory)
    await engine.dispose()
    print("ETL complete.")


if __name__ == "__main__":
    asyncio.run(main())
