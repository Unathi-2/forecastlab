import io
import json
from typing import Annotated

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from app.validation import validate_series

app = FastAPI(title="ForecastLab")

MAX_BYTES = 5 * 1024 * 1024  # 5 MB upload limit


async def read_csv(file: UploadFile) -> pd.DataFrame:
    content = await file.read()
    if len(content) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="File is larger than 5 MB")
    try:
        return pd.read_csv(io.BytesIO(content))
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError):
        raise HTTPException(
            status_code=400, detail="Could not read the file as a CSV"
        ) from None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/preview")
async def preview(file: Annotated[UploadFile, File()]):
    df = await read_csv(file)
    return {
        "rows": len(df),
        "columns": list(df.columns),
        "head": json.loads(df.head(5).to_json(orient="records")),
    }


@app.post("/validate")
async def validate(
    file: Annotated[UploadFile, File()],
    date_col: Annotated[str, Form()],
    target_col: Annotated[str, Form()],
):
    df = await read_csv(file)
    try:
        return validate_series(df, date_col, target_col)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from None
