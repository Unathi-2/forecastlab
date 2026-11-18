from fastapi import FastAPI

app = FastAPI(title="ForecastLab")


@app.get("/health")
def health():
    return {"status": "ok"}
