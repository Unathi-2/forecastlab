from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

CSV = b"date,sales\n2024-01-01,10\n2024-01-02,12\n2024-01-03,14\n2024-01-04,15\n"


def upload(data=None, content=CSV):
    return client.post(
        "/validate",
        files={"file": ("data.csv", content, "text/csv")},
        data=data,
    )


def test_preview_returns_columns():
    response = client.post(
        "/preview", files={"file": ("data.csv", CSV, "text/csv")}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["columns"] == ["date", "sales"]
    assert body["rows"] == 4


def test_validate_returns_report():
    response = upload({"date_col": "date", "target_col": "sales"})
    assert response.status_code == 200
    assert response.json()["is_regular"] is True


def test_validate_unknown_column_gives_422():
    response = upload({"date_col": "date", "target_col": "revenue"})
    assert response.status_code == 422


def test_unreadable_file_gives_400():
    response = upload({"date_col": "date", "target_col": "sales"}, content=b"")
    assert response.status_code == 400
