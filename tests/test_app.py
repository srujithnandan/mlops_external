from fastapi.testclient import TestClient

import app


client = TestClient(app.app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_prediction_rejects_unknown_fields():
    response = client.post(
        "/predict",
        json={
            "material": "PLA",
            "nozzle_temperature": 205,
            "bed_temperature": 55,
            "printing_speed": 50,
            "ambient_temperature": 23,
            "cooling_fan_speed": 50,
            "component_base_area": 100,
            "layer_height": 0.2,
            "unexpected": 1,
        },
    )
    assert response.status_code == 422
