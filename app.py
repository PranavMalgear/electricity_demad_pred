from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

app = Flask(__name__)
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "electricity_demand.csv"
FEATURES = [
    "temperature_c", "humidity", "hour", "day_of_week",
    "month", "day", "is_weekend"
]


@lru_cache(maxsize=1)
def load_data():
    """Read and prepare the bundled dataset once per server worker."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH, parse_dates=["datetime"])
    return df.sort_values("datetime").reset_index(drop=True)


def add_features(df):
    out = df.copy()
    out["hour"] = out["datetime"].dt.hour
    out["day_of_week"] = out["datetime"].dt.dayofweek
    out["month"] = out["datetime"].dt.month
    out["day"] = out["datetime"].dt.day
    out["is_weekend"] = (out["day_of_week"] >= 5).astype(int)
    return out


@lru_cache(maxsize=1)
def get_trained_model():
    """Train once and reuse the model for homepage metrics and predictions."""
    df = load_data()
    featured = add_features(df)
    split = int(len(featured) * 0.8)
    train, test = featured.iloc[:split], featured.iloc[split:]

    model = RandomForestRegressor(
        n_estimators=80,
        max_depth=12,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=1,
    )
    model.fit(train[FEATURES], train["demand_mw"])
    pred = model.predict(test[FEATURES])
    metrics = {
        "mae": round(float(mean_absolute_error(test["demand_mw"], pred)), 2),
        "rmse": round(float(np.sqrt(mean_squared_error(test["demand_mw"], pred))), 2),
        "r2": round(float(r2_score(test["demand_mw"], pred)), 3),
    }
    return model, metrics


@app.route("/")
def home():
    df = load_data()
    _, metrics = get_trained_model()
    latest = df.iloc[-1]
    return render_template(
        "index.html",
        metrics=metrics,
        latest_date=latest.datetime.strftime("%d %b %Y, %H:%M"),
        latest_demand=round(float(latest.demand_mw), 1),
        average=round(float(df.demand_mw.mean()), 1),
        peak=round(float(df.demand_mw.max()), 1),
        records=len(df),
    )


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/api/overview")
def overview():
    df = load_data()
    daily = df.set_index("datetime").resample("D")["demand_mw"].mean().tail(30)
    hourly = df.groupby(df.datetime.dt.hour)["demand_mw"].mean()
    return jsonify({
        "daily_labels": [x.strftime("%d %b") for x in daily.index],
        "daily_values": [round(float(x), 1) for x in daily.values],
        "hour_labels": [f"{int(x):02d}:00" for x in hourly.index],
        "hour_values": [round(float(x), 1) for x in hourly.values],
        "recent_labels": [x.strftime("%d %b %H:%M") for x in df.datetime.tail(48)],
        "recent_values": [round(float(x), 1) for x in df.demand_mw.tail(48)],
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        payload = request.get_json(silent=True) or {}
        dt = pd.to_datetime(payload["datetime"], errors="raise")
        temp = float(payload["temperature"])
        humidity = float(payload["humidity"])

        if not -10 <= temp <= 55:
            raise ValueError("Temperature must be between -10 and 55 °C.")
        if not 0 <= humidity <= 100:
            raise ValueError("Humidity must be between 0 and 100%.")

        row = pd.DataFrame([{
            "temperature_c": temp,
            "humidity": humidity,
            "hour": dt.hour,
            "day_of_week": dt.dayofweek,
            "month": dt.month,
            "day": dt.day,
            "is_weekend": int(dt.dayofweek >= 5),
        }])
        model, _ = get_trained_model()
        value = float(model.predict(row[FEATURES])[0])
        return jsonify({
            "demand": round(value, 1),
            "unit": "MW",
            "note": "Model estimate based on the included demonstration dataset.",
        })
    except (KeyError, TypeError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify({"error": "The prediction could not be completed. Please try again."}), 500


@app.route("/api/data")
def data():
    df = load_data().tail(100)
    return jsonify([
        {
            "datetime": r.datetime.strftime("%Y-%m-%d %H:%M"),
            "demand_mw": round(float(r.demand_mw), 1),
            "temperature_c": round(float(r.temperature_c), 1),
            "humidity": int(r.humidity),
        }
        for r in df.itertuples()
    ])


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
