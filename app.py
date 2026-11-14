from fastapi import FastAPI, Response
import pandas as pd
import joblib
import datetime
import os
from train import run_training_pipeline

app = FastAPI(title="CTA 6-Page Power BI Dashboard - MLOps API")

MODEL_PATH = "model_artifacts.pkl"
LATEST_CSV_PATH = "latest_forecast_report.csv"
latest_predicted_date = "2026-11-15"

def to_csv_response(data_list):
    df = pd.DataFrame(data_list)
    return Response(content=df.to_csv(index=False), media_type="text/csv")

@app.get("/")
def home():
    return {
        "status": "Online",
        "currently_predicted_date": latest_predicted_date,
        "swagger_ui": "Go to /docs to change forecast date or trigger retraining"
    }

# 1. Change Forecast Date (Takes ONLY year, month, day)
@app.post("/update-forecast-date")
def update_forecast_date(year: int = 2026, month: int = 11, day: int = 15):
    global latest_predicted_date
    if not os.path.exists(MODEL_PATH):
        return {"error": "model_artifacts.pkl not found. Run train.py first."}

    try:
        target_date = datetime.date(year, month, day)
    except ValueError as e:
        return {
            "status": "Invalid Date Error",
            "reason": str(e),
            "tip": "Month must be 1 to 12, and Day must be valid for that month."
        }

    art = joblib.load(MODEL_PATH)
    models = art["models"]
    means = art["means"]
    stations = art["stations"]

    # Extract day_of_week (0=Monday ... 5=Saturday, 6=Sunday) automatically from calendar date
    day_of_week = target_date.weekday()
    day_name = target_date.strftime("%A")
    
    # Pass exact 4 date features to the trained model: [year, month, day, day_of_week]
    input_features = [[year, month, day, day_of_week]]

    # Subtle yearly growth factor for future years
    year_factor = 1.0 + ((year - 2026) * 0.015)
    year_factor = max(0.8, min(year_factor, 1.5))

    results = []
    for st in stations:
        sid = st['station_id']
        sname = st['stationname']
        model = models.get(sid)
        base_pred = model.predict(input_features)[0] if model else means.get(sid, 0)
        pred = int(base_pred * year_factor)
        results.append({
            "Station ID": sid,
            "Station Name": sname,
            "Estimated Passengers": pred,
            "Forecast Date": str(target_date),
            "Day Name": day_name
        })

    pd.DataFrame(results).to_csv(LATEST_CSV_PATH, index=False)
    latest_predicted_date = str(target_date)

    return {
        "status": "Success",
        "predicted_date": str(target_date),
        "predicted_day_name": day_name,
        "stations_updated": len(results),
        "message": f"Forecast updated for {target_date} ({day_name})! Refresh Power BI to see new visuals."
    }

# 2. Power BI Table 1: forecast_report.csv
@app.get("/powerbi/forecast_report.csv")
def get_forecast_report():
    if not os.path.exists(LATEST_CSV_PATH):
        update_forecast_date(2026, 11, 15)
    df = pd.read_csv(LATEST_CSV_PATH)
    return Response(content=df.to_csv(index=False), media_type="text/csv")

# 3. Power BI Table 2: top_10_stations.csv
@app.get("/powerbi/top_10_stations.csv")
def get_top_10_stations():
    art = joblib.load(MODEL_PATH)
    return to_csv_response(art.get("top_10_historical", []))

# 4. Power BI Table 3: avg_rides_by_daytype.csv
@app.get("/powerbi/avg_rides_by_daytype.csv")
def get_avg_rides_by_daytype():
    art = joblib.load(MODEL_PATH)
    return to_csv_response(art.get("avg_by_daytype", []))

# 5. Power BI Table 4: final_forecast_results.csv
@app.get("/powerbi/final_forecast_results.csv")
def get_final_forecast_results():
    art = joblib.load(MODEL_PATH)
    return to_csv_response(art.get("historical_forecast", []))

# 6. Power BI Table 5: dim_station.csv
@app.get("/powerbi/dim_station.csv")
def get_dim_station():
    art = joblib.load(MODEL_PATH)
    return to_csv_response(art.get("stations", []))

# 7. Power BI Table 6: summary_file.csv
@app.get("/powerbi/summary_file.csv")
def get_summary_file():
    art = joblib.load(MODEL_PATH)
    summary_data = [
        {"File Name": "forecast_report.csv", "Type": "Live Prediction", "Source": f"FastAPI ({latest_predicted_date})", "File Rows": len(art.get("stations", [])), "File Status": "Active"},
        {"File Name": "final_forecast_results.csv", "Type": "Historical + Forecast", "Source": "MySQL + RF Model", "File Rows": len(art.get("historical_forecast", [])), "File Status": "Active"},
        {"File Name": "top_10_stations.csv", "Type": "SQL Aggregation", "Source": "MySQL Database", "File Rows": 10, "File Status": "Active"},
        {"File Name": "avg_rides_by_daytype.csv", "Type": "SQL Aggregation", "Source": "MySQL Database", "File Rows": 3, "File Status": "Active"}
    ]
    return to_csv_response(summary_data)

# 8. MLOps Continuous Training Endpoint
@app.post("/retrain")
def trigger_retrain():
    run_training_pipeline()
    return {"status": "Success", "message": "Continuous Training completed from MySQL!"}