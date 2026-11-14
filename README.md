# CTA Public Transit Crowd Forecasting & MLOps Pipeline

End-to-end MLOps pipeline for forecasting Chicago Transit Authority (CTA) passenger volume across 145 stations, integrated live with a 6-page Power BI Dashboard.

## 📂 Dataset
- **Download Dataset:** [Click here to download cta_daily_ridership.csv](https://drive.google.com/uc?export=download&id=1M3m2O91hrxGsrYR0eDvSMLSS9cxggpcT)

## 🚀 Tech Stack
- **Python 3.12 / FastAPI**: For REST API and dynamic date-based predictions.
- **MySQL / SQLAlchemy**: For local database storage and historical aggregation.
- **Scikit-Learn (Random Forest Regressor)**: For station-specific crowd forecasting models.
- **Docker**: For containerization and cross-platform deployment.
- **Render**: For cloud web service deployment.
- **Power BI**: For 6-page interactive analytics and visualization.

## 📌 API Endpoints
- `GET /`: Health check and status.
- `POST /update-forecast-date?year=2026&month=11&day=15`: Dynamically updates predictions for all stations.
- `GET /powerbi/forecast_report.csv`: Live prediction data source for Power BI.
- `POST /retrain`: Triggers continuous training from MySQL.