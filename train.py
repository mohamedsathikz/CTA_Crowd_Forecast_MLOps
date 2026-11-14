import pandas as pd
from sqlalchemy import create_engine
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

# 1. MySQL Connection (Replace YOUR_PASSWORD with your MySQL root password)
DB_URL = os.getenv("DATABASE_URL", "mysql+pymysql://root:0207@localhost:3306/cta_ridership")
engine = create_engine(DB_URL)

def run_training_pipeline():
    dataset_file = 'cta_daily_ridership.csv'
    
    # Step A: Upload to MySQL
    if os.path.exists(dataset_file):
        print("1. Reading CSV and uploading to MySQL...")
        df_raw = pd.read_csv(dataset_file).tail(100000)
        df_raw.to_sql('daily_ridership', engine, if_exists='replace', index=False, chunksize=10000)
        print("   -> MySQL Upload Complete!")

    # Step B: Fetch Data from MySQL
    print("2. Fetching data from MySQL...")
    df = pd.read_sql("select * from daily_ridership", engine)

    # Step C: Feature Engineering (Pure Calendar Date Features ONLY - No W/A/U input needed)
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['day_of_week'] = df['date'].dt.day_of_week # 0=Mon, 1=Tue ... 5=Sat, 6=Sun
    df['Day Name'] = df['date'].dt.day_name()

    # Model trains ONLY on date-derived features!
    feature_cols = ['year', 'month', 'day', 'day_of_week']
    
    # Step D: Train Station-Specific Models
    station_models = {}
    station_means = {}
    stations = df[['station_id', 'stationname']].drop_duplicates()

    print(f"3. Training models for {len(stations)} stations using ONLY Date features...")
    df['Forecasted Crowd'] = 0

    for _, row in stations.iterrows():
        sid = row['station_id']
        station_data = df[df['station_id'] == sid]
        mean_val = int(station_data['rides'].mean())
        station_means[sid] = mean_val

        if len(station_data) > 30:
            X = station_data[feature_cols]
            y = station_data['rides']
            model = RandomForestRegressor(n_estimators=20, max_depth=10, random_state=42)
            model.fit(X, y)
            station_models[sid] = model
            df.loc[df['station_id'] == sid, 'Forecasted Crowd'] = model.predict(X).astype(int)
        else:
            df.loc[df['station_id'] == sid, 'Forecasted Crowd'] = mean_val

    df = df.rename(columns={'rides': 'Actual Passengers'})

    # Step E: Prepare Power BI Analytical Tables from MySQL
    print("4. Preparing Power BI Analytical Tables from MySQL...")
    top_10_df = pd.read_sql("""
        select stationname, sum(rides) as total_rides 
        from daily_ridership 
        group by stationname 
        order by total_rides desc limit 10;
    """, engine)

    avg_day_df = pd.read_sql("""
        select 
            case daytype
                when 'W' then 'Weekday'
                when 'A' then 'Saturday'
                when 'U' then 'Sunday/Holiday'
            end as `Day Category`,
            avg(rides) as average_rides
        from daily_ridership 
        group by daytype;
    """, engine)

    hist_sample = df.sort_values('date').tail(1000)[
        ['station_id', 'stationname', 'date', 'daytype', 'Actual Passengers', 
         'year', 'month', 'day', 'day_of_week', 'Forecasted Crowd', 'Day Name']
    ].copy()
    hist_sample['date'] = hist_sample['date'].dt.strftime('%Y-%m-%d')

    # Step F: Save Artifacts
    artifacts = {
        "models": station_models,
        "means": station_means,
        "stations": stations.to_dict(orient='records'),
        "top_10_historical": top_10_df.to_dict(orient='records'),
        "avg_by_daytype": avg_day_df.to_dict(orient='records'),
        "historical_forecast": hist_sample.to_dict(orient='records')
    }
    joblib.dump(artifacts, "model_artifacts.pkl")
    print("\n[SUCCESS] 'model_artifacts.pkl' updated with pure Date-based models!")

if __name__ == "__main__":
    run_training_pipeline()