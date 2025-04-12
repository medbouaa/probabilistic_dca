import requests
import pandas as pd

def fetch_eia_bakken_production(api_key, output_csv='eia_bakken_production.csv'):
    url = "https://api.eia.gov/v2/petroleum/drilling/production/data/"

    params = {
        'api_key': api_key,
        'frequency': 'monthly',
        'data[0]': 'oil',
        'facets[region]': 'Bakken',
        'start': '2010-01',
        'end': '2025-12',
        'sort[0][column]': 'period',
        'sort[0][direction]': 'asc'
    }

    response = requests.get(url, params=params)
    response.raise_for_status()

    data = response.json()

    if not data['response']['data']:
        print("No data returned from EIA API v2.")
        return None

    df = pd.json_normalize(data['response']['data'])
    df = df[['period', 'oil']]
    df.rename(columns={'period': 'Date', 'oil': 'Production (thousand barrels per day)'}, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])

    df.to_csv(output_csv, index=False)
    print(f"🎉 Exported Bakken production data to {output_csv}")

    return df

# Run
if __name__ == "__main__":
    eia_api_key = "YOUR_API_KEY_HERE"
    df_bakken = fetch_eia_bakken_production(api_key=eia_api_key)
    if df_bakken is not None:
        print(df_bakken.head())

