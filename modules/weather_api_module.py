import requests
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

class WeatherDataFetcher:
    """
    Fetches and processes weather data from OpenWeatherMap API
    for construction delay prediction.
    """
    
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.openweathermap.org/data/2.5"
        self.geo_url = "http://api.openweathermap.org/geo/1.0/direct"
    
    def geocode_location(self, location):
        """
        Convert location string to lat/lon coordinates.
        
        Args:
            location: String like "New York, NY" or "Los Angeles, CA"
        
        Returns:
            dict with lat, lon, and formatted location name
        """
        try:
            url = f"{self.geo_url}?q={location}&limit=1&appid={self.api_key}"
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            
            if data:
                return {
                    'lat': data[0]['lat'],
                    'lon': data[0]['lon'],
                    'name': data[0]['name'],
                    'state': data[0].get('state', ''),
                    'country': data[0]['country']
                }
            else:
                return None
        except Exception as e:
            print(f"Geocoding error: {e}")
            return None
    
    def get_current_weather(self, lat, lon):
        """
        Get current weather conditions.
        
        Returns:
            dict with temperature, humidity, wind_speed, condition, etc.
        """
        try:
            url = f"{self.base_url}/weather?lat={lat}&lon={lon}&appid={self.api_key}&units=imperial"
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            
            return {
                'temperature': data['main']['temp'],
                'feels_like': data['main']['feels_like'],
                'humidity': data['main']['humidity'],
                'pressure': data['main']['pressure'],
                'wind_speed': data['wind']['speed'],
                'wind_direction': data['wind'].get('deg', 0),
                'clouds': data['clouds']['all'],
                'condition': data['weather'][0]['main'],
                'description': data['weather'][0]['description'],
                'visibility': data.get('visibility', 10000) / 1000,  # Convert to km
                'rain_1h': data.get('rain', {}).get('1h', 0),
                'snow_1h': data.get('snow', {}).get('1h', 0),
                'timestamp': datetime.fromtimestamp(data['dt'])
            }
        except Exception as e:
            print(f"Current weather error: {e}")
            return None
    
    def get_forecast(self, lat, lon, days=5):
        """
        Get weather forecast for next N days.
        
        Returns:
            list of weather data points (3-hour intervals)
        """
        try:
            url = f"{self.base_url}/forecast?lat={lat}&lon={lon}&appid={self.api_key}&units=imperial"
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            
            forecast_list = []
            for item in data['list']:
                forecast_list.append({
                    'timestamp': datetime.fromtimestamp(item['dt']),
                    'temperature': item['main']['temp'],
                    'humidity': item['main']['humidity'],
                    'wind_speed': item['wind']['speed'],
                    'condition': item['weather'][0]['main'],
                    'description': item['weather'][0]['description'],
                    'rain_3h': item.get('rain', {}).get('3h', 0),
                    'snow_3h': item.get('snow', {}).get('3h', 0),
                    'clouds': item['clouds']['all']
                })
            
            return forecast_list[:days * 8]  # 8 forecasts per day (3-hour intervals)
        except Exception as e:
            print(f"Forecast error: {e}")
            return None
    
    def calculate_construction_risks(self, current_weather, forecast_data):
        """
        Calculate construction-specific weather risk factors.
        
        Returns:
            dict with risk scores and metrics
        """
        if not forecast_data:
            return None
        
        # Count adverse weather days
        rain_days = 0
        high_wind_days = 0
        extreme_temp_days = 0
        poor_visibility_days = 0
        
        daily_buckets = {}
        for forecast in forecast_data:
            date_key = forecast['timestamp'].date()
            if date_key not in daily_buckets:
                daily_buckets[date_key] = []
            daily_buckets[date_key].append(forecast)
        
        for date, forecasts in daily_buckets.items():
            # Rain day: any forecast shows rain > 0.5mm
            if any(f['rain_3h'] > 0.5 for f in forecasts):
                rain_days += 1
            
            # High wind: sustained winds > 25 mph
            if any(f['wind_speed'] > 25 for f in forecasts):
                high_wind_days += 1
            
            # Extreme temps: below 32°F or above 95°F
            if any(f['temperature'] < 32 or f['temperature'] > 95 for f in forecasts):
                extreme_temp_days += 1
        
        # Calculate averages
        avg_temp = np.mean([f['temperature'] for f in forecast_data])
        avg_humidity = np.mean([f['humidity'] for f in forecast_data])
        avg_wind = np.mean([f['wind_speed'] for f in forecast_data])
        total_rain = sum([f['rain_3h'] for f in forecast_data])
        
        # Weather severity score (1-5)
        severity = 1
        if rain_days >= 3 or high_wind_days >= 2:
            severity = 3
        if rain_days >= 5 or high_wind_days >= 4 or extreme_temp_days >= 3:
            severity = 4
        if (rain_days >= 5 and high_wind_days >= 3) or extreme_temp_days >= 5:
            severity = 5
        
        # Calculate expected weather delay (rough estimate)
        base_delay = rain_days * 0.7  # Each rain day causes ~0.7 day delay
        wind_delay = high_wind_days * 0.5  # High wind causes ~0.5 day delay
        temp_delay = extreme_temp_days * 0.3  # Extreme temp causes ~0.3 day delay
        
        expected_weather_delay = base_delay + wind_delay + temp_delay
        
        return {
            'rain_days': rain_days,
            'high_wind_days': high_wind_days,
            'extreme_temp_days': extreme_temp_days,
            'avg_temperature': avg_temp,
            'avg_humidity': avg_humidity,
            'avg_wind_speed': avg_wind,
            'total_rainfall': total_rain,
            'weather_severity': severity,
            'expected_weather_delay_days': expected_weather_delay,
            'forecast_period_days': len(daily_buckets),
            'workable_days': len(daily_buckets) - rain_days - high_wind_days
        }
    
    def get_construction_forecast(self, location, project_duration_days=30):
        """
        Complete weather analysis for construction project.
        
        Args:
            location: City/location string
            project_duration_days: How long project will last
        
        Returns:
            Complete weather analysis dict
        """
        # Geocode location
        coords = self.geocode_location(location)
        if not coords:
            return {'error': 'Location not found'}
        
        # Get current weather
        current = self.get_current_weather(coords['lat'], coords['lon'])
        if not current:
            return {'error': 'Could not fetch current weather'}
        
        # Get forecast
        forecast = self.get_forecast(coords['lat'], coords['lon'])
        if not forecast:
            return {'error': 'Could not fetch forecast'}
        
        # Calculate construction risks
        risks = self.calculate_construction_risks(current, forecast)
        
        return {
            'location': f"{coords['name']}, {coords['state']}, {coords['country']}",
            'coordinates': {'lat': coords['lat'], 'lon': coords['lon']},
            'current_weather': current,
            'forecast': forecast,
            'construction_risks': risks,
            'timestamp': datetime.now()
        }
    
    def format_for_model_input(self, weather_data):
        """
        Convert weather API response to model input features.
        
        Returns:
            dict ready for model prediction
        """
        if 'error' in weather_data:
            return None
        
        risks = weather_data['construction_risks']
        
        return {
            'Rain_Days': risks['rain_days'],
            'Weather_Severity': risks['weather_severity'],
            'Avg_Temperature': risks['avg_temperature'],
            'Avg_Humidity': risks['avg_humidity'],
            'Avg_Wind_Speed': risks['avg_wind_speed'],
            'Expected_Weather_Delay': risks['expected_weather_delay_days'],
            'Workable_Days_Pct': (risks['workable_days'] / risks['forecast_period_days']) * 100
        }


# Example usage
if __name__ == "__main__":
    # Get API key from environment or input
    API_KEY = "your_api_key_here"
    
    fetcher = WeatherDataFetcher(API_KEY)
    
    # Get weather for construction project in NYC
    weather = fetcher.get_construction_forecast("New York, NY", project_duration_days=90)
    
    if 'error' not in weather:
        print(f"Location: {weather['location']}")
        print(f"\nCurrent Conditions:")
        print(f"  Temperature: {weather['current_weather']['temperature']:.1f}°F")
        print(f"  Condition: {weather['current_weather']['condition']}")
        print(f"  Wind: {weather['current_weather']['wind_speed']:.1f} mph")
        
        print(f"\nConstruction Risk Analysis:")
        risks = weather['construction_risks']
        print(f"  Rain Days (next 5): {risks['rain_days']}")
        print(f"  Weather Severity: {risks['weather_severity']}/5")
        print(f"  Expected Delay: {risks['expected_weather_delay_days']:.1f} days")
        print(f"  Workable Days: {risks['workable_days']}/{risks['forecast_period_days']}")
        
        # Get model input format
        model_input = fetcher.format_for_model_input(weather)
        print(f"\nModel Input Features:")
        for key, value in model_input.items():
            print(f"  {key}: {value}")
    else:
        print(f"Error: {weather['error']}")
