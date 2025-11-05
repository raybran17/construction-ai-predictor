import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pickle
import os

# Page config
st.set_page_config(
    page_title="Construction AI Predictor",
    page_icon="🏗️",
    layout="wide"
)

# Title
st.title("🏗️ Construction Project AI Predictor")
st.markdown("### Predict delays and costs using AI + real-time weather data")

# Sidebar for navigation
mode = st.sidebar.selectbox(
    "Select Mode",
    ["📊 Test Model Accuracy", "🔮 Predict New Project", "📈 Dashboard Overview"]
)

# Load models (with error handling for when they don't exist yet)
@st.cache_resource
def load_models():
    models = {}
    try:
        if os.path.exists('trained_delay_model.pkl'):
            with open('trained_delay_model.pkl', 'rb') as f:
                models['delay'] = pickle.load(f)
        if os.path.exists('trained_cost_model.pkl'):
            with open('trained_cost_model.pkl', 'rb') as f:
                models['cost'] = pickle.load(f)
    except Exception as e:
        st.sidebar.warning(f"Models not loaded yet: {e}")
    return models

models = load_models()

# ============================================================================
# MODE 1: TEST MODEL ACCURACY (Upload Past Data)
# ============================================================================
if mode == "📊 Test Model Accuracy":
    st.header("📊 Test Model Accuracy on Historical Data")
    st.markdown("Upload your past project data to see how well the AI performs")
    
    uploaded_file = st.file_uploader("Upload CSV with historical projects", type=['csv'])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        
        st.success(f"✅ Loaded {len(df)} projects")
        
        # Show data preview
        with st.expander("📋 View Data Preview"):
            st.dataframe(df.head(10))
        
        col1, col2 = st.columns(2)
        
        # Select target column
        with col1:
            target_col = st.selectbox(
                "Select Target Variable",
                [col for col in df.columns if 'delay' in col.lower() or 'cost' in col.lower()]
            )
        
        # Run preprocessing button
        if st.button("🔄 Preprocess & Evaluate", type="primary"):
            with st.spinner("Processing data..."):
                # Import preprocessing engine
                try:
                    from preprocessing_delay_engine import ConstructionDataPreprocessor
                    
                    # Save uploaded file temporarily
                    temp_path = 'temp_upload.csv'
                    df.to_csv(temp_path, index=False)
                    
                    # Run preprocessing
                    preprocessor = ConstructionDataPreprocessor()
                    preprocessor.run_full_pipeline(temp_path, 'temp_cleaned.csv')
                    
                    # Get model-ready data
                    X, y = preprocessor.get_model_ready_data(target_column=target_col)
                    
                    st.success("✅ Preprocessing complete!")
                    
                    # Display results
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Projects", len(df))
                    with col2:
                        st.metric("Features Created", X.shape[1])
                    with col3:
                        st.metric("Missing Values", preprocessor.df.isnull().sum().sum())
                    
                    # Show feature importance (if model exists)
                    if 'delay' in models and target_col and 'delay' in target_col.lower():
                        st.subheader("🎯 Model Performance")
                        
                        # Make predictions
                        predictions = models['delay'].predict(X)
                        
                        # Calculate metrics
                        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
                        
                        mae = mean_absolute_error(y, predictions)
                        rmse = np.sqrt(mean_squared_error(y, predictions))
                        r2 = r2_score(y, predictions)
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("MAE", f"{mae:.2f} days")
                        with col2:
                            st.metric("RMSE", f"{rmse:.2f} days")
                        with col3:
                            st.metric("R² Score", f"{r2:.3f}")
                        
                        # Predicted vs Actual plot
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=y, y=predictions,
                            mode='markers',
                            name='Predictions',
                            marker=dict(size=10, color='blue', opacity=0.6)
                        ))
                        fig.add_trace(go.Scatter(
                            x=[y.min(), y.max()],
                            y=[y.min(), y.max()],
                            mode='lines',
                            name='Perfect Prediction',
                            line=dict(color='red', dash='dash')
                        ))
                        fig.update_layout(
                            title="Predicted vs Actual Delays",
                            xaxis_title="Actual Delays (days)",
                            yaxis_title="Predicted Delays (days)",
                            height=500
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Distribution plots
                    st.subheader("📊 Data Distribution")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig = px.histogram(preprocessor.df, x=target_col, 
                                         title=f"Distribution of {target_col}")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Feature correlation
                        numeric_cols = X.columns[:5]  # Top 5 features
                        corr_data = pd.DataFrame({
                            'Feature': numeric_cols,
                            'Correlation': [X[col].corr(y) for col in numeric_cols]
                        })
                        fig = px.bar(corr_data, x='Feature', y='Correlation',
                                   title="Top 5 Feature Correlations")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Download processed data
                    st.download_button(
                        "📥 Download Processed Data",
                        data=preprocessor.df.to_csv(index=False),
                        file_name="processed_data.csv",
                        mime="text/csv"
                    )
                    
                except Exception as e:
                    st.error(f"Error during preprocessing: {e}")
                    st.info("Make sure your preprocessing engines are in the same directory!")

# ============================================================================
# MODE 2: PREDICT NEW PROJECT (With Live Weather)
# ============================================================================
elif mode == "🔮 Predict New Project":
    st.header("🔮 Predict New Project Delays & Costs")
    st.markdown("Enter project details + get live weather data for prediction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Project Details")
        project_type = st.selectbox("Project Type", 
                                    ["Residential", "Commercial", "Infrastructure", "Industrial"])
        location = st.text_input("Location (City, State)", "New York, NY")
        crew_size = st.number_input("Crew Size", min_value=1, max_value=200, value=20)
        start_date = st.date_input("Project Start Date", datetime.now())
        duration = st.number_input("Planned Duration (days)", min_value=1, value=90)
        
        st.subheader("🚚 Supply Chain Info")
        supplier_reliability = st.slider("Supplier Reliability Score", 0, 100, 85)
        material_lead_time = st.number_input("Material Lead Time (days)", 0, 30, 7)
        inspections_planned = st.number_input("Inspections Planned", 1, 20, 5)
    
    with col2:
        st.subheader("☁️ Weather Integration")
        
        # Weather API key input
        weather_api_key = st.text_input(
            "OpenWeatherMap API Key", 
            type="password",
            help="Get free API key at openweathermap.org/api"
        )
        
        if st.button("🌤️ Fetch Live Weather Data", type="primary"):
            if weather_api_key:
                import requests
                
                try:
                    # Geocode location
                    geo_url = f"http://api.openweathermap.org/geo/1.0/direct?q={location}&limit=1&appid={weather_api_key}"
                    geo_response = requests.get(geo_url)
                    geo_data = geo_response.json()
                    
                    if geo_data:
                        lat = geo_data[0]['lat']
                        lon = geo_data[0]['lon']
                        
                        # Get current weather
                        weather_url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={weather_api_key}&units=imperial"
                        weather_response = requests.get(weather_url)
                        weather_data = weather_response.json()
                        
                        # Get forecast (5-day)
                        forecast_url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={weather_api_key}&units=imperial"
                        forecast_response = requests.get(forecast_url)
                        forecast_data = forecast_response.json()
                        
                        # Display current weather
                        st.success("✅ Weather data fetched!")
                        
                        weather_col1, weather_col2 = st.columns(2)
                        with weather_col1:
                            st.metric("🌡️ Temperature", f"{weather_data['main']['temp']:.1f}°F")
                            st.metric("💧 Humidity", f"{weather_data['main']['humidity']}%")
                        with weather_col2:
                            st.metric("💨 Wind Speed", f"{weather_data['wind']['speed']} mph")
                            st.metric("🌧️ Condition", weather_data['weather'][0]['main'])
                        
                        # Calculate weather risk factors
                        rain_days_forecast = sum(1 for item in forecast_data['list'] 
                                               if 'rain' in item.get('weather', [{}])[0].get('main', '').lower())
                        avg_wind = np.mean([item['wind']['speed'] for item in forecast_data['list'][:8]])
                        
                        # Store in session state for prediction
                        st.session_state['weather_data'] = {
                            'rain_days': rain_days_forecast,
                            'avg_temp': weather_data['main']['temp'],
                            'avg_humidity': weather_data['main']['humidity'],
                            'avg_wind': avg_wind,
                            'weather_severity': 3 if rain_days_forecast > 2 else 1
                        }
                        
                        st.info(f"📊 Forecast shows {rain_days_forecast} rainy days in next 5 days")
                        
                    else:
                        st.error("Location not found. Try format: 'City, State'")
                        
                except Exception as e:
                    st.error(f"Weather API Error: {e}")
            else:
                st.warning("Please enter your OpenWeatherMap API key")
        
        # Manual weather input option
        with st.expander("⚙️ Or Enter Weather Manually"):
            manual_rain_days = st.number_input("Expected Rain Days", 0, 30, 5)
            manual_wind_speed = st.number_input("Avg Wind Speed (mph)", 0, 50, 10)
            manual_temp = st.number_input("Avg Temperature (°F)", 0, 120, 70)
            manual_severity = st.slider("Weather Severity (1-5)", 1, 5, 2)
    
    # Make prediction
    st.markdown("---")
    if st.button("🎯 Generate Prediction", type="primary", use_container_width=True):
        with st.spinner("Running AI model..."):
            # Create feature vector
            weather_data = st.session_state.get('weather_data', {
                'rain_days': manual_rain_days,
                'avg_temp': manual_temp,
                'avg_humidity': 60,
                'avg_wind': manual_wind_speed,
                'weather_severity': manual_severity
            })
            
            # Prepare input features (mock for now - adjust to your model's features)
            input_features = pd.DataFrame({
                'Crew_Size': [crew_size],
                'Rain_Days': [weather_data['rain_days']],
                'Material_Delay_Days': [material_lead_time],
                'Inspections_Passed': [inspections_planned],
                'Supplier_Reliability': [supplier_reliability],
                'Weather_Severity': [weather_data['weather_severity']]
            })
            
            # Mock prediction (replace with actual model when ready)
            if 'delay' in models:
                predicted_delay = models['delay'].predict(input_features)[0]
            else:
                # Simple heuristic for demo
                base_delay = weather_data['rain_days'] * 0.8
                supplier_delay = (100 - supplier_reliability) / 10
                weather_multiplier = weather_data['weather_severity'] / 3
                predicted_delay = (base_delay + supplier_delay) * weather_multiplier
            
            # Display results
            st.success("✅ Prediction Complete!")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📅 Planned Duration", f"{duration} days")
            with col2:
                st.metric("⚠️ Predicted Delay", f"{predicted_delay:.1f} days", 
                         delta=f"{(predicted_delay/duration*100):.1f}%", delta_color="inverse")
            with col3:
                actual_duration = duration + predicted_delay
                st.metric("🎯 Total Expected Duration", f"{actual_duration:.1f} days")
            
            # Risk breakdown
            st.subheader("⚠️ Risk Factor Breakdown")
            risk_data = pd.DataFrame({
                'Factor': ['Weather', 'Supply Chain', 'Crew Efficiency', 'Inspections'],
                'Impact': [
                    weather_data['rain_days'] * 10,
                    (100 - supplier_reliability),
                    max(0, 50 - crew_size),
                    max(0, inspections_planned * 5)
                ]
            })
            
            fig = px.bar(risk_data, x='Factor', y='Impact', 
                        title="Delay Risk Contributors",
                        color='Impact',
                        color_continuous_scale='Reds')
            st.plotly_chart(fig, use_container_width=True)
            
            # Recommendations
            st.subheader("💡 Recommendations")
            if predicted_delay > duration * 0.15:
                st.warning("⚠️ High delay risk detected!")
                st.markdown(f"""
                - Consider increasing crew size by {int(crew_size * 0.3)} workers
                - Add {int(weather_data['rain_days'] * 1.5)} buffer days for weather
                - Source materials {material_lead_time + 7} days earlier
                """)
            else:
                st.success("✅ Project has low delay risk")
                st.markdown("- Current planning appears adequate")

# ============================================================================
# MODE 3: DASHBOARD OVERVIEW
# ============================================================================
elif mode == "📈 Dashboard Overview":
    st.header("📈 Model Performance Dashboard")
    
    # Model status
    st.subheader("🤖 Model Status")
    col1, col2 = st.columns(2)
    
    with col1:
        if 'delay' in models:
            st.success("✅ Delay Model: Loaded")
        else:
            st.warning("⚠️ Delay Model: Not trained yet")
            st.info("Train your model and save as 'trained_delay_model.pkl'")
    
    with col2:
        if 'cost' in models:
            st.success("✅ Cost Model: Loaded")
        else:
            st.warning("⚠️ Cost Model: Not trained yet")
            st.info("Train your model and save as 'trained_cost_model.pkl'")
    
    # Quick stats (mock data for visualization)
    st.subheader("📊 Historical Performance")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Projects Analyzed", "127")
    with col2:
        st.metric("Avg Prediction Accuracy", "84.3%", "↑ 5.2%")
    with col3:
        st.metric("Avg Delay Predicted", "12.4 days")
    with col4:
        st.metric("Cost Savings Identified", "$2.4M")
    
    # Sample visualizations
    st.subheader("📉 Model Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Sample prediction accuracy over time
        dates = pd.date_range(start='2024-01-01', periods=12, freq='M')
        accuracy = np.random.uniform(0.75, 0.92, 12)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates, y=accuracy,
            mode='lines+markers',
            name='Accuracy',
            line=dict(color='green', width=3)
        ))
        fig.update_layout(title="Model Accuracy Over Time", 
                         yaxis_title="Accuracy (%)",
                         yaxis=dict(range=[0.7, 1.0]))
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Project types distribution
        project_types = ['Residential', 'Commercial', 'Infrastructure', 'Industrial']
        counts = [45, 38, 28, 16]
        
        fig = px.pie(values=counts, names=project_types, 
                    title="Projects by Type")
        st.plotly_chart(fig, use_container_width=True)
    
    # Instructions
    st.markdown("---")
    st.subheader("🚀 Next Steps")
    st.markdown("""
    1. **Train Your Models**: Use your training pipeline to create `.pkl` files
    2. **Test Accuracy**: Upload historical data in the Test Mode
    3. **Make Predictions**: Try the prediction mode with live weather data
    4. **Iterate**: Improve model based on results
    """)

# Footer
st.markdown("---")
st.caption("🏗️ Construction AI Predictor v1.0 | Powered by Machine Learning + Real-time Weather")
