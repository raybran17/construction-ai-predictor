import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class UniversalDataAdapter:
    """
    Pipeline 1: Universal Data Adapter
    
    Translates ANY construction CSV format into standardized format
    that downstream pipelines (Delay/Cost engines) can process.
    
    This is the SECRET SAUCE that lets you work with ANY PM's data.
    """
    
    def __init__(self, weather_api_key=None):
        self.weather_api_key = weather_api_key
        self.column_mapping = {}
        self.detected_columns = {}
        self.missing_columns = []
        self.confidence_scores = {}
        
        # Standard column names our models expect
        self.standard_schema = {
            'project_id': ['project_id', 'id', 'project_number', 'job_number', 'project'],
            'project_type': ['project_type', 'type', 'category', 'project_category'],
            'location': ['location', 'site', 'address', 'city', 'project_location'],
            'start_date': ['start_date', 'start', 'begin_date', 'project_start', 'commenced'],
            'end_date': ['end_date', 'finish', 'completion_date', 'project_end'],
            'planned_duration': ['planned_duration', 'duration', 'planned_days', 'estimated_duration'],
            'actual_duration': ['actual_duration', 'real_duration', 'completed_duration'],
            'crew_size': ['crew_size', 'workers', 'worker_count', 'team_size', 'labor_count'],
            'rain_days': ['rain_days', 'weather_days', 'rain', 'weather_delay'],
            'material_delay_days': ['material_delay_days', 'material_delay', 'supply_delay'],
            'inspections': ['inspections_passed', 'inspections', 'inspection_count'],
            'supplier_reliability': ['supplier_reliability', 'supplier_score', 'vendor_rating'],
            'weather_severity': ['weather_severity', 'weather_score', 'weather_impact'],
            'project_delay_days': ['project_delay_days', 'delay', 'total_delay', 'time_deviation'],
            'planned_cost': ['planned_cost', 'budget', 'estimated_cost', 'planned_budget'],
            'actual_cost': ['actual_cost', 'real_cost', 'final_cost', 'spent'],
            'cost_overrun': ['cost_overrun_pct', 'cost_overrun', 'cost_deviation', 'budget_variance']
        }
    
    def load_data(self, filepath_or_df):
        """Load data from CSV file or DataFrame."""
        if isinstance(filepath_or_df, str):
            print(f"📂 Loading data from {filepath_or_df}...")
            self.raw_df = pd.read_csv(filepath_or_df)
        else:
            self.raw_df = filepath_or_df.copy()
        
        print(f"✅ Loaded {len(self.raw_df)} rows with {len(self.raw_df.columns)} columns")
        print(f"📋 Columns found: {', '.join(self.raw_df.columns.tolist())}")
        return self
    
    def detect_column_mapping(self):
        """
        Smart column detection - maps user's columns to standard schema.
        Uses fuzzy matching and pattern recognition.
        """
        print("\n🔍 Detecting column mappings...")
        
        user_columns = [col.lower().strip().replace(' ', '_') for col in self.raw_df.columns]
        
        for standard_col, possible_names in self.standard_schema.items():
            matched = False
            best_match = None
            best_score = 0
            
            for user_col_original, user_col in zip(self.raw_df.columns, user_columns):
                # Direct match
                if user_col in possible_names:
                    self.column_mapping[standard_col] = user_col_original
                    self.confidence_scores[standard_col] = 100
                    matched = True
                    break
                
                # Partial match (fuzzy)
                for possible_name in possible_names:
                    if possible_name in user_col or user_col in possible_name:
                        score = len(set(user_col) & set(possible_name)) / len(set(user_col) | set(possible_name)) * 100
                        if score > best_score:
                            best_score = score
                            best_match = user_col_original
            
            if not matched and best_match and best_score > 40:
                self.column_mapping[standard_col] = best_match
                self.confidence_scores[standard_col] = best_score
                matched = True
            
            if not matched:
                self.missing_columns.append(standard_col)
        
        print(f"✅ Mapped {len(self.column_mapping)} columns")
        print(f"⚠️  Missing {len(self.missing_columns)} columns: {', '.join(self.missing_columns)}")
        
        return self
    
    def show_mapping_report(self):
        """Display what was detected and what's missing."""
        print("\n" + "="*70)
        print("COLUMN MAPPING REPORT")
        print("="*70)
        
        print("\n✅ DETECTED COLUMNS:")
        for standard, user_col in self.column_mapping.items():
            confidence = self.confidence_scores.get(standard, 0)
            status = "🟢" if confidence == 100 else "🟡" if confidence > 70 else "🟠"
            print(f"  {status} {standard:25s} ← {user_col:30s} ({confidence:.0f}% confidence)")
        
        print(f"\n⚠️  MISSING COLUMNS ({len(self.missing_columns)}):")
        for col in self.missing_columns:
            print(f"  ❌ {col:25s} (will be estimated/fetched)")
        
        print("="*70)
        return self
    
    def standardize_data(self):
        """Create standardized DataFrame with mapped columns."""
        print("\n🔄 Creating standardized dataset...")
        
        self.standard_df = pd.DataFrame()
        
        # Map detected columns
        for standard_col, user_col in self.column_mapping.items():
            self.standard_df[standard_col] = self.raw_df[user_col]
        
        # Add placeholder columns for missing data
        for missing_col in self.missing_columns:
            self.standard_df[missing_col] = np.nan
        
        print(f"✅ Standardized dataset created with {len(self.standard_df.columns)} columns")
        return self
    
    def infer_missing_data(self):
        """
        Intelligently fill missing columns using:
        1. Calculations from other columns
        2. Industry defaults
        3. API calls (weather)
        4. Pattern recognition
        """
        print("\n🧠 Inferring missing data...")
        filled_count = 0
        
        # Calculate duration if we have dates
        if 'planned_duration' in self.missing_columns:
            if 'start_date' in self.column_mapping and 'end_date' in self.column_mapping:
                try:
                    start = pd.to_datetime(self.standard_df['start_date'])
                    end = pd.to_datetime(self.standard_df['end_date'])
                    self.standard_df['planned_duration'] = (end - start).dt.days
                    self.missing_columns.remove('planned_duration')
                    filled_count += 1
                    print("  ✓ Calculated planned_duration from dates")
                except:
                    pass
        
        # Calculate delay if we have actual and planned duration
        if 'project_delay_days' in self.missing_columns:
            if 'actual_duration' in self.column_mapping and 'planned_duration' in self.standard_df.columns:
                self.standard_df['project_delay_days'] = (
                    self.standard_df['actual_duration'] - self.standard_df['planned_duration']
                )
                self.standard_df['project_delay_days'] = self.standard_df['project_delay_days'].clip(lower=0)
                self.missing_columns.remove('project_delay_days')
                filled_count += 1
                print("  ✓ Calculated project_delay_days from durations")
        
        # Calculate cost overrun if we have costs
        if 'cost_overrun' in self.missing_columns:
            if 'actual_cost' in self.column_mapping and 'planned_cost' in self.column_mapping:
                self.standard_df['cost_overrun'] = (
                    (self.standard_df['actual_cost'] - self.standard_df['planned_cost']) / 
                    self.standard_df['planned_cost'] * 100
                )
                self.missing_columns.remove('cost_overrun')
                filled_count += 1
                print("  ✓ Calculated cost_overrun from costs")
        
        # Set reasonable defaults for missing operational data
        defaults = {
            'crew_size': 25,
            'inspections': 5,
            'supplier_reliability': 80,
            'weather_severity': 2,
            'rain_days': 5,
            'material_delay_days': 3
        }
        
        for col, default_value in defaults.items():
            if col in self.missing_columns and self.standard_df[col].isnull().all():
                self.standard_df[col] = default_value
                print(f"  ⚙️  Set {col} to default: {default_value}")
                filled_count += 1
        
        # Infer project type from cost if missing
        if 'project_type' in self.missing_columns and 'planned_cost' in self.standard_df.columns:
            def categorize_by_cost(cost):
                if pd.isna(cost):
                    return 'Unknown'
                elif cost < 500000:
                    return 'Residential'
                elif cost < 3000000:
                    return 'Commercial'
                elif cost < 10000000:
                    return 'Industrial'
                else:
                    return 'Infrastructure'
            
            self.standard_df['project_type'] = self.standard_df['planned_cost'].apply(categorize_by_cost)
            filled_count += 1
            print("  ✓ Inferred project_type from cost ranges")
        
        print(f"✅ Filled/inferred {filled_count} missing columns")
        return self
    
    def fetch_weather_data(self):
        """
        Fetch weather data for projects with location + dates but missing weather info.
        Only runs if weather API key is provided.
        """
        if not self.weather_api_key:
            print("\n⚠️  No weather API key provided - skipping weather enrichment")
            return self
        
        print("\n☁️  Fetching weather data...")
        
        needs_weather = (
            'location' in self.column_mapping and 
            'start_date' in self.column_mapping and
            ('rain_days' in self.missing_columns or 'weather_severity' in self.missing_columns)
        )
        
        if needs_weather:
            try:
                # Import weather module
                from weather_api_module import WeatherDataFetcher
                fetcher = WeatherDataFetcher(self.weather_api_key)
                
                # Sample first location to test
                sample_location = self.standard_df['location'].iloc[0]
                weather = fetcher.get_construction_forecast(sample_location, project_duration_days=30)
                
                if 'error' not in weather:
                    print(f"  ✓ Weather API working - fetched data for {sample_location}")
                    # In production, would fetch for all unique locations
                    # For now, use sample data as proxy
                    if 'rain_days' in self.missing_columns:
                        self.standard_df['rain_days'] = weather['construction_risks']['rain_days']
                    if 'weather_severity' in self.missing_columns:
                        self.standard_df['weather_severity'] = weather['construction_risks']['weather_severity']
                else:
                    print(f"  ⚠️  Weather API error: {weather['error']}")
            except Exception as e:
                print(f"  ⚠️  Could not fetch weather: {e}")
        else:
            print("  ℹ️  Weather data not needed or already present")
        
        return self
    
    def validate_and_clean(self):
        """Final validation and cleaning of standardized data."""
        print("\n🧹 Validating and cleaning data...")
        
        # Remove rows with too many missing values
        threshold = len(self.standard_df.columns) * 0.7  # Need at least 70% data
        before_count = len(self.standard_df)
        self.standard_df = self.standard_df.dropna(thresh=threshold)
        after_count = len(self.standard_df)
        
        if before_count != after_count:
            print(f"  ⚠️  Removed {before_count - after_count} rows with insufficient data")
        
        # Convert data types
        numeric_cols = ['crew_size', 'rain_days', 'material_delay_days', 'inspections',
                       'supplier_reliability', 'weather_severity', 'project_delay_days',
                       'planned_cost', 'actual_cost', 'cost_overrun']
        
        for col in numeric_cols:
            if col in self.standard_df.columns:
                self.standard_df[col] = pd.to_numeric(self.standard_df[col], errors='coerce')
        
        # Standardize dates
        date_cols = ['start_date', 'end_date']
        for col in date_cols:
            if col in self.standard_df.columns:
                self.standard_df[col] = pd.to_datetime(self.standard_df[col], errors='coerce')
        
        print(f"✅ Validation complete - {len(self.standard_df)} rows ready")
        return self
    
    def get_standardized_data(self):
        """Return the clean, standardized DataFrame."""
        return self.standard_df
    
    def run_full_adapter(self, filepath_or_df, weather_api_key=None):
        """
        Execute complete Universal Adapter pipeline.
        
        Input: ANY construction CSV
        Output: Standardized, clean, ready-for-ML DataFrame
        """
        print("\n" + "🚀 "+"="*68)
        print("UNIVERSAL DATA ADAPTER - PIPELINE 1")
        print("="*70)
        
        if weather_api_key:
            self.weather_api_key = weather_api_key
        
        (self.load_data(filepath_or_df)
            .detect_column_mapping()
            .show_mapping_report()
            .standardize_data()
            .infer_missing_data()
            .fetch_weather_data()
            .validate_and_clean())
        
        print("\n" + "="*70)
        print("✅ ADAPTER COMPLETE - Data is standardized and ready!")
        print("="*70)
        print(f"\n📊 Output shape: {self.standard_df.shape}")
        print(f"📋 Columns: {', '.join(self.standard_df.columns.tolist())}")
        
        return self


# Example Usage
if __name__ == "__main__":
    # Initialize adapter
    adapter = UniversalDataAdapter()
    
    # Run on ANY construction CSV
    adapter.run_full_adapter('your_messy_construction_data.csv')
    
    # Get clean, standardized data
    clean_data = adapter.get_standardized_data()
    
    # Now ready to feed into Delay/Cost engines!
    print("\n✨ Ready for Pipeline 2a (Delay Engine) and 2b (Cost Engine)!")
    print(clean_data.head())
