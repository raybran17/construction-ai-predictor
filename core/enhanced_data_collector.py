"""
Enhanced Data Collection Pipeline
Combines: global_data_collector.py + universal_adapter_pipeline.py
Adds: Real data sources, validation, cleaning
"""

import os
import glob
import pandas as pd
import numpy as np
import requests
from typing import Optional, List, Dict, Union
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class EnhancedDataCollector:
    """
    Complete data collection and standardization pipeline.
    
    Features:
    - Load from CSV, Excel, folder of files, or API
    - Auto-detect and map column names
    - Validate data quality
    - Clean and standardize
    - Calculate derived features
    - Connect to real data sources
    """
    
    # Standard schema for construction projects
    STANDARD_SCHEMA = {
        'project_id': 'string',
        'project_name': 'string',
        'project_type': 'category',
        'location': 'string',
        'planned_start_date': 'datetime',
        'planned_end_date': 'datetime',
        'actual_end_date': 'datetime',
        'estimated_cost': 'float',
        'actual_cost': 'float',
        'crew_size_avg': 'int',
        'total_labor_hours': 'float',
        'equipment_downtime_hours': 'float',
        # Derived fields (calculated)
        'calculated_delay_days': 'float',
        'actual_overrun_percent': 'float'
    }
    
    # Column name mapping (handles different naming conventions)
    COLUMN_MAPPINGS = {
        # IDs
        'job_number': 'project_id',
        'job_id': 'project_id',
        'project_number': 'project_id',
        'id': 'project_id',
        
        # Names
        'job_name': 'project_name',
        'name': 'project_name',
        'project': 'project_name',
        
        # Types
        'job_type': 'project_type',
        'type': 'project_type',
        'category': 'project_type',
        
        # Location
        'city': 'location',
        'borough': 'location',
        'address': 'location',
        'site_location': 'location',
        
        # Dates
        'start_date': 'planned_start_date',
        'planned_start': 'planned_start_date',
        'end_date': 'planned_end_date',
        'planned_end': 'planned_end_date',
        'planned_completion': 'planned_end_date',
        'completion_date': 'actual_end_date',
        'actual_end': 'actual_end_date',
        'actual_completion': 'actual_end_date',
        
        # Costs
        'budget': 'estimated_cost',
        'initial_cost': 'estimated_cost',
        'planned_cost': 'estimated_cost',
        'final_cost': 'actual_cost',
        'total_cost': 'actual_cost',
        
        # Labor
        'crew_size': 'crew_size_avg',
        'workers': 'crew_size_avg',
        'labor_hours': 'total_labor_hours',
        'man_hours': 'total_labor_hours',
        
        # Equipment
        'downtime': 'equipment_downtime_hours',
        'equipment_delays': 'equipment_downtime_hours'
    }
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.df = None
        self.validation_report = {}
        self.source_info = {}
        
    # =========================================================================
    # DATA LOADING
    # =========================================================================
    
    def load_from_file(self, filepath: str) -> pd.DataFrame:
        """Load single CSV or Excel file."""
        ext = os.path.splitext(filepath)[1].lower()
        
        try:
            if ext == '.csv':
                df = pd.read_csv(filepath)
            elif ext in ['.xlsx', '.xls']:
                df = pd.read_excel(filepath)
            else:
                raise ValueError(f"Unsupported file type: {ext}")
            
            if self.verbose:
                print(f"✓ Loaded {len(df)} rows from {os.path.basename(filepath)}")
            
            df['_source_file'] = os.path.basename(filepath)
            return df
            
        except Exception as e:
            if self.verbose:
                print(f"✗ Error loading {filepath}: {e}")
            return pd.DataFrame()
    
    def load_from_folder(self, folder: str, pattern: str = "*.csv") -> pd.DataFrame:
        """Load all matching files from folder."""
        files = glob.glob(os.path.join(folder, pattern))
        
        if not files:
            if self.verbose:
                print(f"✗ No files found matching '{pattern}' in {folder}")
            return pd.DataFrame()
        
        if self.verbose:
            print(f"Found {len(files)} files in {folder}")
        
        dfs = []
        for filepath in files:
            df = self.load_from_file(filepath)
            if not df.empty:
                dfs.append(df)
        
        if dfs:
            combined = pd.concat(dfs, ignore_index=True, sort=False)
            if self.verbose:
                print(f"✓ Combined {len(combined)} total rows")
            return combined
        
        return pd.DataFrame()
    
    def load_from_nyc_open_data(self, limit: int = 1000) -> pd.DataFrame:
        """
        Load construction data from NYC Open Data API.
        Dataset: DOB Job Application Filings
        """
        if self.verbose:
            print("Fetching data from NYC Open Data API...")
        
        try:
            # NYC DOB Job Applications API
            url = "https://data.cityofnewyork.us/resource/ic3t-wcy2.json"
            params = {
                '$limit': limit,
                '$where': "job_type IN('A1','A2','NB') AND filing_status='APPROVED'",
                '$order': 'pre__filing_date DESC'
            }
            
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if not data:
                if self.verbose:
                    print("✗ No data returned from API")
                return pd.DataFrame()
            
            df = pd.DataFrame(data)
            df['_source'] = 'NYC_Open_Data'
            
            if self.verbose:
                print(f"✓ Fetched {len(df)} projects from NYC Open Data")
            
            return df
            
        except Exception as e:
            if self.verbose:
                print(f"✗ NYC Open Data API error: {e}")
            return pd.DataFrame()
    
    def load(self, 
             filepath: Optional[str] = None,
             folder: Optional[str] = None,
             use_nyc_data: bool = False,
             nyc_limit: int = 1000) -> 'EnhancedDataCollector':
        """
        Main loading method - handles multiple sources.
        
        Args:
            filepath: Path to single file
            folder: Path to folder of files
            use_nyc_data: Fetch from NYC Open Data
            nyc_limit: Max records from NYC API
        """
        dfs = []
        
        # Load from file
        if filepath:
            df = self.load_from_file(filepath)
            if not df.empty:
                dfs.append(df)
        
        # Load from folder
        if folder:
            df = self.load_from_folder(folder)
            if not df.empty:
                dfs.append(df)
        
        # Load from NYC Open Data
        if use_nyc_data:
            df = self.load_from_nyc_open_data(limit=nyc_limit)
            if not df.empty:
                dfs.append(df)
        
        # Combine all sources
        if dfs:
            self.df = pd.concat(dfs, ignore_index=True, sort=False)
            self.source_info = {
                'total_rows': len(self.df),
                'total_columns': len(self.df.columns),
                'sources': self.df.get('_source_file', pd.Series(['Unknown'])).unique().tolist()
            }
            
            if self.verbose:
                print(f"\n{'='*60}")
                print(f"DATA LOADING SUMMARY")
                print(f"{'='*60}")
                print(f"Total projects loaded: {len(self.df)}")
                print(f"Total columns: {len(self.df.columns)}")
                print(f"Sources: {', '.join(self.source_info['sources'][:3])}")
                print(f"{'='*60}\n")
        else:
            if self.verbose:
                print("✗ No data loaded from any source")
            self.df = pd.DataFrame()
        
        return self
    
    # =========================================================================
    # COLUMN STANDARDIZATION
    # =========================================================================
    
    def _normalize_column_name(self, col: str) -> str:
        """Normalize column name (lowercase, underscores)."""
        return col.strip().lower().replace(' ', '_').replace('-', '_')
    
    def standardize_columns(self) -> 'EnhancedDataCollector':
        """Map dataset-specific column names to standard schema."""
        if self.df is None or self.df.empty:
            return self
        
        # Normalize all column names first
        normalized_cols = {col: self._normalize_column_name(col) for col in self.df.columns}
        self.df.rename(columns=normalized_cols, inplace=True)
        
        # Apply mappings
        rename_dict = {}
        for col in self.df.columns:
            if col in self.COLUMN_MAPPINGS:
                rename_dict[col] = self.COLUMN_MAPPINGS[col]
        
        if rename_dict:
            self.df.rename(columns=rename_dict, inplace=True)
            if self.verbose:
                print(f"✓ Standardized {len(rename_dict)} column names")
        
        return self
    
    # =========================================================================
    # DATA VALIDATION
    # =========================================================================
    
    def validate(self) -> 'EnhancedDataCollector':
        """Validate data quality and completeness."""
        if self.df is None or self.df.empty:
            return self
        
        self.validation_report = {
            'total_rows': len(self.df),
            'issues': [],
            'warnings': [],
            'quality_score': 100
        }
        
        # Check for required columns
        required = ['project_id', 'estimated_cost', 'actual_cost']
        missing_required = [col for col in required if col not in self.df.columns]
        
        if missing_required:
            self.validation_report['issues'].append(
                f"Missing required columns: {missing_required}"
            )
            self.validation_report['quality_score'] -= 30
        
        # Check for missing data
        if 'estimated_cost' in self.df.columns:
            missing_cost = self.df['estimated_cost'].isna().sum()
            if missing_cost > 0:
                pct = (missing_cost / len(self.df)) * 100
                self.validation_report['warnings'].append(
                    f"{missing_cost} projects ({pct:.1f}%) missing estimated_cost"
                )
                self.validation_report['quality_score'] -= min(20, pct)
        
        # Check for date columns
        date_cols = ['planned_start_date', 'planned_end_date', 'actual_end_date']
        for col in date_cols:
            if col in self.df.columns:
                try:
                    pd.to_datetime(self.df[col], errors='coerce')
                except:
                    self.validation_report['warnings'].append(
                        f"Date column '{col}' has invalid dates"
                    )
        
        # Check for negative values
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            negative_count = (self.df[col] < 0).sum()
            if negative_count > 0:
                self.validation_report['warnings'].append(
                    f"{col} has {negative_count} negative values"
                )
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"DATA VALIDATION")
            print(f"{'='*60}")
            print(f"Quality Score: {self.validation_report['quality_score']:.0f}/100")
            
            if self.validation_report['issues']:
                print(f"\n⚠️  ISSUES:")
                for issue in self.validation_report['issues']:
                    print(f"  - {issue}")
            
            if self.validation_report['warnings']:
                print(f"\n⚠️  WARNINGS:")
                for warning in self.validation_report['warnings'][:5]:
                    print(f"  - {warning}")
            
            print(f"{'='*60}\n")
        
        return self
    
    # =========================================================================
    # DATA CLEANING
    # =========================================================================
    
    def clean(self) -> 'EnhancedDataCollector':
        """Clean and prepare data for modeling."""
        if self.df is None or self.df.empty:
            return self
        
        original_rows = len(self.df)
        
        # Convert date columns
        date_cols = ['planned_start_date', 'planned_end_date', 'actual_end_date']
        for col in date_cols:
            if col in self.df.columns:
                self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
        
        # Convert numeric columns
        numeric_cols = ['estimated_cost', 'actual_cost', 'crew_size_avg', 
                       'total_labor_hours', 'equipment_downtime_hours']
        for col in numeric_cols:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        
        # Calculate derived fields
        if 'planned_end_date' in self.df.columns and 'actual_end_date' in self.df.columns:
            self.df['calculated_delay_days'] = (
                self.df['actual_end_date'] - self.df['planned_end_date']
            ).dt.days.clip(lower=0)
        
        if 'estimated_cost' in self.df.columns and 'actual_cost' in self.df.columns:
            self.df['actual_overrun_percent'] = (
                (self.df['actual_cost'] - self.df['estimated_cost']) / 
                self.df['estimated_cost']
            ) * 100
        
        # Create project_id if missing
        if 'project_id' not in self.df.columns or self.df['project_id'].isna().all():
            self.df['project_id'] = [f"PROJ_{i:05d}" for i in range(len(self.df))]
        
        # Fill missing values strategically
        if 'crew_size_avg' in self.df.columns:
            median_crew = self.df['crew_size_avg'].median()
            self.df['crew_size_avg'].fillna(median_crew, inplace=True)
        
        if 'equipment_downtime_hours' in self.df.columns:
            self.df['equipment_downtime_hours'].fillna(0, inplace=True)
        
        # Remove rows with critical missing data
        critical_cols = ['estimated_cost', 'actual_cost']
        before_drop = len(self.df)
        self.df.dropna(subset=[c for c in critical_cols if c in self.df.columns], 
                      inplace=True)
        dropped = before_drop - len(self.df)
        
        if self.verbose and dropped > 0:
            print(f"⚠️  Dropped {dropped} rows with missing critical data")
        
        if self.verbose:
            print(f"✓ Data cleaning complete: {len(self.df)}/{original_rows} rows retained")
        
        return self
    
    # =========================================================================
    # OUTPUT
    # =========================================================================
    
    def get_data(self) -> pd.DataFrame:
        """Return cleaned, standardized DataFrame."""
        if self.df is None:
            return pd.DataFrame()
        return self.df.copy()
    
    def save(self, filepath: str = "cleaned_projects.csv") -> 'EnhancedDataCollector':
        """Save cleaned data to file."""
        if self.df is None or self.df.empty:
            print("✗ No data to save")
            return self
        
        self.df.to_csv(filepath, index=False)
        if self.verbose:
            print(f"✓ Saved {len(self.df)} projects to {filepath}")
        
        return self
    
    def get_validation_report(self) -> Dict:
        """Return validation report."""
        return self.validation_report
    
    # =========================================================================
    # CONVENIENCE METHOD (Complete Pipeline)
    # =========================================================================
    
    def process(self,
                filepath: Optional[str] = None,
                folder: Optional[str] = None,
                use_nyc_data: bool = False) -> pd.DataFrame:
        """
        Complete pipeline: Load → Standardize → Validate → Clean
        
        Usage:
            collector = EnhancedDataCollector()
            df = collector.process(filepath="my_projects.csv")
        """
        self.load(filepath=filepath, folder=folder, use_nyc_data=use_nyc_data)
        self.standardize_columns()
        self.validate()
        self.clean()
        
        return self.get_data()


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    collector = EnhancedDataCollector(verbose=True)
    
    # Example 1: Load from file
    df = collector.process(filepath="sample_projects.txt")
    print(f"\nFinal dataset: {df.shape[0]} rows × {df.shape[1]} columns")
    print(df.head())
    
    # Example 2: Load from NYC Open Data (requires internet)
    # df = collector.process(use_nyc_data=True, nyc_limit=100)
    
    # Example 3: Load from folder
    # df = collector.process(folder="data/raw_projects/")
    
    # Save cleaned data
    # collector.save("data/cleaned_projects.csv")