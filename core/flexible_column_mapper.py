"""
Flexible Column Mapper - Handles Different Data Formats
Makes the system work with ANY construction firm's data structure
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class FlexibleColumnMapper:
    """
    Maps various column naming conventions to system requirements.
    Handles missing columns by creating reasonable defaults or deriving from available data.
    """
    
    def __init__(self):
        # Define all possible column name variations
        self.column_mappings = {
            'project_id': ['projectid', 'project_id', 'id', 'job_number', 'job#', 'project_number', 'projectnumber'],
            'project_type': ['project_type', 'projecttype', 'type', 'taskname', 'task_name', 'job_type', 'building_type'],
            'location': ['location', 'site', 'address', 'city', 'borough', 'region'],
            'planned_start_date': ['planned_start_date', 'start_date', 'startdate', 'planned_start', 'schedule_start'],
            'planned_end_date': ['planned_end_date', 'end_date', 'enddate', 'planned_end', 'schedule_end', 'target_end'],
            'actual_end_date': ['actual_end_date', 'enddate', 'actual_end', 'completion_date', 'finished_date'],
            'estimated_cost': ['estimated_cost', 'estimatedcost', 'budget', 'initial_cost', 'planned_cost', 'est_cost'],
            'actual_cost': ['actual_cost', 'actualcost', 'final_cost', 'total_cost', 'spent'],
            'crew_size_avg': ['crew_size_avg', 'averagecrew', 'average_crew', 'crew_size', 'workers', 'labor_count'],
            'total_labor_hours': ['total_labor_hours', 'labor_hours', 'man_hours', 'work_hours', 'hours'],
            'equipment_downtime_hours': ['equipment_downtime_hours', 'downtime', 'delay_hours', 'equipment_delay']
        }
    
    def find_column(self, df, target_column):
        """
        Find a column in the dataframe that matches the target column.
        Returns the actual column name or None.
        """
        possible_names = self.column_mappings.get(target_column, [])
        
        # Check for exact matches (case-insensitive)
        for col in df.columns:
            if col.lower() in [name.lower() for name in possible_names]:
                return col
        
        return None
    
    def map_dataframe(self, df, verbose=True):
        """
        Map dataframe columns to required format.
        Creates missing columns with reasonable defaults.
        """
        if verbose:
            print("\n" + "="*70)
            print("FLEXIBLE COLUMN MAPPING")
            print("="*70)
        
        mapped_df = pd.DataFrame()
        mapping_report = []
        
        # Map each required column
        for required_col in self.column_mappings.keys():
            actual_col = self.find_column(df, required_col)
            
            if actual_col:
                # Column exists - map it
                mapped_df[required_col] = df[actual_col]
                mapping_report.append({
                    'required': required_col,
                    'found': actual_col,
                    'status': 'MAPPED',
                    'method': 'Direct mapping'
                })
                if verbose:
                    print(f"✓ {required_col:<30} <- {actual_col}")
            else:
                # Column missing - create it
                created_col = self._create_missing_column(df, required_col, mapped_df)
                if created_col is not None:
                    mapped_df[required_col] = created_col
                    mapping_report.append({
                        'required': required_col,
                        'found': 'N/A',
                        'status': 'CREATED',
                        'method': 'Derived/Default'
                    })
                    if verbose:
                        print(f"⚠ {required_col:<30} <- CREATED (derived/default)")
                else:
                    mapping_report.append({
                        'required': required_col,
                        'found': 'N/A',
                        'status': 'FAILED',
                        'method': 'Could not create'
                    })
                    if verbose:
                        print(f"✗ {required_col:<30} <- FAILED")
        
        if verbose:
            print("="*70)
            print(f"Mapping complete: {len(mapped_df.columns)}/{len(self.column_mappings)} columns")
            print("="*70)
        
        return mapped_df, mapping_report
    
    def _create_missing_column(self, original_df, column_name, mapped_df):
        """
        Create missing columns using available data or reasonable defaults.
        """
        n_rows = len(original_df)
        
        if column_name == 'project_id':
            # Create sequential IDs if none exist
            return [f"PROJ_{i:04d}" for i in range(1, n_rows + 1)]
        
        elif column_name == 'project_type':
            # Try to infer from task name, notes, or default to 'Commercial'
            if 'TaskName' in original_df.columns:
                return original_df['TaskName']
            elif 'Notes' in original_df.columns:
                return ['Commercial'] * n_rows  # Default
            return ['Commercial'] * n_rows
        
        elif column_name == 'location':
            # Default location if not specified
            return ['Unknown'] * n_rows
        
        elif column_name == 'planned_start_date':
            # Use actual start date or create dates
            if 'StartDate' in original_df.columns:
                return original_df['StartDate']
            return [datetime.now().strftime('%Y-%m-%d')] * n_rows
        
        elif column_name == 'planned_end_date':
            # Calculate from start date + duration - delays
            if 'StartDate' in original_df.columns and 'Duration' in original_df.columns:
                planned_ends = []
                for idx, row in original_df.iterrows():
                    start = pd.to_datetime(row['StartDate'])
                    duration = row.get('Duration', 30)
                    delay = row.get('TotalDelayDays', 0)
                    # Planned end = start + duration - delays (what was originally planned)
                    planned_end = start + timedelta(days=duration - delay)
                    planned_ends.append(planned_end.strftime('%Y-%m-%d'))
                return planned_ends
            return None
        
        elif column_name == 'actual_end_date':
            # Use EndDate if available
            if 'EndDate' in original_df.columns:
                return original_df['EndDate']
            return None
        
        elif column_name == 'estimated_cost':
            # Use EstimatedCost if available
            if 'EstimatedCost' in original_df.columns:
                return original_df['EstimatedCost']
            # Or create estimate based on square footage
            elif 'SquareFootage' in original_df.columns:
                # Rough estimate: $150-300 per sq ft
                return original_df['SquareFootage'] * np.random.uniform(150, 300, n_rows)
            return [100000] * n_rows  # Default estimate
        
        elif column_name == 'actual_cost':
            # Use ActualCost if available
            if 'ActualCost' in original_df.columns:
                return original_df['ActualCost']
            # Or derive from estimated cost with variance
            elif 'estimated_cost' in mapped_df.columns:
                variance = np.random.uniform(0.95, 1.15, n_rows)
                return mapped_df['estimated_cost'] * variance
            return [110000] * n_rows  # Default
        
        elif column_name == 'crew_size_avg':
            # Use AverageCrew or PeakCrew
            if 'AverageCrew' in original_df.columns:
                return original_df['AverageCrew']
            elif 'PeakCrew' in original_df.columns:
                return original_df['PeakCrew'] * 0.7  # Assume avg is 70% of peak
            # Estimate based on square footage
            elif 'SquareFootage' in original_df.columns:
                return np.maximum(5, original_df['SquareFootage'] / 500).astype(int)
            return [20] * n_rows  # Default crew size
        
        elif column_name == 'total_labor_hours':
            # Estimate: crew_size * duration * 8 hours/day
            if 'crew_size_avg' in mapped_df.columns and 'Duration' in original_df.columns:
                return mapped_df['crew_size_avg'] * original_df['Duration'] * 8
            elif 'SquareFootage' in original_df.columns:
                # Rough estimate: 10-20 hours per sq ft
                return original_df['SquareFootage'] * np.random.uniform(10, 20, n_rows)
            return [1000] * n_rows  # Default
        
        elif column_name == 'equipment_downtime_hours':
            # Convert delay days to hours or use permit/material delays
            if 'PermitInspectionDelayDays' in original_df.columns and 'MaterialDelayDays' in original_df.columns:
                total_delay_days = original_df['PermitInspectionDelayDays'] + original_df['MaterialDelayDays']
                return total_delay_days * 8  # Convert to hours (8-hour workday)
            elif 'TotalDelayDays' in original_df.columns:
                return original_df['TotalDelayDays'] * 8
            return [10] * n_rows  # Default minimal downtime
        
        return None
    
    def export_mapping_report(self, mapping_report, output_path='column_mapping_report.csv'):
        """Export mapping report for documentation."""
        df_report = pd.DataFrame(mapping_report)
        df_report.to_csv(output_path, index=False)
        print(f"Mapping report saved to {output_path}")


def convert_dataset(input_file, output_file='sample_projects.csv'):
    """
    Main function to convert any dataset to required format.
    """
    print(f"\nConverting {input_file} to standard format...")
    
    # Load original data
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} rows from {input_file}")
    
    # Map columns
    mapper = FlexibleColumnMapper()
    mapped_df, mapping_report = mapper.map_dataframe(df, verbose=True)
    
    # Save converted data
    mapped_df.to_csv(output_file, index=False)
    print(f"\n✓ Converted data saved to {output_file}")
    
    # Save mapping report
    mapper.export_mapping_report(mapping_report)
    
    # Show sample
    print("\n" + "="*70)
    print("SAMPLE OF CONVERTED DATA (First 3 rows)")
    print("="*70)
    print(mapped_df.head(3))
    
    return mapped_df


if __name__ == "__main__":
    # Convert construction_projects_100.csv to required format
    convert_dataset('construction_projects_100.csv', 'sample_projects.csv')
    
    print("\n" + "="*70)
    print("READY TO USE")
    print("="*70)
    print("\nYou can now run:")
    print("  python main.py")
    print("\nThe system will automatically use sample_projects.csv")