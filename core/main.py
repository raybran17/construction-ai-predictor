"""
Construction Project Analytics System - Production Entry Point
Zero-error, presentation-ready version with automatic data handling
"""

import pandas as pd
import sys
import os
import glob

# Import system components
try:
    from enhanced_data_collector import EnhancedDataCollector
except ImportError:
    from global_data_collector import GlobalDataCollector as EnhancedDataCollector

from delay_cost_engines import DelayEngineV2, CostEngineV2
from integrated_master_pipeline import IntegratedMasterPipeline


def find_best_data_file():
    """
    Intelligently find the best available data file.
    Priority: mapped files > synthetic > raw data > any CSV
    """
    # Priority 1: Mapped/cleaned files (ready to use)
    priority_files = [
        "sample_projects.csv",
        "cleaned_projects.csv",
        "mapped_projects.csv",
    ]
    
    for f in priority_files:
        if os.path.exists(f):
            return f, "mapped"
    
    # Priority 2: Known synthetic data
    synthetic_files = [
        "construction_projects_synthetic_data.csv",
        "synthetic_data.csv",
    ]
    
    for f in synthetic_files:
        if os.path.exists(f):
            return f, "synthetic"
    
    # Priority 3: Known project files
    known_files = [
        "construction_projects_100.csv",
        "construction_projects.csv",
        "projects.csv",
        "project_data.csv",
    ]
    
    for f in known_files:
        if os.path.exists(f):
            return f, "raw"
    
    # Priority 4: Any CSV in current directory
    csv_files = glob.glob("*.csv")
    if csv_files:
        return csv_files[0], "unknown"
    
    # Priority 5: Check archive folder
    if os.path.exists("archive"):
        archive_files = glob.glob("archive/*.csv")
        if archive_files:
            return archive_files[0], "archive"
    
    return None, None


def auto_map_columns(df):
    """
    Automatically map columns to required format.
    Embedded column mapper - no external dependencies.
    """
    required_columns = {
        'project_id': ['projectid', 'project_id', 'id', 'job_number', 'job#'],
        'project_type': ['project_type', 'type', 'taskname', 'task_name'],
        'location': ['location', 'site', 'address', 'city', 'borough'],
        'planned_start_date': ['planned_start_date', 'startdate', 'start_date'],
        'planned_end_date': ['planned_end_date', 'enddate', 'end_date', 'target_end'],
        'actual_end_date': ['actual_end_date', 'enddate', 'completion_date'],
        'estimated_cost': ['estimated_cost', 'estimatedcost', 'budget', 'initial_cost'],
        'actual_cost': ['actual_cost', 'actualcost', 'final_cost', 'total_cost'],
        'crew_size_avg': ['crew_size_avg', 'averagecrew', 'average_crew', 'crew_size'],
        'total_labor_hours': ['total_labor_hours', 'labor_hours', 'man_hours'],
        'equipment_downtime_hours': ['equipment_downtime_hours', 'downtime', 'delay_hours']
    }
    
    mapped_df = pd.DataFrame()
    mapping_success = []
    
    # Map existing columns
    for required, variations in required_columns.items():
        found = False
        for col in df.columns:
            if col.lower() in [v.lower() for v in variations]:
                mapped_df[required] = df[col]
                found = True
                mapping_success.append(f"  ✓ {required} <- {col}")
                break
        
        if not found:
            # Create missing column with intelligent defaults
            if required == 'project_id':
                mapped_df[required] = [f"PROJ_{i:04d}" for i in range(1, len(df) + 1)]
            elif required == 'project_type':
                mapped_df[required] = df.get('TaskName', ['Commercial'] * len(df))
            elif required == 'location':
                mapped_df[required] = df.get('Location', ['Unknown'] * len(df))
            elif required == 'planned_start_date':
                mapped_df[required] = df.get('StartDate', pd.Timestamp.now().strftime('%Y-%m-%d'))
            elif required == 'planned_end_date':
                if 'StartDate' in df.columns and 'Duration' in df.columns:
                    mapped_df[required] = pd.to_datetime(df['StartDate']) + pd.to_timedelta(df['Duration'], unit='D')
                else:
                    mapped_df[required] = pd.Timestamp.now().strftime('%Y-%m-%d')
            elif required == 'actual_end_date':
                mapped_df[required] = df.get('EndDate', mapped_df.get('planned_end_date'))
            elif required == 'estimated_cost':
                mapped_df[required] = df.get('EstimatedCost', 100000)
            elif required == 'actual_cost':
                mapped_df[required] = df.get('ActualCost', mapped_df.get('estimated_cost', 100000) * 1.05)
            elif required == 'crew_size_avg':
                mapped_df[required] = df.get('AverageCrew', 20)
            elif required == 'total_labor_hours':
                if 'crew_size_avg' in mapped_df.columns and 'Duration' in df.columns:
                    mapped_df[required] = mapped_df['crew_size_avg'] * df.get('Duration', 30) * 8
                else:
                    mapped_df[required] = 1000
            elif required == 'equipment_downtime_hours':
                if 'PermitInspectionDelayDays' in df.columns and 'MaterialDelayDays' in df.columns:
                    mapped_df[required] = (df['PermitInspectionDelayDays'] + df['MaterialDelayDays']) * 8
                else:
                    mapped_df[required] = 10
            
            mapping_success.append(f"  ⚠ {required} <- CREATED (default)")
    
    if mapping_success:
        print("\nCOLUMN MAPPING:")
        for msg in mapping_success:
            print(msg)
        print()
    
    return mapped_df


def main():
    """Main execution pipeline - bulletproof for presentations."""
    
    print("\n" + "="*70)
    print("CONSTRUCTION PROJECT ANALYTICS SYSTEM")
    print("="*70)
    print("Predictive Analytics with Prescriptive Recommendations\n")
    
    try:
        # =====================================================================
        # STEP 1: FIND AND LOAD DATA (AUTOMATIC)
        # =====================================================================
        
        print("STEP 1: Loading and validating project data...")
        
        data_file, file_type = find_best_data_file()
        
        if data_file is None:
            print("\nERROR: No CSV files found in current directory.")
            print("\nTo use this system:")
            print("  1. Place a CSV file with project data in this folder")
            print("  2. Run: python main.py")
            print("\nThe system will automatically detect and process your data.")
            return
        
        print(f"Found data file: {data_file} (type: {file_type})")
        
        # Load the raw data
        df_raw = pd.read_csv(data_file)
        print(f"Loaded {len(df_raw)} rows with {len(df_raw.columns)} columns")
        
        # Auto-map columns if needed
        if file_type in ['raw', 'unknown', 'archive']:
            print("\nAuto-mapping columns to standard format...")
            df = auto_map_columns(df_raw)
            print(f"Mapped to {len(df.columns)} standard columns")
        else:
            # Already in correct format
            df = df_raw
        
        if df.empty:
            print("\nERROR: No valid data after processing.")
            return
        
        print(f"\nSuccessfully prepared {len(df)} projects for analysis")
        
        # =====================================================================
        # STEP 2: FEATURE ENGINEERING
        # =====================================================================
        
        print("\nSTEP 2: Engineering features and preparing data...")
        
        delay_engine = DelayEngineV2()
        cost_engine = CostEngineV2()
        
        delay_engine.process(df)
        df_with_delay = delay_engine.df.copy()
        
        cost_engine.process(df_with_delay)
        
        X_delay, y_delay = delay_engine.get_features()
        X_cost, y_cost = cost_engine.get_features()
        
        print(f"Prepared {X_delay.shape[1]} delay features and {X_cost.shape[1]} cost features")
        
        # =====================================================================
        # STEP 3: MODEL TRAINING
        # =====================================================================
        
        print("\nSTEP 3: Training predictive models...")
        
        pipeline = IntegratedMasterPipeline()
        
        pipeline.train_delay_model(X_delay, y_delay)
        pipeline.train_cost_model(X_cost, y_cost)
        
        # =====================================================================
        # STEP 4: GENERATE PREDICTIONS
        # =====================================================================
        
        print("\nSTEP 4: Generating predictions and recommendations...")
        
        pipeline.predict_with_details(
            X_delay, X_cost, 
            delay_engine, cost_engine,
            project_data=cost_engine.df
        )
        
        # =====================================================================
        # STEP 5: DISPLAY SAMPLE RESULTS
        # =====================================================================
        
        print("\n" + "="*70)
        print("SAMPLE PROJECT ANALYSIS")
        print("="*70)
        
        report_output = pipeline.generate_dashboard_output(project_idx=0)
        print(report_output)
        
        # =====================================================================
        # STEP 6: EXPORT RESULTS
        # =====================================================================
        
        print("\nSTEP 6: Exporting results...")
        
        pipeline.export_results(output_path="predictions_with_recommendations.csv")
        pipeline.save_models()
        
        # =====================================================================
        # STEP 7: SUMMARY
        # =====================================================================
        
        print("\n" + "="*70)
        print("SYSTEM PERFORMANCE SUMMARY")
        print("="*70)
        
        metrics = pipeline.get_metrics()
        
        if 'delay' in metrics and 'test' in metrics['delay']:
            print("\nDelay Prediction Model:")
            print(f"  Mean Absolute Error: {metrics['delay']['test']['mae']:.2f} days")
            print(f"  Root Mean Squared Error: {metrics['delay']['test']['rmse']:.2f} days")
            if not pd.isna(metrics['delay']['test'].get('r2')):
                print(f"  R-Squared Score: {metrics['delay']['test']['r2']:.3f}")
        
        if 'cost' in metrics and 'test' in metrics['cost']:
            print("\nCost Prediction Model:")
            print(f"  Mean Absolute Error: {metrics['cost']['test']['mae']:.2f}%")
            print(f"  Root Mean Squared Error: {metrics['cost']['test']['rmse']:.2f}%")
            if not pd.isna(metrics['cost']['test'].get('r2')):
                print(f"  R-Squared Score: {metrics['cost']['test']['r2']:.3f}")
        
        predictions = pipeline.get_predictions()
        print(f"\nPredictions Generated: {predictions['count']} projects")
        print(f"Average Predicted Delay: {predictions['delays'].mean():.1f} days")
        print(f"Average Cost Impact: {predictions['costs'].mean():.1f}%")
        
        print("\n" + "="*70)
        print("ANALYSIS COMPLETED SUCCESSFULLY")
        print("="*70)
        print("\nOutput Files:")
        print("  - predictions_with_recommendations.csv")
        print("  - models/delay_model.pkl")
        print("  - models/cost_model.pkl")
        print("="*70 + "\n")
        
    except KeyboardInterrupt:
        print("\n\nProcess interrupted by user")
    except Exception as e:
        print(f"\n\nERROR: {str(e)}")
        print("\nIf this error persists:")
        print("  1. Verify your CSV file has project data")
        print("  2. Check column names match expected format")
        print("  3. Run: python main.py --help for more info")
        
        import traceback
        print("\nDetailed error information:")
        traceback.print_exc()


def demo_mode():
    """Demo mode with sample data."""
    print("\n" + "="*70)
    print("DEMO MODE - Construction Analytics System")
    print("="*70)
    print("\nDemonstrating system capabilities with available data.\n")
    main()


def help_info():
    """Display help information."""
    print("""
Construction Project Analytics System - Help

USAGE:
    python main.py              Run with any available CSV file
    python main.py --demo       Run demo mode
    python main.py --help       Show this help

HOW IT WORKS:
    1. Place any CSV file with construction project data in this folder
    2. Run: python main.py
    3. The system automatically detects, maps, and analyzes your data

DATA REQUIREMENTS:
    The system works best with these columns (but adapts to what you have):
    - Project ID, type, and location
    - Start and end dates (planned and actual)
    - Costs (estimated and actual)
    - Crew size and labor hours
    - Delay information

FEATURES:
    - Automatic column mapping (no data reformatting needed)
    - Delay prediction (days beyond schedule)
    - Cost overrun prediction (% over budget)
    - Risk assessment and classification
    - Actionable recommendations

OUTPUT:
    - predictions_with_recommendations.csv (detailed results)
    - models/ folder (trained ML models)
    - Console report (executive summary)

For more information or support, contact your system administrator.
    """)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        
        if arg in ['--help', '-h', 'help']:
            help_info()
        elif arg in ['--demo', '-d', 'demo']:
            demo_mode()
        else:
            print(f"Unknown argument: {arg}")
            print("Use --help for usage information")
    else:
        main()