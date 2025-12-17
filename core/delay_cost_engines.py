import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder


class DelayEngineV2:
    """
    Delay prediction engine for construction projects.
    Analyzes historical project data to identify delay patterns and risk factors.
    """

    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.feature_names = []
        self.df = None

    def process(self, df):
        """Process input dataframe and prepare features for modeling."""
        print("\n" + "=" * 60)
        print("DELAY ENGINE V2 - Core Delay Prediction")
        print("=" * 60)

        self.df = df.copy()
        self._create_delay_features()
        self._encode_categoricals()
        self._scale_features()
        self._select_features()

        print(f"Delay engine processed {self.df.shape[0]} projects")
        return self

    def _create_delay_features(self):
        """Feature engineering for core project delays."""
        # Calculate actual delay in days
        self.df["calculated_delay_days"] = (
            pd.to_datetime(self.df["actual_end_date"]) - 
            pd.to_datetime(self.df["planned_end_date"])
        ).dt.days.clip(lower=0)

        # Crew efficiency metric
        self.df["crew_efficiency"] = (
            self.df["total_labor_hours"] / (self.df["crew_size_avg"] + 1)
        ).round(2)

        # Delay risk composite index
        self.df["delay_risk_index"] = (
            0.5 * self.df["equipment_downtime_hours"] / (self.df["crew_size_avg"] + 1)
            + 0.001 * self.df["total_labor_hours"]
        ).round(2)

    def _encode_categoricals(self):
        """Encode categorical columns for machine learning."""
        cat_cols = ["project_type", "location"]
        for col in cat_cols:
            if col in self.df.columns:
                le = LabelEncoder()
                self.df[f"{col}_enc"] = le.fit_transform(self.df[col].fillna("Unknown"))
                self.encoders[col] = le

    def _scale_features(self):
        """Standardize numeric features for modeling."""
        num_cols = ["crew_size_avg", "equipment_downtime_hours", "total_labor_hours"]
        for col in num_cols:
            if col in self.df.columns:
                scaler = StandardScaler()
                self.df[f"{col}_scaled"] = scaler.fit_transform(self.df[[col]])
                self.scalers[col] = scaler

    def _select_features(self):
        """Select final feature set for modeling."""
        self.feature_names = [
            "crew_size_avg_scaled",
            "equipment_downtime_hours_scaled",
            "total_labor_hours_scaled",
            "crew_efficiency",
            "delay_risk_index"
        ]
        self.X = self.df[self.feature_names]
        self.y = self.df["calculated_delay_days"]

    def get_features(self):
        """Return feature matrix and target variable."""
        return self.X, self.y

    def get_all_phase_analysis(self):
        """
        Generate phase-level delay analysis for all projects.
        
        Returns:
            dict: Phase analysis indexed by project with delay breakdowns
        """
        if self.df is None or self.df.empty:
            return {}

        phase_analysis = {}
        
        for idx, row in self.df.iterrows():
            delay = row.get('calculated_delay_days', 0)
            crew_size = row.get('crew_size_avg', 20)
            equipment_downtime = row.get('equipment_downtime_hours', 0)
            
            # Distribute delays across construction phases based on historical patterns
            # These ratios are industry averages that would be refined with real data
            foundation_delay = delay * 0.25
            framing_delay = delay * 0.40  
            finishing_delay = delay * 0.35
            
            # Identify primary delay drivers based on project characteristics
            drivers = []
            if equipment_downtime > 20:
                drivers.append("Equipment availability and downtime")
            if crew_size < 15:
                drivers.append("Insufficient labor resources")
            if delay > 15:
                drivers.append("Material delivery and procurement delays")
            if not drivers:
                drivers.append("Normal project schedule variance")
            
            phase_analysis[idx] = {
                'foundation': {
                    'delay_days': round(foundation_delay, 1),
                    'drivers': [drivers[0]] if drivers else ["Site conditions and weather"]
                },
                'framing': {
                    'delay_days': round(framing_delay, 1),
                    'drivers': drivers[:2] if len(drivers) > 1 else drivers
                },
                'finishing': {
                    'delay_days': round(finishing_delay, 1),
                    'drivers': [drivers[-1]] if drivers else ["Inspection scheduling"]
                }
            }

        print("Phase-level delay analysis completed.")
        return phase_analysis


class CostEngineV2:
    """
    Cost overrun prediction engine for construction projects.
    Analyzes budget vs actual costs to identify overrun patterns.
    """

    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.feature_names = []
        self.df = None

    def process(self, df):
        """Process input dataframe and prepare cost features."""
        print("\n" + "=" * 60)
        print("COST ENGINE V2 - Core Cost Prediction")
        print("=" * 60)

        self.df = df.copy()
        self._create_cost_features()
        self._encode_categoricals()
        self._scale_features()
        self._select_features()

        print(f"Cost engine processed {self.df.shape[0]} projects")
        return self

    def _create_cost_features(self):
        """Engineer cost-related features."""
        # Calculate actual cost overrun percentage
        self.df["actual_overrun_percent"] = (
            (self.df["actual_cost"] - self.df["estimated_cost"]) / 
            self.df["estimated_cost"]
        ) * 100

        # Predicted overrun based on project characteristics
        self.df["predicted_overrun_percent"] = (
            0.3 * self.df["calculated_delay_days"]
            + 0.1 * self.df["equipment_downtime_hours"]
            - 0.05 * self.df["crew_size_avg"]
        ).round(2)

    def _encode_categoricals(self):
        """Encode categorical variables."""
        cat_cols = ["project_type", "location"]
        for col in cat_cols:
            if col in self.df.columns:
                le = LabelEncoder()
                self.df[f"{col}_enc"] = le.fit_transform(self.df[col].fillna("Unknown"))
                self.encoders[col] = le

    def _scale_features(self):
        """Standardize numeric features."""
        num_cols = ["estimated_cost", "actual_cost", "crew_size_avg", "equipment_downtime_hours"]
        for col in num_cols:
            if col in self.df.columns:
                scaler = StandardScaler()
                self.df[f"{col}_scaled"] = scaler.fit_transform(self.df[[col]])
                self.scalers[col] = scaler

    def _select_features(self):
        """Select final feature set."""
        self.feature_names = [
            "estimated_cost_scaled",
            "actual_cost_scaled",
            "crew_size_avg_scaled",
            "equipment_downtime_hours_scaled",
            "predicted_overrun_percent"
        ]
        self.X = self.df[self.feature_names]
        self.y = self.df["actual_overrun_percent"]

    def get_features(self):
        """Return feature matrix and target variable."""
        return self.X, self.y

    def get_all_cost_breakdowns(self):
        """
        Generate detailed cost breakdowns for all projects.
        
        Returns:
            dict: Cost breakdowns indexed by project
        """
        if self.df is None or self.df.empty:
            return {}

        cost_breakdowns = {}
        
        for idx, row in self.df.iterrows():
            estimated = row.get('estimated_cost', 1000000)
            actual = row.get('actual_cost', estimated)
            overrun = actual - estimated
            delay_days = row.get('calculated_delay_days', 0)
            
            # Industry-standard cost distribution percentages
            labor_pct = 0.45
            materials_pct = 0.40
            admin_pct = 0.15
            
            # Distribute overruns based on typical construction cost patterns
            labor_overrun = overrun * 0.50
            materials_overrun = overrun * 0.35
            admin_overrun = overrun * 0.15
            
            cost_breakdowns[idx] = {
                'labor': {
                    'total': int(estimated * labor_pct + labor_overrun),
                    'extended_timeline': int(delay_days * 500),
                    'overtime': int(labor_overrun * 0.60)
                },
                'materials': {
                    'total': int(estimated * materials_pct + materials_overrun),
                    'price_increases': int(materials_overrun * 0.70),
                    'rush_delivery': int(materials_overrun * 0.30)
                },
                'admin': {
                    'total': int(estimated * admin_pct + admin_overrun)
                }
            }

        print("Cost breakdown analysis completed.")
        return cost_breakdowns

    def get_all_worker_optimizations(self):
        """
        Generate worker allocation optimization recommendations.
        
        Returns:
            dict: Optimization recommendations indexed by project
        """
        if self.df is None or self.df.empty:
            return {}

        worker_optimizations = {}
        
        for idx, row in self.df.iterrows():
            current_crew = int(row.get('crew_size_avg', 20))
            delay_days = row.get('calculated_delay_days', 0)
            estimated_cost = row.get('estimated_cost', 1000000)
            
            # Calculate optimal crew size based on delay patterns
            if delay_days > 15:
                optimal_crew = int(current_crew * 1.25)
                potential_delay_reduction = delay_days * 0.40
            elif delay_days > 7:
                optimal_crew = int(current_crew * 1.15)
                potential_delay_reduction = delay_days * 0.25
            else:
                optimal_crew = current_crew
                potential_delay_reduction = 0
            
            # Calculate financial impact
            additional_workers = optimal_crew - current_crew
            additional_cost = additional_workers * 80 * 8 * 30  # $80/hr * 8hrs * 30 days
            
            # Estimated savings from reduced delays
            delay_cost_per_day = estimated_cost * 0.002
            savings = potential_delay_reduction * delay_cost_per_day
            
            net_benefit = savings - additional_cost
            
            worker_optimizations[idx] = {
                'phases': {
                    'critical': {
                        'current': current_crew,
                        'optimal': optimal_crew
                    }
                },
                'net_benefit': int(net_benefit),
                'additional_cost': int(additional_cost),
                'savings': int(savings)
            }

        print("Worker optimization analysis completed.")
        return worker_optimizations


if __name__ == "__main__":
    print("\nTesting DelayEngineV2 and CostEngineV2...")

    sample_data = {
        "project_id": [1, 2, 3],
        "planned_end_date": ["2024-01-01", "2024-02-01", "2024-03-01"],
        "actual_end_date": ["2024-01-10", "2024-02-05", "2024-03-20"],
        "estimated_cost": [1000000, 800000, 1200000],
        "actual_cost": [1100000, 820000, 1300000],
        "crew_size_avg": [25, 15, 35],
        "equipment_downtime_hours": [10, 60, 25],
        "total_labor_hours": [1000, 800, 1500],
        "project_type": ["Residential", "Commercial", "Infrastructure"],
        "location": ["NYC", "Boston", "Chicago"]
    }

    df = pd.DataFrame(sample_data)

    delay_engine = DelayEngineV2().process(df)
    cost_engine = CostEngineV2().process(df)

    # Test all methods
    X_delay, y_delay = delay_engine.get_features()
    X_cost, y_cost = cost_engine.get_features()
    
    phase_analysis = delay_engine.get_all_phase_analysis()
    cost_breakdowns = cost_engine.get_all_cost_breakdowns()
    worker_opts = cost_engine.get_all_worker_optimizations()
    
    print("\nAll methods working successfully!")
    print(f"\nPhase Analysis Keys: {list(phase_analysis.keys())}")
    print(f"Cost Breakdown Keys: {list(cost_breakdowns.keys())}")
    print(f"Worker Optimization Keys: {list(worker_opts.keys())}")