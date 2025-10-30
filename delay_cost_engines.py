import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class DelayPreprocessingEngine:
    """
    Pipeline 2a: Delay-Focused Preprocessing Engine
    
    Takes standardized data from Universal Adapter and creates
    delay-specific features for ML model training/prediction.
    """
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.feature_names = []
    
    def process(self, standardized_df):
        """
        Process standardized data for delay prediction.
        
        Input: Clean standardized DataFrame from Pipeline 1
        Output: Feature-engineered DataFrame for delay modeling
        """
        print("\n" + "🔧 "+"="*68)
        print("DELAY PREPROCESSING ENGINE - PIPELINE 2a")
        print("="*70)
        
        self.df = standardized_df.copy()
        
        # Feature Engineering for Delays
        self._create_delay_features()
        self._encode_categoricals()
        self._scale_numeric_features()
        self._select_delay_features()
        
        print("="*70)
        print(f"✅ Delay engine complete - {self.X.shape[1]} features created")
        print("="*70)
        
        return self
    
    def _create_delay_features(self):
        """Create delay-specific engineered features."""
        print("\n📊 Engineering delay features...")
        
        # Weather impact ratio
        if 'rain_days' in self.df.columns and 'planned_duration' in self.df.columns:
            self.df['weather_impact_ratio'] = self.df['rain_days'] / (self.df['planned_duration'] + 1)
        
        # Material risk score
        if 'material_delay_days' in self.df.columns and 'supplier_reliability' in self.df.columns:
            self.df['material_risk_score'] = (
                self.df['material_delay_days'] * (100 - self.df['supplier_reliability']) / 100
            )
        
        # Crew efficiency (inverse relationship with delays)
        if 'crew_size' in self.df.columns:
            self.df['crew_efficiency_score'] = np.where(
                self.df['crew_size'] > 40, 1.0, 0.7
            )
        
        # Total environmental delay
        delay_cols = [col for col in self.df.columns if 'delay' in col.lower() and col != 'project_delay_days']
        if delay_cols:
            self.df['total_external_delays'] = self.df[delay_cols].sum(axis=1)
        
        # Inspection complexity factor
        if 'inspections' in self.df.columns and 'planned_duration' in self.df.columns:
            self.df['inspection_density'] = self.df['inspections'] / (self.df['planned_duration'] / 30)
        
        # Project scale indicator
        if 'planned_duration' in self.df.columns:
            self.df['project_scale'] = pd.cut(
                self.df['planned_duration'], 
                bins=[0, 90, 180, 365, 1000],
                labels=['Small', 'Medium', 'Large', 'Mega']
            )
        
        print(f"  ✓ Created {6} delay-specific features")
    
    def _encode_categoricals(self):
        """Encode categorical variables."""
        print("\n🏷️  Encoding categorical variables...")
        
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        for col in categorical_cols:
            if col in self.df.columns and self.df[col].notna().any():
                le = LabelEncoder()
                self.df[f'{col}_encoded'] = le.fit_transform(self.df[col].fillna('Unknown'))
                self.encoders[col] = le
        
        print(f"  ✓ Encoded {len(categorical_cols)} categorical columns")
    
    def _scale_numeric_features(self):
        """Scale numerical features."""
        print("\n📏 Scaling numerical features...")
        
        numeric_cols = [
            'crew_size', 'rain_days', 'material_delay_days', 'inspections',
            'supplier_reliability', 'weather_severity', 'planned_duration'
        ]
        
        scaled_count = 0
        for col in numeric_cols:
            if col in self.df.columns:
                scaler = StandardScaler()
                self.df[f'{col}_scaled'] = scaler.fit_transform(self.df[[col]].fillna(0))
                self.scalers[col] = scaler
                scaled_count += 1
        
        print(f"  ✓ Scaled {scaled_count} numerical features")
    
    def _select_delay_features(self):
        """Select final feature set for delay prediction."""
        print("\n🎯 Selecting delay prediction features...")
        
        # Priority features for delay prediction
        feature_cols = []
        
        # Scaled numeric features
        feature_cols.extend([col for col in self.df.columns if col.endswith('_scaled')])
        
        # Encoded categorical features
        feature_cols.extend([col for col in self.df.columns if col.endswith('_encoded')])
        
        # Engineered features
        engineered = [
            'weather_impact_ratio', 'material_risk_score', 'crew_efficiency_score',
            'total_external_delays', 'inspection_density'
        ]
        feature_cols.extend([col for col in engineered if col in self.df.columns])
        
        # Remove target from features
        feature_cols = [col for col in feature_cols if 'project_delay' not in col]
        
        self.X = self.df[feature_cols].fillna(0)
        self.feature_names = feature_cols
        
        # Target variable
        if 'project_delay_days' in self.df.columns:
            self.y = self.df['project_delay_days'].fillna(0)
        else:
            self.y = None
        
        print(f"  ✓ Selected {len(feature_cols)} features for modeling")
    
    def get_features(self):
        """Return feature matrix X and target y."""
        if self.y is not None:
            return self.X, self.y
        return self.X
    
    def get_feature_names(self):
        """Return list of feature names."""
        return self.feature_names


class CostPreprocessingEngine:
    """
    Pipeline 2b: Cost-Focused Preprocessing Engine
    
    Takes standardized data from Universal Adapter and creates
    cost-specific features for ML model training/prediction.
    """
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.feature_names = []
    
    def process(self, standardized_df):
        """
        Process standardized data for cost prediction.
        
        Input: Clean standardized DataFrame from Pipeline 1
        Output: Feature-engineered DataFrame for cost modeling
        """
        print("\n" + "💰 "+"="*68)
        print("COST PREPROCESSING ENGINE - PIPELINE 2b")
        print("="*70)
        
        self.df = standardized_df.copy()
        
        # Feature Engineering for Costs
        self._create_cost_features()
        self._encode_categoricals()
        self._scale_numeric_features()
        self._select_cost_features()
        
        print("="*70)
        print(f"✅ Cost engine complete - {self.X.shape[1]} features created")
        print("="*70)
        
        return self
    
    def _create_cost_features(self):
        """Create cost-specific engineered features."""
        print("\n📊 Engineering cost features...")
        
        # Cost per day (efficiency metric)
        if 'planned_cost' in self.df.columns and 'planned_duration' in self.df.columns:
            self.df['cost_per_day'] = self.df['planned_cost'] / (self.df['planned_duration'] + 1)
        
        # Delay cost impact (delays increase costs)
        if 'project_delay_days' in self.df.columns and 'planned_cost' in self.df.columns:
            self.df['delay_cost_factor'] = (
                1 + (self.df['project_delay_days'] / self.df['planned_duration']) * 0.5
            )
        
        # Material cost risk
        if 'material_delay_days' in self.df.columns:
            self.df['material_cost_risk'] = self.df['material_delay_days'] * 0.02  # 2% cost increase per delay day
        
        # Labor cost (crew size × duration)
        if 'crew_size' in self.df.columns and 'planned_duration' in self.df.columns:
            self.df['labor_cost_index'] = self.df['crew_size'] * self.df['planned_duration']
        
        # Budget efficiency
        if 'planned_cost' in self.df.columns and 'actual_cost' in self.df.columns:
            self.df['budget_efficiency'] = (
                self.df['planned_cost'] / (self.df['actual_cost'] + 1)
            )
        
        # Weather cost impact
        if 'rain_days' in self.df.columns and 'planned_cost' in self.df.columns:
            self.df['weather_cost_impact'] = (
                self.df['rain_days'] * self.df['planned_cost'] * 0.001  # 0.1% cost per rain day
            )
        
        # Project size category
        if 'planned_cost' in self.df.columns:
            self.df['project_size_category'] = pd.cut(
                self.df['planned_cost'],
                bins=[0, 500000, 2000000, 10000000, np.inf],
                labels=['Small', 'Medium', 'Large', 'Mega']
            )
        
        print(f"  ✓ Created {7} cost-specific features")
    
    def _encode_categoricals(self):
        """Encode categorical variables."""
        print("\n🏷️  Encoding categorical variables...")
        
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        for col in categorical_cols:
            if col in self.df.columns and self.df[col].notna().any():
                le = LabelEncoder()
                self.df[f'{col}_encoded'] = le.fit_transform(self.df[col].fillna('Unknown'))
                self.encoders[col] = le
        
        print(f"  ✓ Encoded {len(categorical_cols)} categorical columns")
    
    def _scale_numeric_features(self):
        """Scale numerical features."""
        print("\n📏 Scaling numerical features...")
        
        numeric_cols = [
            'planned_cost', 'planned_duration', 'crew_size', 'supplier_reliability',
            'rain_days', 'material_delay_days', 'project_delay_days'
        ]
        
        scaled_count = 0
        for col in numeric_cols:
            if col in self.df.columns:
                scaler = StandardScaler()
                self.df[f'{col}_scaled'] = scaler.fit_transform(self.df[[col]].fillna(0))
                self.scalers[col] = scaler
                scaled_count += 1
        
        print(f"  ✓ Scaled {scaled_count} numerical features")
    
    def _select_cost_features(self):
        """Select final feature set for cost prediction."""
        print("\n🎯 Selecting cost prediction features...")
        
        feature_cols = []
        
        # Scaled numeric features
        feature_cols.extend([col for col in self.df.columns if col.endswith('_scaled')])
        
        # Encoded categorical features
        feature_cols.extend([col for col in self.df.columns if col.endswith('_encoded')])
        
        # Engineered features
        engineered = [
            'cost_per_day', 'delay_cost_factor', 'material_cost_risk',
            'labor_cost_index', 'budget_efficiency', 'weather_cost_impact'
        ]
        feature_cols.extend([col for col in engineered if col in self.df.columns])
        
        # Remove targets from features
        feature_cols = [col for col in feature_cols 
                       if not any(x in col for x in ['actual_cost', 'cost_overrun'])]
        
        self.X = self.df[feature_cols].fillna(0)
        self.feature_names = feature_cols
        
        # Target variable
        if 'cost_overrun' in self.df.columns:
            self.y = self.df['cost_overrun'].fillna(0)
        elif 'actual_cost' in self.df.columns:
            self.y = self.df['actual_cost'].fillna(0)
        else:
            self.y = None
        
        print(f"  ✓ Selected {len(feature_cols)} features for modeling")
    
    def get_features(self):
        """Return feature matrix X and target y."""
        if self.y is not None:
            return self.X, self.y
        return self.X
    
    def get_feature_names(self):
        """Return list of feature names."""
        return self.feature_names


# Example Usage
if __name__ == "__main__":
    # Assume we have standardized data from Pipeline 1
    from universal_data_adapter import UniversalDataAdapter
    
    # Step 1: Run Universal Adapter
    adapter = UniversalDataAdapter()
    adapter.run_full_adapter('your_data.csv')
    standardized_data = adapter.get_standardized_data()
    
    # Step 2a: Delay Engine
    delay_engine = DelayPreprocessingEngine()
    delay_engine.process(standardized_data)
    X_delay, y_delay = delay_engine.get_features()
    
    print(f"\n🎯 Delay Features Ready:")
    print(f"   Shape: {X_delay.shape}")
    print(f"   Target: {y_delay.name if hasattr(y_delay, 'name') else 'delay predictions'}")
    
    # Step 2b: Cost Engine
    cost_engine = CostPreprocessingEngine()
    cost_engine.process(standardized_data)
    X_cost, y_cost = cost_engine.get_features()
    
    print(f"\n💰 Cost Features Ready:")
    print(f"   Shape: {X_cost.shape}")
    print(f"   Target: {y_cost.name if hasattr(y_cost, 'name') else 'cost predictions'}")
    
    print("\n✨ Ready for Pipeline 3 (Training/Prediction)!")