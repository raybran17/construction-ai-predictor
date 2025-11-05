import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class EnhancedDelayEngine:
    """
    Enhanced Delay Analysis Engine with Phase-Level Breakdown
    
    Outputs:
    - Total delay prediction
    - Breakdown by project phase
    - Root cause analysis
    - Worker allocation impact
    """
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.feature_names = []
        self.phase_delays = {}
        self.delay_drivers = {}
    
    def process(self, standardized_df):
        """Process data and create delay analysis with phase breakdown."""
        print("\n" + "⏱️  "+"="*68)
        print("ENHANCED DELAY ENGINE - Phase-Level Analysis")
        print("="*70)
        
        self.df = standardized_df.copy()
        
        self._create_delay_features()
        self._analyze_phases()
        self._encode_categoricals()
        self._scale_features()
        self._select_features()
        
        print("="*70)
        print(f"✅ Delay engine complete - {self.X.shape[1]} features + phase analysis")
        print("="*70)
        
        return self
    
    def _create_delay_features(self):
        """Create delay-specific features."""
        print("\n📊 Engineering delay features...")
        
        # Weather impact
        if 'rain_days' in self.df.columns and 'planned_duration' in self.df.columns:
            self.df['weather_impact_ratio'] = self.df['rain_days'] / (self.df['planned_duration'] + 1)
        
        # Material risk
        if 'material_delay_days' in self.df.columns and 'supplier_reliability' in self.df.columns:
            self.df['material_risk_score'] = (
                self.df['material_delay_days'] * (100 - self.df['supplier_reliability']) / 100
            )
        
        # Worker efficiency
        if 'crew_size' in self.df.columns:
            self.df['crew_efficiency_score'] = np.where(
                self.df['crew_size'] > 40, 1.0,
                np.where(self.df['crew_size'] > 25, 0.85, 0.7)
            )
            self.df['worker_shortage_risk'] = np.where(
                self.df['crew_size'] < 25, 1, 0
            )
        
        # Total external delays
        delay_cols = [col for col in self.df.columns if 'delay' in col.lower() and col != 'project_delay_days']
        if delay_cols:
            self.df['total_external_delays'] = self.df[delay_cols].sum(axis=1)
        
        print(f"  ✓ Created delay-specific features")
    
    def _analyze_phases(self):
        """Analyze delays by project phase."""
        print("\n🔍 Analyzing project phases...")
        
        # Estimate phase delays based on project characteristics
        for idx in self.df.index:
            phases = {}
            
            total_delay = self.df.loc[idx, 'project_delay_days'] if 'project_delay_days' in self.df.columns else 0
            
            # Foundation phase (30% of typical delays)
            material_delay = self.df.loc[idx, 'material_delay_days'] if 'material_delay_days' in self.df.columns else 0
            phases['foundation'] = {
                'delay_days': int(material_delay * 0.6 + total_delay * 0.3),
                'drivers': ['Material delays', 'Inspection waits']
            }
            
            # Framing phase (40% of typical delays)
            crew_size = self.df.loc[idx, 'crew_size'] if 'crew_size' in self.df.columns else 30
            crew_delay = 5 if crew_size < 25 else 0
            phases['framing'] = {
                'delay_days': int(crew_delay + total_delay * 0.4),
                'drivers': ['Worker shortage', 'Equipment delays'] if crew_size < 25 else ['Weather delays']
            }
            
            # Finishing phase (30% of typical delays)
            phases['finishing'] = {
                'delay_days': int(total_delay * 0.3),
                'drivers': ['Permit modifications', 'Final inspections']
            }
            
            self.phase_delays[idx] = phases
        
        print(f"  ✓ Analyzed {len(self.phase_delays)} project phases")
    
    def _encode_categoricals(self):
        """Encode categorical variables."""
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        for col in categorical_cols:
            if col in self.df.columns and self.df[col].notna().any():
                le = LabelEncoder()
                self.df[f'{col}_encoded'] = le.fit_transform(self.df[col].fillna('Unknown'))
                self.encoders[col] = le
    
    def _scale_features(self):
        """Scale numerical features."""
        numeric_cols = [
            'crew_size', 'rain_days', 'material_delay_days', 'inspections',
            'supplier_reliability', 'weather_severity', 'planned_duration'
        ]
        
        for col in numeric_cols:
            if col in self.df.columns:
                scaler = StandardScaler()
                self.df[f'{col}_scaled'] = scaler.fit_transform(self.df[[col]].fillna(0))
                self.scalers[col] = scaler
    
    def _select_features(self):
        """Select features for modeling."""
        feature_cols = []
        feature_cols.extend([col for col in self.df.columns if col.endswith('_scaled')])
        feature_cols.extend([col for col in self.df.columns if col.endswith('_encoded')])
        
        engineered = [
            'weather_impact_ratio', 'material_risk_score', 'crew_efficiency_score',
            'total_external_delays', 'worker_shortage_risk'
        ]
        feature_cols.extend([col for col in engineered if col in self.df.columns])
        
        feature_cols = [col for col in feature_cols if 'project_delay' not in col]
        
        self.X = self.df[feature_cols].fillna(0)
        self.feature_names = feature_cols
        
        if 'project_delay_days' in self.df.columns:
            self.y = self.df['project_delay_days'].fillna(0)
        else:
            self.y = None
    
    def get_features(self):
        """Return features and target."""
        if self.y is not None:
            return self.X, self.y
        return self.X
    
    def get_phase_analysis(self, project_idx):
        """Get detailed phase breakdown for a project."""
        return self.phase_delays.get(project_idx, {})
    
    def get_all_phase_analysis(self):
        """Get phase analysis for all projects."""
        return self.phase_delays


class EnhancedCostEngine:
    """
    Enhanced Cost Analysis Engine with Category Breakdown + Worker Optimization
    
    Outputs:
    - Total cost impact
    - Breakdown by category (labor, materials, permits)
    - Worker allocation cost/benefit
    - Cost driver analysis
    """
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.feature_names = []
        self.cost_breakdown = {}
        self.worker_optimization = {}
    
    def process(self, standardized_df):
        """Process data and create cost analysis with breakdowns."""
        print("\n" + "💰 "+"="*68)
        print("ENHANCED COST ENGINE - Category Breakdown + Worker Analysis")
        print("="*70)
        
        self.df = standardized_df.copy()
        
        self._create_cost_features()
        self._analyze_cost_categories()
        self._analyze_worker_allocation()
        self._encode_categoricals()
        self._scale_features()
        self._select_features()
        
        print("="*70)
        print(f"✅ Cost engine complete - {self.X.shape[1]} features + cost breakdown")
        print("="*70)
        
        return self
    
    def _create_cost_features(self):
        """Create cost-specific features."""
        print("\n📊 Engineering cost features...")
        
        # Cost per day
        if 'planned_cost' in self.df.columns and 'planned_duration' in self.df.columns:
            self.df['cost_per_day'] = self.df['planned_cost'] / (self.df['planned_duration'] + 1)
        
        # Delay cost impact
        if 'project_delay_days' in self.df.columns and 'planned_cost' in self.df.columns:
            self.df['delay_cost_factor'] = (
                1 + (self.df['project_delay_days'] / self.df['planned_duration']) * 0.5
            )
        
        # Material cost risk
        if 'material_delay_days' in self.df.columns:
            self.df['material_cost_risk'] = self.df['material_delay_days'] * 0.02
        
        # Labor cost index
        if 'crew_size' in self.df.columns and 'planned_duration' in self.df.columns:
            self.df['labor_cost_index'] = self.df['crew_size'] * self.df['planned_duration']
        
        # Worker cost per day
        if 'crew_size' in self.df.columns:
            self.df['daily_labor_cost'] = self.df['crew_size'] * 250  # $250/worker/day avg
        
        print(f"  ✓ Created cost-specific features")
    
    def _analyze_cost_categories(self):
        """Break down costs by category."""
        print("\n🔍 Analyzing cost categories...")
        
        for idx in self.df.index:
            planned_cost = self.df.loc[idx, 'planned_cost'] if 'planned_cost' in self.df.columns else 0
            actual_cost = self.df.loc[idx, 'actual_cost'] if 'actual_cost' in self.df.columns else planned_cost
            total_overrun = actual_cost - planned_cost
            
            # Estimate breakdown
            delay_days = self.df.loc[idx, 'project_delay_days'] if 'project_delay_days' in self.df.columns else 0
            crew_size = self.df.loc[idx, 'crew_size'] if 'crew_size' in self.df.columns else 25
            material_delay = self.df.loc[idx, 'material_delay_days'] if 'material_delay_days' in self.df.columns else 0
            
            # Labor overruns (60% of typical overrun)
            labor_overrun = int(total_overrun * 0.6)
            extended_timeline_cost = int(delay_days * crew_size * 250)
            overtime_cost = labor_overrun - extended_timeline_cost
            
            # Materials (30% of typical overrun)
            material_overrun = int(total_overrun * 0.3)
            price_increase = int(material_overrun * 0.7)
            rush_fees = int(material_overrun * 0.3)
            
            # Permits/Admin (10%)
            admin_overrun = int(total_overrun * 0.1)
            
            self.cost_breakdown[idx] = {
                'total_overrun': total_overrun,
                'labor': {
                    'total': labor_overrun,
                    'extended_timeline': extended_timeline_cost,
                    'overtime': max(0, overtime_cost)
                },
                'materials': {
                    'total': material_overrun,
                    'price_increases': price_increase,
                    'rush_delivery': rush_fees
                },
                'admin': {
                    'total': admin_overrun,
                    'permit_mods': int(admin_overrun * 0.8),
                    'fees': int(admin_overrun * 0.2)
                }
            }
        
        print(f"  ✓ Analyzed cost breakdown for {len(self.cost_breakdown)} projects")
    
    def _analyze_worker_allocation(self):
        """Analyze worker allocation optimization."""
        print("\n👷 Analyzing worker allocation...")
        
        for idx in self.df.index:
            crew_size = self.df.loc[idx, 'crew_size'] if 'crew_size' in self.df.columns else 25
            planned_duration = self.df.loc[idx, 'planned_duration'] if 'planned_duration' in self.df.columns else 100
            delay_days = self.df.loc[idx, 'project_delay_days'] if 'project_delay_days' in self.df.columns else 0
            
            # Calculate optimal allocation
            phases = {
                'early': {'current': crew_size, 'optimal': max(12, crew_size - 3)},
                'critical': {'current': crew_size, 'optimal': crew_size + 7},
                'finishing': {'current': crew_size, 'optimal': crew_size + 3}
            }
            
            # Cost/benefit analysis
            additional_labor_cost = 7 * 250 * (planned_duration // 3)  # 7 workers for 1/3 project
            delay_cost_saved = delay_days * crew_size * 250 * 0.5  # Save 50% of delays
            net_savings = delay_cost_saved - additional_labor_cost
            
            self.worker_optimization[idx] = {
                'phases': phases,
                'additional_cost': additional_labor_cost,
                'savings': delay_cost_saved,
                'net_benefit': net_savings,
                'recommendation': 'increase_critical_phase' if net_savings > 0 else 'maintain_current'
            }
        
        print(f"  ✓ Optimized worker allocation for {len(self.worker_optimization)} projects")
    
    def _encode_categoricals(self):
        """Encode categorical variables."""
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        for col in categorical_cols:
            if col in self.df.columns and self.df[col].notna().any():
                le = LabelEncoder()
                self.df[f'{col}_encoded'] = le.fit_transform(self.df[col].fillna('Unknown'))
                self.encoders[col] = le
    
    def _scale_features(self):
        """Scale numerical features."""
        numeric_cols = [
            'planned_cost', 'planned_duration', 'crew_size', 'supplier_reliability',
            'rain_days', 'material_delay_days', 'project_delay_days'
        ]
        
        for col in numeric_cols:
            if col in self.df.columns:
                scaler = StandardScaler()
                self.df[f'{col}_scaled'] = scaler.fit_transform(self.df[[col]].fillna(0))
                self.scalers[col] = scaler
    
    def _select_features(self):
        """Select features for modeling."""
        feature_cols = []
        feature_cols.extend([col for col in self.df.columns if col.endswith('_scaled')])
        feature_cols.extend([col for col in self.df.columns if col.endswith('_encoded')])
        
        engineered = [
            'cost_per_day', 'delay_cost_factor', 'material_cost_risk',
            'labor_cost_index', 'daily_labor_cost'
        ]
        feature_cols.extend([col for col in engineered if col in self.df.columns])
        
        feature_cols = [col for col in feature_cols 
                       if not any(x in col for x in ['actual_cost', 'cost_overrun'])]
        
        self.X = self.df[feature_cols].fillna(0)
        self.feature_names = feature_cols
        
        if 'cost_overrun' in self.df.columns:
            self.y = self.df['cost_overrun'].fillna(0)
        elif 'actual_cost' in self.df.columns:
            self.y = self.df['actual_cost'].fillna(0)
        else:
            self.y = None
    
    def get_features(self):
        """Return features and target."""
        if self.y is not None:
            return self.X, self.y
        return self.X
    
    def get_cost_breakdown(self, project_idx):
        """Get detailed cost breakdown for a project."""
        return self.cost_breakdown.get(project_idx, {})
    
    def get_worker_optimization(self, project_idx):
        """Get worker allocation analysis for a project."""
        return self.worker_optimization.get(project_idx, {})
    
    def get_all_cost_breakdowns(self):
        """Get cost breakdowns for all projects."""
        return self.cost_breakdown
    
    def get_all_worker_optimizations(self):
        """Get worker optimizations for all projects."""
        return self.worker_optimization
