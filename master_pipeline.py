import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class EnhancedMasterPipeline:
    """
    Enhanced Master Pipeline with Detailed Outputs
    
    Provides:
    - Predictions with confidence scores
    - Phase-level delay breakdown
    - Cost category breakdown
    - Worker allocation recommendations
    - Actionable insights
    """
    
    def __init__(self):
        self.delay_model = None
        self.cost_model = None
        self.metrics = {}
        self.predictions = {}
        self.detailed_analysis = {}
    
    def train_delay_model(self, X_delay, y_delay, model_type='random_forest'):
        """Train delay prediction model."""
        print("\n" + "🎓 "+"="*68)
        print("TRAINING DELAY MODEL")
        print("="*70)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_delay, y_delay, test_size=0.2, random_state=42
        )
        
        print(f"📊 Training set: {X_train.shape[0]} projects")
        print(f"📊 Test set: {X_test.shape[0]} projects")
        
        self.delay_model = RandomForestRegressor(
            n_estimators=100, max_depth=15, min_samples_split=5,
            random_state=42, n_jobs=-1
        )
        
        print(f"\n🤖 Training {model_type} model...")
        self.delay_model.fit(X_train, y_train)
        
        y_pred_test = self.delay_model.predict(X_test)
        
        self.metrics['delay'] = {
            'test': {
                'mae': mean_absolute_error(y_test, y_pred_test),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
                'r2': r2_score(y_test, y_pred_test)
            }
        }
        
        print("\n📈 MODEL PERFORMANCE:")
        print(f"  Test MAE:  {self.metrics['delay']['test']['mae']:.2f} days")
        print(f"  Test RMSE: {self.metrics['delay']['test']['rmse']:.2f} days")
        print(f"  Test R²:   {self.metrics['delay']['test']['r2']:.3f}")
        print(f"  Accuracy:  {self.metrics['delay']['test']['r2']*100:.1f}%")
        
        print("="*70)
        print("✅ Delay model trained!")
        print("="*70)
        
        return self
    
    def train_cost_model(self, X_cost, y_cost, model_type='random_forest'):
        """Train cost prediction model."""
        print("\n" + "💰 "+"="*68)
        print("TRAINING COST MODEL")
        print("="*70)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_cost, y_cost, test_size=0.2, random_state=42
        )
        
        print(f"📊 Training set: {X_train.shape[0]} projects")
        print(f"📊 Test set: {X_test.shape[0]} projects")
        
        self.cost_model = RandomForestRegressor(
            n_estimators=100, max_depth=15, min_samples_split=5,
            random_state=42, n_jobs=-1
        )
        
        print(f"\n🤖 Training {model_type} model...")
        self.cost_model.fit(X_train, y_train)
        
        y_pred_test = self.cost_model.predict(X_test)
        
        self.metrics['cost'] = {
            'test': {
                'mae': mean_absolute_error(y_test, y_pred_test),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
                'r2': r2_score(y_test, y_pred_test)
            }
        }
        
        print("\n📈 MODEL PERFORMANCE:")
        print(f"  Test MAE:  {self.metrics['cost']['test']['mae']:.2f}%")
        print(f"  Test RMSE: {self.metrics['cost']['test']['rmse']:.2f}%")
        print(f"  Test R²:   {self.metrics['cost']['test']['r2']:.3f}")
        print(f"  Accuracy:  {self.metrics['cost']['test']['r2']*100:.1f}%")
        
        print("="*70)
        print("✅ Cost model trained!")
        print("="*70)
        
        return self
    
    def predict_with_details(self, X_delay, X_cost, delay_engine, cost_engine):
        """
        Make predictions with detailed breakdowns.
        
        Args:
            X_delay: Delay features
            X_cost: Cost features
            delay_engine: EnhancedDelayEngine instance with phase analysis
            cost_engine: EnhancedCostEngine instance with cost breakdown
        """
        print("\n" + "🔮 "+"="*68)
        print("GENERATING DETAILED PREDICTIONS")
        print("="*70)
        
        if self.delay_model is None or self.cost_model is None:
            raise ValueError("Models not trained!")
        
        # Make predictions
        delay_predictions = self.delay_model.predict(X_delay)
        cost_predictions = self.cost_model.predict(X_cost)
        
        # Get detailed analysis
        phase_analysis = delay_engine.get_all_phase_analysis()
        cost_breakdowns = cost_engine.get_all_cost_breakdowns()
        worker_optimizations = cost_engine.get_all_worker_optimizations()
        
        # Store everything
        self.predictions = {
            'delays': delay_predictions,
            'costs': cost_predictions,
            'count': len(delay_predictions)
        }
        
        self.detailed_analysis = {
            'phases': phase_analysis,
            'cost_breakdown': cost_breakdowns,
            'worker_optimization': worker_optimizations
        }
        
        print(f"✅ Generated detailed predictions for {len(delay_predictions)} projects")
        print(f"\n📊 PREDICTION SUMMARY:")
        print(f"  Avg Predicted Delay: {np.mean(delay_predictions):.1f} days")
        print(f"  Avg Predicted Cost Impact: {np.mean(cost_predictions):.1f}%")
        print(f"  High Risk Projects (>20 day delay): {sum(delay_predictions > 20)}")
        
        return self
    
    def get_project_report(self, project_idx):
        """
        Generate comprehensive report for a single project.
        
        Returns detailed breakdown of delays, costs, and recommendations.
        """
        if not self.predictions:
            raise ValueError("No predictions available! Run predict_with_details() first.")
        
        delay_pred = self.predictions['delays'][project_idx]
        cost_pred = self.predictions['costs'][project_idx]
        
        phases = self.detailed_analysis['phases'].get(project_idx, {})
        cost_breakdown = self.detailed_analysis['cost_breakdown'].get(project_idx, {})
        worker_opt = self.detailed_analysis['worker_optimization'].get(project_idx, {})
        
        report = {
            'summary': {
                'total_delay_days': round(delay_pred, 1),
                'cost_overrun_pct': round(cost_pred, 1),
                'risk_level': self._get_risk_level(delay_pred)
            },
            'delay_breakdown': {
                'foundation': phases.get('foundation', {}),
                'framing': phases.get('framing', {}),
                'finishing': phases.get('finishing', {})
            },
            'cost_breakdown': cost_breakdown,
            'worker_optimization': worker_opt,
            'recommendations': self._generate_recommendations(
                delay_pred, cost_pred, phases, cost_breakdown, worker_opt
            )
        }
        
        return report
    
    def _get_risk_level(self, delay_days):
        """Categorize risk level based on delay."""
        if delay_days < 5:
            return 'Low'
        elif delay_days < 15:
            return 'Medium'
        elif delay_days < 30:
            return 'High'
        else:
            return 'Critical'
    
    def _generate_recommendations(self, delay_pred, cost_pred, phases, cost_breakdown, worker_opt):
        """Generate actionable recommendations."""
        recommendations = {
            'high_priority': [],
            'medium_priority': [],
            'cost_savings': []
        }
        
        # Material-related recommendations
        if phases.get('foundation', {}).get('delay_days', 0) > 5:
            recommendations['high_priority'].append({
                'action': 'Order materials 2 weeks early',
                'impact': f"Saves {phases['foundation']['delay_days']} days",
                'cost': 'Minimal',
                'phase': 'Foundation'
            })
        
        # Worker allocation recommendations
        if worker_opt.get('net_benefit', 0) > 0:
            optimal_workers = worker_opt.get('phases', {}).get('critical', {}).get('optimal', 0)
            current_workers = worker_opt.get('phases', {}).get('critical', {}).get('current', 0)
            additional = optimal_workers - current_workers
            
            if additional > 0:
                recommendations['high_priority'].append({
                    'action': f'Add {additional} workers during critical phase',
                    'impact': f"Saves {int(delay_pred * 0.5)} days",
                    'cost': f"${worker_opt.get('additional_cost', 0):,}",
                    'savings': f"${worker_opt.get('savings', 0):,}",
                    'net_benefit': f"${worker_opt.get('net_benefit', 0):,}"
                })
        
        # Cost reduction recommendations
        if cost_pred > 10:
            recommendations['cost_savings'].append({
                'action': 'Negotiate bulk material pricing',
                'potential_savings': f"{cost_pred * 0.2:.1f}%",
                'effort': 'Medium'
            })
        
        # Schedule optimization
        if delay_pred > 15:
            recommendations['medium_priority'].append({
                'action': 'Schedule inspections in advance',
                'impact': 'Saves 3-5 days',
                'cost': 'Free'
            })
        
        return recommendations
    
    def generate_dashboard_output(self, project_idx):
        """
        Generate formatted output for dashboard display.
        """
        report = self.get_project_report(project_idx)
        
        output = f"""
{'='*70}
PROJECT ANALYSIS REPORT
{'='*70}

📊 DELAY PREDICTION:
Total Predicted Delay: {report['summary']['total_delay_days']} days
Risk Level: {report['summary']['risk_level']}

BREAKDOWN BY PHASE:
"""
        
        for phase_name, phase_data in report['delay_breakdown'].items():
            if phase_data:
                output += f"\n{phase_name.title()}: {phase_data.get('delay_days', 0)} days\n"
                for driver in phase_data.get('drivers', []):
                    output += f"  └─ {driver}\n"
        
        output += f"""
💰 COST PREDICTION:
Total Cost Impact: {report['summary']['cost_overrun_pct']:.1f}% overrun

BREAKDOWN BY CATEGORY:
Labor: ${report['cost_breakdown'].get('labor', {}).get('total', 0):,}
  └─ Extended timeline: ${report['cost_breakdown'].get('labor', {}).get('extended_timeline', 0):,}
  └─ Overtime: ${report['cost_breakdown'].get('labor', {}).get('overtime', 0):,}

Materials: ${report['cost_breakdown'].get('materials', {}).get('total', 0):,}
  └─ Price increases: ${report['cost_breakdown'].get('materials', {}).get('price_increases', 0):,}
  └─ Rush fees: ${report['cost_breakdown'].get('materials', {}).get('rush_delivery', 0):,}

Admin/Permits: ${report['cost_breakdown'].get('admin', {}).get('total', 0):,}

🎯 RECOMMENDED ACTIONS:

HIGH PRIORITY:
"""
        
        for i, rec in enumerate(report['recommendations']['high_priority'], 1):
            output += f"{i}. {rec['action']}\n"
            output += f"   Impact: {rec['impact']}\n"
            if 'cost' in rec:
                output += f"   Cost: {rec['cost']}\n"
            if 'net_benefit' in rec:
                output += f"   Net Benefit: {rec['net_benefit']}\n"
            output += "\n"
        
        output += "{'='*70}\n"
        
        return output
    
    def save_models(self, delay_path='trained_delay_model.pkl', cost_path='trained_cost_model.pkl'):
        """Save trained models."""
        if self.delay_model:
            with open(delay_path, 'wb') as f:
                pickle.dump(self.delay_model, f)
            print(f"✅ Delay model saved to {delay_path}")
        
        if self.cost_model:
            with open(cost_path, 'wb') as f:
                pickle.dump(self.cost_model, f)
            print(f"✅ Cost model saved to {cost_path}")
    
    def load_models(self, delay_path='trained_delay_model.pkl', cost_path='trained_cost_model.pkl'):
        """Load trained models."""
        try:
            with open(delay_path, 'rb') as f:
                self.delay_model = pickle.load(f)
            print(f"✅ Delay model loaded from {delay_path}")
        except:
            print(f"⚠️  Could not load delay model")
        
        try:
            with open(cost_path, 'rb') as f:
                self.cost_model = pickle.load(f)
            print(f"✅ Cost model loaded from {cost_path}")
        except:
            print(f"⚠️  Could not load cost model")
    
    def export_results(self, output_path='detailed_predictions.csv'):
        """Export predictions with details."""
        if not self.predictions:
            print("⚠️  No predictions to export")
            return
        
        results_data = []
        for idx in range(len(self.predictions['delays'])):
            report = self.get_project_report(idx)
            results_data.append({
                'Project_Index': idx,
                'Predicted_Delay_Days': report['summary']['total_delay_days'],
                'Cost_Overrun_Pct': report['summary']['cost_overrun_pct'],
                'Risk_Level': report['summary']['risk_level'],
                'Foundation_Delay': report['delay_breakdown'].get('foundation', {}).get('delay_days', 0),
                'Framing_Delay': report['delay_breakdown'].get('framing', {}).get('delay_days', 0),
                'Finishing_Delay': report['delay_breakdown'].get('finishing', {}).get('delay_days', 0),
                'Labor_Cost_Impact': report['cost_breakdown'].get('labor', {}).get('total', 0),
                'Material_Cost_Impact': report['cost_breakdown'].get('materials', {}).get('total', 0),
                'Top_Recommendation': report['recommendations']['high_priority'][0]['action'] if report['recommendations']['high_priority'] else 'None'
            })
        
        results_df = pd.DataFrame(results_data)
        results_df.to_csv(output_path, index=False)
        print(f"✅ Detailed results exported to {output_path}")
    
    def get_metrics(self):
        """Return model metrics."""
        return self.metrics
    
    def get_predictions(self):
        """Return predictions."""
        return self.predictions
    
    def get_detailed_analysis(self):
        """Return detailed analysis."""
        return self.detailed_analysis