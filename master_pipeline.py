import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class MasterPredictionPipeline:
    """
    Pipeline 3: Master Training & Prediction Engine
    
    Orchestrates the entire 3-pipeline system:
    1. Universal Adapter → 2a. Delay Engine → 2b. Cost Engine → 3. This (Training/Prediction)
    
    Trains models, makes predictions, generates insights for PMs.
    """
    
    def __init__(self):
        self.delay_model = None
        self.cost_model = None
        self.metrics = {}
        self.predictions = {}
        self.insights = {}
    
    def train_delay_model(self, X_delay, y_delay, model_type='random_forest'):
        """
        Train delay prediction model.
        
        Args:
            X_delay: Feature matrix from Delay Engine (Pipeline 2a)
            y_delay: Target delays
            model_type: 'random_forest' or 'gradient_boosting'
        """
        print("\n" + "🎓 "+"="*68)
        print("TRAINING DELAY MODEL")
        print("="*70)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_delay, y_delay, test_size=0.2, random_state=42
        )
        
        print(f"📊 Training set: {X_train.shape[0]} projects")
        print(f"📊 Test set: {X_test.shape[0]} projects")
        
        # Select model
        if model_type == 'random_forest':
            self.delay_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            )
        else:
            self.delay_model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
        
        print(f"\n🤖 Training {model_type} model...")
        self.delay_model.fit(X_train, y_train)
        
        # Evaluate
        y_pred_train = self.delay_model.predict(X_train)
        y_pred_test = self.delay_model.predict(X_test)
        
        # Calculate metrics
        self.metrics['delay'] = {
            'train': {
                'mae': mean_absolute_error(y_train, y_pred_train),
                'rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
                'r2': r2_score(y_train, y_pred_train)
            },
            'test': {
                'mae': mean_absolute_error(y_test, y_pred_test),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
                'r2': r2_score(y_test, y_pred_test)
            }
        }
        
        # Display results
        print("\n📈 MODEL PERFORMANCE:")
        print(f"  Training Set:")
        print(f"    MAE:  {self.metrics['delay']['train']['mae']:.2f} days")
        print(f"    RMSE: {self.metrics['delay']['train']['rmse']:.2f} days")
        print(f"    R²:   {self.metrics['delay']['train']['r2']:.3f}")
        print(f"\n  Test Set:")
        print(f"    MAE:  {self.metrics['delay']['test']['mae']:.2f} days")
        print(f"    RMSE: {self.metrics['delay']['test']['rmse']:.2f} days")
        print(f"    R²:   {self.metrics['delay']['test']['r2']:.3f}")
        
        # Feature importance
        if hasattr(self.delay_model, 'feature_importances_'):
            importances = self.delay_model.feature_importances_
            top_features_idx = np.argsort(importances)[-5:]
            print(f"\n🔍 TOP 5 DELAY DRIVERS:")
            for idx in reversed(top_features_idx):
                print(f"    • {X_delay.columns[idx]}: {importances[idx]:.3f}")
        
        print("="*70)
        print("✅ Delay model trained successfully!")
        print("="*70)
        
        return self
    
    def train_cost_model(self, X_cost, y_cost, model_type='random_forest'):
        """
        Train cost prediction model.
        
        Args:
            X_cost: Feature matrix from Cost Engine (Pipeline 2b)
            y_cost: Target cost overruns
            model_type: 'random_forest' or 'gradient_boosting'
        """
        print("\n" + "💰 "+"="*68)
        print("TRAINING COST MODEL")
        print("="*70)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_cost, y_cost, test_size=0.2, random_state=42
        )
        
        print(f"📊 Training set: {X_train.shape[0]} projects")
        print(f"📊 Test set: {X_test.shape[0]} projects")
        
        # Select model
        if model_type == 'random_forest':
            self.cost_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            )
        else:
            self.cost_model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
        
        print(f"\n🤖 Training {model_type} model...")
        self.cost_model.fit(X_train, y_train)
        
        # Evaluate
        y_pred_train = self.cost_model.predict(X_train)
        y_pred_test = self.cost_model.predict(X_test)
        
        # Calculate metrics
        self.metrics['cost'] = {
            'train': {
                'mae': mean_absolute_error(y_train, y_pred_train),
                'rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
                'r2': r2_score(y_train, y_pred_train)
            },
            'test': {
                'mae': mean_absolute_error(y_test, y_pred_test),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
                'r2': r2_score(y_test, y_pred_test)
            }
        }
        
        # Display results
        print("\n📈 MODEL PERFORMANCE:")
        print(f"  Training Set:")
        print(f"    MAE:  ${self.metrics['cost']['train']['mae']:,.2f}")
        print(f"    RMSE: ${self.metrics['cost']['train']['rmse']:,.2f}")
        print(f"    R²:   {self.metrics['cost']['train']['r2']:.3f}")
        print(f"\n  Test Set:")
        print(f"    MAE:  ${self.metrics['cost']['test']['mae']:,.2f}")
        print(f"    RMSE: ${self.metrics['cost']['test']['rmse']:,.2f}")
        print(f"    R²:   {self.metrics['cost']['test']['r2']:.3f}")
        
        # Feature importance
        if hasattr(self.cost_model, 'feature_importances_'):
            importances = self.cost_model.feature_importances_
            top_features_idx = np.argsort(importances)[-5:]
            print(f"\n🔍 TOP 5 COST DRIVERS:")
            for idx in reversed(top_features_idx):
                print(f"    • {X_cost.columns[idx]}: {importances[idx]:.3f}")
        
        print("="*70)
        print("✅ Cost model trained successfully!")
        print("="*70)
        
        return self
    
    def predict(self, X_delay, X_cost, project_info=None):
        """
        Make predictions for new projects.
        
        Args:
            X_delay: Delay features from Pipeline 2a
            X_cost: Cost features from Pipeline 2b
            project_info: Original project data for context
        
        Returns:
            Dictionary with predictions and insights
        """
        print("\n" + "🔮 "+"="*68)
        print("GENERATING PREDICTIONS")
        print("="*70)
        
        if self.delay_model is None or self.cost_model is None:
            raise ValueError("Models not trained! Run train_delay_model() and train_cost_model() first.")
        
        # Make predictions
        delay_predictions = self.delay_model.predict(X_delay)
        cost_predictions = self.cost_model.predict(X_cost)
        
        # Store predictions
        self.predictions = {
            'delays': delay_predictions,
            'costs': cost_predictions,
            'count': len(delay_predictions)
        }
        
        print(f"✅ Generated predictions for {len(delay_predictions)} projects")
        print(f"\n📊 PREDICTION SUMMARY:")
        print(f"  Avg Predicted Delay: {np.mean(delay_predictions):.1f} days")
        print(f"  Avg Predicted Cost Impact: ${np.mean(cost_predictions):,.2f}")
        print(f"  High Risk Projects (>15 day delay): {sum(delay_predictions > 15)}")
        
        return self
    
    def generate_insights(self, project_info=None):
        """
        Generate actionable insights for PMs.
        
        Creates risk assessments, recommendations, and visualizations.
        """
        print("\n" + "💡 "+"="*68)
        print("GENERATING INSIGHTS")
        print("="*70)
        
        if not self.predictions:
            raise ValueError("No predictions available! Run predict() first.")
        
        delays = self.predictions['delays']
        costs = self.predictions['costs']
        
        # Risk categorization
        def categorize_risk(delay):
            if delay < 5:
                return 'Low'
            elif delay < 15:
                return 'Medium'
            elif delay < 30:
                return 'High'
            else:
                return 'Critical'
        
        risk_levels = [categorize_risk(d) for d in delays]
        
        # Generate insights
        self.insights = {
            'risk_distribution': {
                'Low': risk_levels.count('Low'),
                'Medium': risk_levels.count('Medium'),
                'High': risk_levels.count('High'),
                'Critical': risk_levels.count('Critical')
            },
            'total_predicted_delay': sum(delays),
            'total_predicted_cost_impact': sum(costs),
            'avg_delay': np.mean(delays),
            'avg_cost_impact': np.mean(costs),
            'highest_risk_indices': np.argsort(delays)[-5:][::-1].tolist()
        }
        
        # Display insights
        print("\n🎯 RISK ANALYSIS:")
        for risk, count in self.insights['risk_distribution'].items():
            percentage = (count / len(delays)) * 100
            print(f"  {risk:10s}: {count:3d} projects ({percentage:5.1f}%)")
        
        print(f"\n💰 FINANCIAL IMPACT:")
        print(f"  Total Predicted Delays: {self.insights['total_predicted_delay']:.0f} days")
        print(f"  Total Cost Impact: ${self.insights['total_predicted_cost_impact']:,.2f}")
        print(f"  Avg Cost per Project: ${self.insights['avg_cost_impact']:,.2f}")
        
        print(f"\n⚠️  TOP 5 HIGHEST RISK PROJECTS:")
        for i, idx in enumerate(self.insights['highest_risk_indices'], 1):
            print(f"  {i}. Project #{idx}: {delays[idx]:.1f} days delay, ${costs[idx]:,.2f} cost impact")
        
        print("\n💡 RECOMMENDATIONS:")
        high_risk_count = self.insights['risk_distribution']['High'] + self.insights['risk_distribution']['Critical']
        if high_risk_count > len(delays) * 0.2:
            print("  ⚠️  HIGH RISK PORTFOLIO - Consider:")
            print("     • Increasing buffer time by 20-30%")
            print("     • Securing backup suppliers")
            print("     • Adding weather contingencies")
        else:
            print("  ✅ MANAGEABLE RISK PORTFOLIO")
            print("     • Current planning appears adequate")
            print("     • Focus mitigation on high-risk projects")
        
        print("="*70)
        print("✅ Insights generated!")
        print("="*70)
        
        return self
    
    def save_models(self, delay_path='trained_delay_model.pkl', cost_path='trained_cost_model.pkl'):
        """Save trained models to disk."""
        if self.delay_model:
            with open(delay_path, 'wb') as f:
                pickle.dump(self.delay_model, f)
            print(f"✅ Delay model saved to {delay_path}")
        
        if self.cost_model:
            with open(cost_path, 'wb') as f:
                pickle.dump(self.cost_model, f)
            print(f"✅ Cost model saved to {cost_path}")
    
    def load_models(self, delay_path='trained_delay_model.pkl', cost_path='trained_cost_model.pkl'):
        """Load trained models from disk."""
        try:
            with open(delay_path, 'rb') as f:
                self.delay_model = pickle.load(f)
            print(f"✅ Delay model loaded from {delay_path}")
        except:
            print(f"⚠️  Could not load delay model from {delay_path}")
        
        try:
            with open(cost_path, 'rb') as f:
                self.cost_model = pickle.load(f)
            print(f"✅ Cost model loaded from {cost_path}")
        except:
            print(f"⚠️  Could not load cost model from {cost_path}")
    
    def export_results(self, output_path='predictions_output.csv'):
        """Export predictions and insights to CSV."""
        if not self.predictions:
            print("⚠️  No predictions to export")
            return
        
        results_df = pd.DataFrame({
            'Predicted_Delay_Days': self.predictions['delays'],
            'Predicted_Cost_Impact': self.predictions['costs'],
            'Risk_Level': [self._categorize_risk(d) for d in self.predictions['delays']]
        })
        
        results_df.to_csv(output_path, index=False)
        print(f"✅ Results exported to {output_path}")
    
    def _categorize_risk(self, delay):
        """Helper to categorize risk level."""
        if delay < 5:
            return 'Low'
        elif delay < 15:
            return 'Medium'
        elif delay < 30:
            return 'High'
        else:
            return 'Critical'
    
    def get_metrics(self):
        """Return model performance metrics."""
        return self.metrics
    
    def get_predictions(self):
        """Return predictions."""
        return self.predictions
    
    def get_insights(self):
        """Return insights."""
        return self.insights


# Complete End-to-End Example
if __name__ == "__main__":
    print("\n" + "🚀 "+"="*68)
    print("COMPLETE 3-PIPELINE SYSTEM DEMONSTRATION")
    print("="*70 + "\n")
    
    # PIPELINE 1: Universal Adapter
    from universal_data_adapter import UniversalDataAdapter
    adapter = UniversalDataAdapter()
    adapter.run_full_adapter('your_construction_data.csv')
    standardized_data = adapter.get_standardized_data()
    
    # PIPELINE 2a: Delay Engine
    from delay_cost_engines import DelayPreprocessingEngine
    delay_engine = DelayPreprocessingEngine()
    delay_engine.process(standardized_data)
    X_delay, y_delay = delay_engine.get_features()
    
    # PIPELINE 2b: Cost Engine
    from delay_cost_engines import CostPreprocessingEngine
    cost_engine = CostPreprocessingEngine()
    cost_engine.process(standardized_data)
    X_cost, y_cost = cost_engine.get_features()
    
    # PIPELINE 3: Master Training & Prediction
    master = MasterPredictionPipeline()
    
    # Train models
    master.train_delay_model(X_delay, y_delay, model_type='random_forest')
    master.train_cost_model(X_cost, y_cost, model_type='random_forest')
    
    # Make predictions
    master.predict(X_delay, X_cost)
    
    # Generate insights
    master.generate_insights()
    
    # Save everything
    master.save_models()
    master.export_results()
    
    print("\n" + "="*70)
    print("🎉 COMPLETE! Your AI prediction system is ready!")
    print("="*70)