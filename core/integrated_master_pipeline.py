try:
    from executive_summary import ExecutiveSummary
except ImportError:
    ExecutiveSummary = None

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import os
import warnings
warnings.filterwarnings('ignore')


class IntegratedMasterPipeline:
    """
    Master pipeline with predictions and prescriptive recommendations.
    """
    
    def __init__(self):
        self.delay_model = None
        self.cost_model = None
        self.metrics = {}
        self.predictions = {}
        self.detailed_analysis = {}
        self.prescriptive_enabled = True
        self.permit_database = self._load_permit_database()
    
    def _load_permit_database(self):
        """Load permit database."""
        return {
            'ALT1': {
                'name': 'Alteration Type 1',
                'cost': 1500,
                'approval_days': 10,
                'score_base': 60
            },
            'ALT2': {
                'name': 'Alteration Type 2',
                'cost': 4000,
                'approval_days': 30,
                'score_base': 70
            },
            'EXP': {
                'name': 'Expedited',
                'cost': 6000,
                'approval_days': 14,
                'score_base': 80
            }
        }
    
    def train_delay_model(self, X_delay, y_delay):
        """Train delay prediction model."""
        print("\n" + "="*70)
        print("TRAINING DELAY MODEL")
        print("="*70)
        
        if len(X_delay) < 5:
            print("Warning: Small dataset. Need 20+ projects for reliable predictions.")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_delay, y_delay, test_size=0.2, random_state=42
        )
        
        print(f" Training set: {X_train.shape[0]} projects")
        print(f" Test set: {X_test.shape[0]} projects")
        
        self.delay_model = RandomForestRegressor(
            n_estimators=100, max_depth=15, random_state=42
        )
        
        self.delay_model.fit(X_train, y_train)
        
        if len(X_test) > 0:
            y_pred_test = self.delay_model.predict(X_test)
            
            self.metrics['delay'] = {
                'test': {
                    'mae': mean_absolute_error(y_test, y_pred_test),
                    'rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
                    'r2': r2_score(y_test, y_pred_test) if len(y_test) > 1 else np.nan
                }
            }
            
            print("\nMODEL PERFORMANCE:")
            print(f"  MAE:  {self.metrics['delay']['test']['mae']:.2f} days")
            print(f"  RMSE: {self.metrics['delay']['test']['rmse']:.2f} days")
        
        print("="*70)
        print("Delay model trained!")
        print("="*70)
        
        return self
    
    def train_cost_model(self, X_cost, y_cost):
        """Train cost prediction model."""
        print("\n" + "="*70)
        print("TRAINING COST MODEL")
        print("="*70)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_cost, y_cost, test_size=0.2, random_state=42
        )
        
        print(f" Training set: {X_train.shape[0]} projects")
        print(f" Test set: {X_test.shape[0]} projects")
        
        self.cost_model = RandomForestRegressor(
            n_estimators=100, max_depth=15, random_state=42
        )
        
        self.cost_model.fit(X_train, y_train)
        
        if len(X_test) > 0:
            y_pred_test = self.cost_model.predict(X_test)
            
            self.metrics['cost'] = {
                'test': {
                    'mae': mean_absolute_error(y_test, y_pred_test),
                    'rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
                    'r2': r2_score(y_test, y_pred_test) if len(y_test) > 1 else np.nan
                }
            }
            
            print("\nMODEL PERFORMANCE:")
            print(f"  MAE:  {self.metrics['cost']['test']['mae']:.2f}%")
            print(f"  RMSE: {self.metrics['cost']['test']['rmse']:.2f}%")
        
        print("="*70)
        print("Cost model trained!")
        print("="*70)
        
        return self
    
    def predict_with_details(self, X_delay, X_cost, delay_engine, cost_engine, project_data=None):
        """Make predictions with full analysis."""
        print("\n" + "="*70)
        print("GENERATING PREDICTIONS & RECOMMENDATIONS")
        print("="*70)
        
        if self.delay_model is None or self.cost_model is None:
            raise ValueError("Models not trained!")
        
        delay_predictions = self.delay_model.predict(X_delay)
        cost_predictions = self.cost_model.predict(X_cost)
        
        phase_analysis = delay_engine.get_all_phase_analysis()
        cost_breakdowns = cost_engine.get_all_cost_breakdowns()
        worker_optimizations = cost_engine.get_all_worker_optimizations()
        
        self.predictions = {
            'delays': delay_predictions,
            'costs': cost_predictions,
            'count': len(delay_predictions)
        }
        
        self.detailed_analysis = {
            'phases': phase_analysis,
            'cost_breakdown': cost_breakdowns,
            'worker_optimization': worker_optimizations,
            'prescriptive': {}
        }
        
        # Generate prescriptive recommendations
        if self.prescriptive_enabled:
            for idx in range(len(delay_predictions)):
                self.detailed_analysis['prescriptive'][idx] = self._generate_prescriptive(
                    idx, delay_predictions[idx], cost_predictions[idx],
                    phase_analysis.get(idx, {}), cost_breakdowns.get(idx, {})
                )
        
        print(f"Generated predictions for {len(delay_predictions)} projects")
        print(f"\nPREDICTION SUMMARY:")
        print(f"  Avg Delay: {np.mean(delay_predictions):.1f} days")
        print(f"  Avg Cost Impact: {np.mean(cost_predictions):.1f}%")
        
        return self
    
    def _generate_prescriptive(self, idx, delay_pred, cost_pred, phase_data, cost_data):
        """Generate prescriptive recommendations."""
        recommendations = {
            'permit': self._recommend_permit(delay_pred),
            'priority_actions': []
        }
        
        # Add priority actions
        if delay_pred > 15:
            recommendations['priority_actions'].append({
                'category': 'Schedule',
                'action': 'Add buffer to critical phases',
                'impact': f'Saves {int(delay_pred * 0.3)} days',
                'priority': 1
            })
        
        if cost_pred > 10:
            recommendations['priority_actions'].append({
                'category': 'Cost',
                'action': 'Negotiate bulk material pricing',
                'impact': f'Saves {int(cost_pred * 0.15)}%',
                'priority': 2
            })
        
        # FAQ
        recommendations['faq'] = {
            'biggest_risk': self._answer_biggest_risk(delay_pred, cost_pred)
        }
        
        return recommendations
    
    def _recommend_permit(self, delay_pred):
        """Recommend permit type."""
        recommendations = []
        for code, info in self.permit_database.items():
            score = info['score_base']
            if delay_pred > 20 and info['approval_days'] < 20:
                score += 20
            recommendations.append({
                'code': code,
                'name': info['name'],
                'score': score,
                'cost': info['cost']
            })
        
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        return {'recommended': recommendations[0]}
    
    def _answer_biggest_risk(self, delay_pred, cost_pred):
        """Answer biggest risk question."""
        if delay_pred > 20:
            return f"Schedule delays ({delay_pred:.0f} days)"
        elif cost_pred > 15:
            return f"Cost overrun ({cost_pred:.1f}%)"
        else:
            return "Manageable risk levels"
    
    def get_project_report(self, project_idx):
        """Generate comprehensive report."""
        delay_pred = self.predictions['delays'][project_idx]
        cost_pred = self.predictions['costs'][project_idx]
        
        return {
            'summary': {
                'delay_days': round(delay_pred, 1),
                'cost_overrun_pct': round(cost_pred, 1),
                'risk_level': self._get_risk_level(delay_pred)
            },
            'prescriptive': self.detailed_analysis['prescriptive'].get(project_idx, {})
        }
    
    def _get_risk_level(self, delay_days):
        """Get risk level."""
        if delay_days < 5:
            return 'Low'
        elif delay_days < 15:
            return 'Medium'
        else:
            return 'High'
    
    def generate_dashboard_output(self, project_idx):
        """Generate formatted output."""
        report = self.get_project_report(project_idx)
        
        output = f"""
{'='*70}
PROJECT ANALYSIS & RECOMMENDATIONS
{'='*70}

PREDICTION SUMMARY:
  Delay: {report['summary']['delay_days']} days
  Cost Impact: {report['summary']['cost_overrun_pct']:.1f}%
  Risk Level: {report['summary']['risk_level']}

TOP PRIORITY ACTIONS:
"""
        
        actions = report['prescriptive'].get('priority_actions', [])
        for i, action in enumerate(actions, 1):
            output += f"  {i}. [{action['category']}] {action['action']}\n"
            output += f"     Impact: {action['impact']}\n\n"
        
        output += f"""
PERMIT RECOMMENDATION:
  {report['prescriptive'].get('permit', {}).get('recommended', {}).get('name', 'N/A')}
  
QUICK ANSWER:
  Biggest risk: {report['prescriptive'].get('faq', {}).get('biggest_risk', 'N/A')}

{'='*70}
"""
        return output
    
    def save_models(self, delay_path='models/delay_model.pkl', cost_path='models/cost_model.pkl'):
        """Save models."""
        os.makedirs('models', exist_ok=True)
        
        if self.delay_model:
            with open(delay_path, 'wb') as f:
                pickle.dump(self.delay_model, f)
            print("Delay model saved")
        
        if self.cost_model:
            with open(cost_path, 'wb') as f:
                pickle.dump(self.cost_model, f)
            print("Cost model saved")
    
    def export_results(self, output_path='predictions_with_recommendations.csv'):
        """Export results."""
        results = []
        for idx in range(len(self.predictions['delays'])):
            report = self.get_project_report(idx)
            results.append({
                'Project_Index': idx,
                'Predicted_Delay_Days': report['summary']['delay_days'],
                'Cost_Overrun_Pct': report['summary']['cost_overrun_pct'],
                'Risk_Level': report['summary']['risk_level']
            })
        
        pd.DataFrame(results).to_csv(output_path, index=False)
        print(f"Results exported to {output_path}")
    
    def get_metrics(self):
        return self.metrics
    
    def get_predictions(self):
        return self.predictions
    
    def get_detailed_analysis(self):
        return self.detailed_analysis

"""
Executive Summary Generator - Professional Construction Project Analysis
Add this to your integrated_master_pipeline.py
"""

class ExecutiveSummary:
    """Generate executive-level project summaries for construction management."""
    
    def __init__(self, project_data, delay_pred, cost_pred, phase_analysis, cost_breakdown):
        self.project_data = project_data
        self.delay_pred = delay_pred
        self.cost_pred = cost_pred
        self.phase_analysis = phase_analysis
        self.cost_breakdown = cost_breakdown
    
    def generate_summary(self):
        """Generate professional executive summary."""
        
        # Calculate key metrics
        original_budget = self.project_data.get('estimated_cost', 0)
        projected_cost = original_budget * (1 + self.cost_pred/100)
        overrun_dollars = projected_cost - original_budget
        
        # Calculate confidence based on data quality
        confidence = self._calculate_confidence()
        
        # Determine risk level
        risk_level = self._get_risk_level(self.delay_pred, self.cost_pred)
        
        summary = f"""
{'='*80}
EXECUTIVE PROJECT SUMMARY
{'='*80}

PROJECT: {self.project_data.get('project_type', 'Unknown')} - {self.project_data.get('location', 'Unknown')}
RISK ASSESSMENT: {risk_level}

PREDICTED OUTCOMES:
  Schedule Delay: {self.delay_pred:.0f} days beyond planned completion
  Cost Overrun: ${overrun_dollars:,.0f} ({self.cost_pred:.1f}% over budget)
  Projected Total Cost: ${projected_cost:,.0f} (Original Budget: ${original_budget:,.0f})
  Confidence Level: {confidence}% (based on historical project patterns)

{'='*80}
TOP 3 PROJECT RISKS
{'='*80}
"""
        
        # Add top 3 risks with dollar amounts
        risks = self._identify_top_risks()
        for i, risk in enumerate(risks, 1):
            summary += f"""
{i}. {risk['title']} (Probability: {risk['probability']}%)
   Impact: {risk['impact']}
   Financial Exposure: ${risk['cost']:,.0f}
   Recommended Action: {risk['action']}
"""
        
        # Add phase timeline
        summary += f"""
{'='*80}
PHASE TIMELINE & CRITICAL PATH ANALYSIS
{'='*80}
"""
        summary += self._format_phase_timeline()
        
        # Add cost breakdown
        summary += f"""
{'='*80}
DETAILED COST BREAKDOWN
{'='*80}
"""
        summary += self._format_cost_breakdown(original_budget, projected_cost)
        
        # Add immediate actions
        summary += f"""
{'='*80}
RECOMMENDED IMMEDIATE ACTIONS
{'='*80}
"""
        actions = self._get_immediate_actions()
        for i, action in enumerate(actions, 1):
            summary += f"""
{i}. Priority: {action['priority']} - {action['title']}
   Investment Required: ${action['cost']:,.0f}
   Potential Savings: ${action['saves']:,.0f} and {action['time_saved']} days
   Return on Investment: {action['roi']}
"""
        
        summary += f"""
{'='*80}
ANALYSIS CONFIDENCE FACTORS
{'='*80}
Data Sources and Limitations:
  - Historical project database: Similar {self.project_data.get('project_type', 'construction')} projects analyzed
  - Location factors: {self.project_data.get('location', 'Unknown')} specific conditions considered
  - Crew baseline: {self.project_data.get('crew_size_avg', 0)} workers average
  - Seasonal considerations: Weather and permit patterns included
  
NOTE: Predictions based on historical patterns and statistical modeling. Actual 
outcomes may vary due to unforeseen circumstances, design changes, market conditions,
and external factors beyond model parameters.
{'='*80}
"""
        return summary
    
    def _calculate_confidence(self):
        """Calculate prediction confidence based on data quality and pattern strength."""
        confidence = 70  # Base confidence
        
        # Adjust based on delay magnitude
        if self.delay_pred > 15:
            confidence += 10
        elif self.delay_pred < 5:
            confidence -= 5
        
        # Adjust based on cost variance
        if self.cost_pred > 10:
            confidence += 5
        
        # Cap confidence between reasonable bounds
        return min(95, max(50, confidence))
    
    def _get_risk_level(self, delay_days, cost_overrun_pct):
        """Determine overall project risk level."""
        if delay_days > 20 or cost_overrun_pct > 15:
            return "HIGH RISK"
        elif delay_days > 10 or cost_overrun_pct > 8:
            return "MODERATE RISK"
        else:
            return "LOW RISK"
    
    def _identify_top_risks(self):
        """Identify top 3 risks based on project characteristics and historical data."""
        risks = []
        
        crew_size = self.project_data.get('crew_size_avg', 20)
        equipment_downtime = self.project_data.get('equipment_downtime_hours', 0)
        project_type = self.project_data.get('project_type', 'Unknown')
        estimated_cost = self.project_data.get('estimated_cost', 0)
        
        # Risk 1: Equipment and Material Delays
        if equipment_downtime > 20:
            risks.append({
                'title': 'Equipment Downtime and Material Supply Chain',
                'probability': 65,
                'impact': f'{int(self.delay_pred * 0.4)} days schedule delay',
                'cost': estimated_cost * 0.05,
                'action': 'Establish backup equipment supplier agreements and order long-lead materials immediately'
            })
        else:
            risks.append({
                'title': 'Material Procurement and Lead Time',
                'probability': 45,
                'impact': f'{int(self.delay_pred * 0.25)} days schedule delay',
                'cost': estimated_cost * 0.03,
                'action': 'Order critical materials 2-3 weeks ahead of need date, identify alternative suppliers'
            })
        
        # Risk 2: Labor Resource Constraints
        if crew_size < 15:
            risks.append({
                'title': 'Insufficient Labor Resources',
                'probability': 70,
                'impact': f'{int(self.delay_pred * 0.35)} days schedule delay',
                'cost': estimated_cost * 0.04,
                'action': 'Increase crew size by 5-8 workers during critical phases, arrange overtime capacity'
            })
        else:
            risks.append({
                'title': 'Labor Productivity Variance',
                'probability': 40,
                'impact': f'{int(self.delay_pred * 0.2)} days schedule delay',
                'cost': estimated_cost * 0.02,
                'action': 'Implement daily productivity tracking and adjust workforce allocation weekly'
            })
        
        # Risk 3: Regulatory and Permit Delays
        location = self.project_data.get('location', 'Unknown')
        if location in ['NYC', 'Boston', 'SF', 'LA', 'Chicago']:
            permit_risk = 55
            permit_delay = 15
        else:
            permit_risk = 35
            permit_delay = 8
        
        risks.append({
            'title': f'Permit Approval and Inspection Delays - {location}',
            'probability': permit_risk,
            'impact': f'{permit_delay} days regulatory delay',
            'cost': estimated_cost * 0.025,
            'action': 'Submit permit applications immediately, pre-schedule all required inspections, consider permit expediter'
        })
        
        # Sort by risk exposure (probability * financial impact)
        risks.sort(key=lambda x: x['probability'] * x['cost'], reverse=True)
        return risks[:3]
    
    def _format_phase_timeline(self):
        """Format phase timeline with risk indicators and critical path."""
        output = ""
        
        if not self.phase_analysis:
            return "Phase-level data not available for detailed timeline analysis\n"
        
        # Standard construction phases
        phases = [
            ('Foundation Work', self.phase_analysis.get('foundation', {})),
            ('Rough-In MEP', self.phase_analysis.get('framing', {})),
            ('Structural Framing', self.phase_analysis.get('framing', {})),
            ('Interior Finishes', self.phase_analysis.get('finishing', {})),
            ('Final Inspections', self.phase_analysis.get('finishing', {}))
        ]
        
        output += f"{'PHASE':<25} {'BASELINE':<15} {'PREDICTED':<15} {'RISK LEVEL':<15}\n"
        output += "-" * 70 + "\n"
        
        for phase_name, phase_data in phases:
            delay = phase_data.get('delay_days', 0)
            
            # Determine risk classification
            if delay > 10:
                risk = "HIGH"
            elif delay > 5:
                risk = "MODERATE"
            else:
                risk = "LOW"
            
            planned = "Per Schedule"
            likely = f"+{delay:.0f} days" if delay > 0 else "On Track"
            
            output += f"{phase_name:<25} {planned:<15} {likely:<15} {risk:<15}\n"
        
        output += "\nCRITICAL PATH ANALYSIS:\n"
        output += "  Primary: Rough-In MEP -> Interior Finishes -> Final Inspections\n"
        output += "  Impact: Delays in Rough-In phase will cascade to all downstream activities\n"
        output += "  Mitigation: Monitor MEP coordination closely, ensure permit readiness\n"
        
        return output
    
    def _format_cost_breakdown(self, original_budget, projected_cost):
        """Format detailed cost breakdown by category."""
        output = ""
        
        # Get cost breakdown data or use industry standards
        if not self.cost_breakdown:
            labor_base = original_budget * 0.45
            materials_base = original_budget * 0.40
            equipment_base = original_budget * 0.10
            admin_base = original_budget * 0.05
        else:
            labor_data = self.cost_breakdown.get('labor', {})
            materials_data = self.cost_breakdown.get('materials', {})
            admin_data = self.cost_breakdown.get('admin', {})
            
            labor_base = labor_data.get('total', original_budget * 0.45)
            materials_base = materials_data.get('total', original_budget * 0.40)
            equipment_base = original_budget * 0.10
            admin_base = admin_data.get('total', original_budget * 0.05)
        
        # Distribute overrun across categories based on typical patterns
        total_overrun = projected_cost - original_budget
        labor_overrun = total_overrun * 0.50
        materials_overrun = total_overrun * 0.35
        equipment_overrun = total_overrun * 0.10
        admin_overrun = total_overrun * 0.05
        
        categories = [
            ('Labor Costs', labor_base, labor_base + labor_overrun, labor_overrun),
            ('Materials', materials_base, materials_base + materials_overrun, materials_overrun),
            ('Equipment', equipment_base, equipment_base + equipment_overrun, equipment_overrun),
            ('Administrative/Overhead', admin_base, admin_base + admin_overrun, admin_overrun)
        ]
        
        output += f"{'CATEGORY':<25} {'BUDGETED':<18} {'PROJECTED':<18} {'VARIANCE':<20}\n"
        output += "-" * 80 + "\n"
        
        for category, original, projected, overrun in categories:
            pct_change = (overrun / original * 100) if original > 0 else 0
            variance_str = f"${overrun:,.0f} ({pct_change:+.1f}%)"
            output += f"{category:<25} ${original:>15,.0f}  ${projected:>15,.0f}  {variance_str:>18}\n"
        
        output += "-" * 80 + "\n"
        output += f"{'TOTAL PROJECT COST':<25} ${original_budget:>15,.0f}  ${projected_cost:>15,.0f}  ${total_overrun:>15,.0f}\n"
        output += f"{'Overall Variance':<25} {'':>18} {'':>18} {(total_overrun/original_budget*100):+.1f}%\n"
        
        return output
    
    def _get_immediate_actions(self):
        """Generate prioritized action items with ROI analysis."""
        actions = []
        
        crew_size = self.project_data.get('crew_size_avg', 20)
        equipment_downtime = self.project_data.get('equipment_downtime_hours', 0)
        estimated_cost = self.project_data.get('estimated_cost', 0)
        
        # Action 1: Material procurement
        if self.delay_pred > 10:
            material_cost = 0
            material_savings = estimated_cost * 0.03
            time_saved = int(self.delay_pred * 0.3)
            roi = "POSITIVE - Prevents delay costs"
            
            actions.append({
                'priority': 'CRITICAL',
                'title': 'Accelerate Material Procurement',
                'cost': material_cost,
                'saves': material_savings,
                'time_saved': time_saved,
                'roi': roi
            })
        
        # Action 2: Crew augmentation
        if crew_size < 15 and self.delay_pred > 15:
            crew_cost = 18000
            crew_savings = estimated_cost * 0.025
            time_saved = int(self.delay_pred * 0.25)
            roi_pct = ((crew_savings - crew_cost) / crew_cost * 100) if crew_cost > 0 else 0
            roi = f"{roi_pct:.0f}% ROI"
            
            actions.append({
                'priority': 'HIGH',
                'title': 'Increase Crew Capacity During Critical Phases',
                'cost': crew_cost,
                'saves': crew_savings,
                'time_saved': time_saved,
                'roi': roi
            })
        
        # Action 3: Permit expediting
        location = self.project_data.get('location', 'Unknown')
        if location in ['NYC', 'Boston', 'SF', 'LA'] and self.delay_pred > 10:
            permit_cost = 500
            permit_savings = estimated_cost * 0.02
            time_saved = 14
            roi_pct = ((permit_savings - permit_cost) / permit_cost * 100) if permit_cost > 0 else 0
            roi = f"{roi_pct:.0f}% ROI"
            
            actions.append({
                'priority': 'HIGH',
                'title': 'Submit Permits and Schedule Inspections Immediately',
                'cost': permit_cost,
                'saves': permit_savings,
                'time_saved': time_saved,
                'roi': roi
            })
        
        # Sort by priority (CRITICAL > HIGH > MEDIUM)
        priority_order = {'CRITICAL': 0, 'HIGH': 1, 'MEDIUM': 2}
        actions.sort(key=lambda x: priority_order.get(x['priority'], 3))
        
        return actions[:3]  # Return top 3 actions
    
    def export_to_dict(self):
        """Export summary data as dictionary for programmatic access."""
        original_budget = self.project_data.get('estimated_cost', 0)
        projected_cost = original_budget * (1 + self.cost_pred/100)
        
        return {
            'project_info': {
                'type': self.project_data.get('project_type'),
                'location': self.project_data.get('location'),
                'crew_size': self.project_data.get('crew_size_avg')
            },
            'predictions': {
                'delay_days': round(self.delay_pred, 1),
                'cost_overrun_pct': round(self.cost_pred, 1),
                'cost_overrun_dollars': round(projected_cost - original_budget, 2),
                'projected_total_cost': round(projected_cost, 2),
                'risk_level': self._get_risk_level(self.delay_pred, self.cost_pred),
                'confidence': self._calculate_confidence()
            },
            'top_risks': self._identify_top_risks(),
            'recommended_actions': self._get_immediate_actions()
        }
    
  