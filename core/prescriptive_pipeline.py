import pandas as pd
import numpy as np
from typing import Dict, List, Any

class PrescriptivePipeline:
    """
    Prescriptive Pipeline - Actionable Recommendations & FAQ Engine
    
    Generates:
    - Permit recommendations
    - Schedule optimization suggestions
    - Worker allocation strategies
    - Cost reduction opportunities
    - Risk mitigation actions
    - Answers to common PM questions
    """
    
    def __init__(self):
        self.recommendations = {}
        self.permit_database = self._load_permit_data()
        self.faq_responses = {}
    
    def _load_permit_data(self):
        """
        Load permit database (placeholder - will be populated with real NYC data).
        """
        # NYC Residential Permits (example structure)
        permits = {
            'ALT1': {
                'name': 'Alteration Type 1 (Minor)',
                'cost': 1500,
                'approval_days': 10,
                'flexibility': 'Low',
                'best_for': ['Small renovations', 'Non-structural'],
                'restrictions': ['No structural changes', 'No plumbing/electrical']
            },
            'ALT2': {
                'name': 'Alteration Type 2 (Major)',
                'cost': 4000,
                'approval_days': 30,
                'flexibility': 'Medium',
                'best_for': ['Major renovations', 'Structural changes'],
                'restrictions': ['Requires engineer approval']
            },
            'NB': {
                'name': 'New Building',
                'cost': 8000,
                'approval_days': 60,
                'flexibility': 'High',
                'best_for': ['New construction', 'Ground-up builds'],
                'restrictions': ['Full plan review required']
            },
            'EXP': {
                'name': 'Expedited Processing',
                'cost': 6000,
                'approval_days': 14,
                'flexibility': 'High',
                'best_for': ['Time-sensitive projects'],
                'restrictions': ['Additional $2K fee', 'Complete docs required']
            }
        }
        return permits
    
    def analyze_project(self, project_data, delay_prediction, cost_prediction, 
                       delay_breakdown, cost_breakdown, worker_optimization):
        """
        Comprehensive project analysis for prescriptive recommendations.
        
        Args:
            project_data: Original project details
            delay_prediction: Predicted delay in days
            cost_prediction: Predicted cost overrun %
            delay_breakdown: Phase-level delay analysis
            cost_breakdown: Category-level cost analysis
            worker_optimization: Worker allocation recommendations
        
        Returns:
            Complete prescriptive analysis with recommendations
        """
        
        self.recommendations = {
            'permit_optimization': self._recommend_permit(project_data, delay_prediction),
            'schedule_optimization': self._optimize_schedule(delay_breakdown, delay_prediction),
            'worker_allocation': self._optimize_workers(worker_optimization),
            'cost_reduction': self._reduce_costs(cost_breakdown, cost_prediction),
            'risk_mitigation': self._mitigate_risks(delay_prediction, cost_prediction, delay_breakdown),
            'priority_actions': []
        }
        
        # Prioritize all recommendations
        self._prioritize_actions()
        
        # Generate FAQ responses based on analysis
        self._generate_faq(project_data, delay_prediction, cost_prediction)
        
        return self.recommendations
    
    def _recommend_permit(self, project_data, delay_prediction):
        """Recommend optimal permit type."""
        project_type = project_data.get('project_type', 'Residential')
        planned_duration = project_data.get('planned_duration', 120)
        budget = project_data.get('planned_cost', 500000)
        
        recommendations = []
        
        # Analyze each permit option
        for permit_code, permit_info in self.permit_database.items():
            # Calculate impact
            time_vs_standard = 30 - permit_info['approval_days']  # vs standard 30 days
            cost = permit_info['cost']
            
            # Score based on project needs
            score = 0
            reasons = []
            
            if delay_prediction > 20 and permit_info['approval_days'] < 20:
                score += 30
                reasons.append(f"Fast approval ({permit_info['approval_days']} days vs 30)")
            
            if cost < 5000:
                score += 20
                reasons.append(f"Cost-effective (${cost:,})")
            
            if permit_info['flexibility'] == 'High' and delay_prediction > 15:
                score += 25
                reasons.append("High flexibility for changes")
            
            if project_type in ' '.join(permit_info['best_for']):
                score += 25
                reasons.append(f"Optimized for {project_type}")
            
            recommendations.append({
                'permit_code': permit_code,
                'permit_name': permit_info['name'],
                'score': score,
                'cost': cost,
                'approval_days': permit_info['approval_days'],
                'time_saved': time_vs_standard,
                'reasons': reasons,
                'restrictions': permit_info['restrictions']
            })
        
        # Sort by score
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        
        return {
            'recommended': recommendations[0],
            'alternatives': recommendations[1:3],
            'comparison': self._compare_permits(recommendations[:3])
        }
    
    def _compare_permits(self, permits):
        """Generate permit comparison table."""
        comparison = []
        for p in permits:
            comparison.append({
                'name': p['permit_name'],
                'cost': f"${p['cost']:,}",
                'approval_time': f"{p['approval_days']} days",
                'time_saved': f"{p['time_saved']} days" if p['time_saved'] > 0 else "Standard",
                'score': f"{p['score']}/100"
            })
        return comparison
    
    def _optimize_schedule(self, delay_breakdown, total_delay):
        """Generate schedule optimization recommendations."""
        recommendations = []
        
        # Foundation phase optimization
        foundation_delay = delay_breakdown.get('foundation', {}).get('delay_days', 0)
        if foundation_delay > 5:
            recommendations.append({
                'phase': 'Foundation',
                'action': 'Pre-order concrete and schedule inspections 2 weeks in advance',
                'impact_days': foundation_delay,
                'impact_cost': f"Saves ${foundation_delay * 500:,}",
                'priority': 'High',
                'effort': 'Low'
            })
        
        # Framing phase optimization
        framing_delay = delay_breakdown.get('framing', {}).get('delay_days', 0)
        if framing_delay > 6:
            recommendations.append({
                'phase': 'Framing',
                'action': 'Increase crew during critical framing phase',
                'impact_days': int(framing_delay * 0.6),
                'impact_cost': f"Net savings ${framing_delay * 300:,}",
                'priority': 'High',
                'effort': 'Medium'
            })
        
        # Finishing phase optimization
        finishing_delay = delay_breakdown.get('finishing', {}).get('delay_days', 0)
        if finishing_delay > 4:
            recommendations.append({
                'phase': 'Finishing',
                'action': 'Parallel task scheduling for finishing work',
                'impact_days': int(finishing_delay * 0.4),
                'impact_cost': f"Saves ${finishing_delay * 400:,}",
                'priority': 'Medium',
                'effort': 'Low'
            })
        
        # Add buffer recommendation
        if total_delay > 15:
            buffer_days = int(total_delay * 0.3)
            recommendations.append({
                'phase': 'Overall',
                'action': f'Add {buffer_days}-day buffer to schedule',
                'impact_days': 0,
                'impact_cost': 'Prevents penalties',
                'priority': 'High',
                'effort': 'Low'
            })
        
        return recommendations
    
    def _optimize_workers(self, worker_optimization):
        """Generate worker allocation recommendations."""
        recommendations = []
        
        if worker_optimization.get('recommendation') == 'increase_critical_phase':
            phases = worker_optimization.get('phases', {})
            net_benefit = worker_optimization.get('net_benefit', 0)
            
            for phase_name, phase_data in phases.items():
                current = phase_data.get('current', 0)
                optimal = phase_data.get('optimal', 0)
                
                if optimal != current:
                    change = optimal - current
                    recommendations.append({
                        'phase': phase_name.title(),
                        'current_workers': current,
                        'recommended_workers': optimal,
                        'change': f"{'+' if change > 0 else ''}{change} workers",
                        'cost_impact': f"${abs(change * 250 * 30):,}",
                        'benefit': f"${net_benefit:,} net savings",
                        'reasoning': 'Accelerates critical path' if change > 0 else 'Reduces overhead'
                    })
        
        return recommendations
    
    def _reduce_costs(self, cost_breakdown, cost_prediction):
        """Generate cost reduction recommendations."""
        recommendations = []
        
        labor_cost = cost_breakdown.get('labor', {}).get('total', 0)
        material_cost = cost_breakdown.get('materials', {}).get('total', 0)
        
        # Labor cost reduction
        if labor_cost > 50000:
            recommendations.append({
                'category': 'Labor',
                'action': 'Negotiate crew rates or optimize worker allocation',
                'potential_savings': f"${labor_cost * 0.15:,.0f}",
                'savings_pct': '15%',
                'effort': 'Medium',
                'timeline': 'Before project start'
            })
        
        # Material cost reduction
        if material_cost > 30000:
            recommendations.append({
                'category': 'Materials',
                'action': 'Bulk purchase agreements with suppliers',
                'potential_savings': f"${material_cost * 0.20:,.0f}",
                'savings_pct': '20%',
                'effort': 'Low',
                'timeline': '2 weeks before material needs'
            })
        
        # Overall cost optimization
        if cost_prediction > 10:
            recommendations.append({
                'category': 'Overall',
                'action': 'Value engineering review',
                'potential_savings': f"${cost_breakdown.get('total_overrun', 0) * 0.25:,.0f}",
                'savings_pct': '25%',
                'effort': 'High',
                'timeline': 'Design phase'
            })
        
        return recommendations
    
    def _mitigate_risks(self, delay_prediction, cost_prediction, delay_breakdown):
        """Generate risk mitigation strategies."""
        risks = []
        
        # High delay risk
        if delay_prediction > 20:
            risks.append({
                'risk_type': 'Schedule Risk',
                'severity': 'High',
                'mitigation': [
                    'Add 20% contingency buffer to schedule',
                    'Implement weekly progress reviews',
                    'Pre-qualify backup suppliers'
                ],
                'impact': 'Reduces delay risk by 40%'
            })
        
        # Material delay risk
        foundation_delay = delay_breakdown.get('foundation', {}).get('delay_days', 0)
        if foundation_delay > 5:
            risks.append({
                'risk_type': 'Material Delivery Risk',
                'severity': 'Medium',
                'mitigation': [
                    'Order critical materials 3 weeks early',
                    'Establish backup supplier relationships',
                    'Monitor material lead times weekly'
                ],
                'impact': 'Prevents 60% of material delays'
            })
        
        # Cost overrun risk
        if cost_prediction > 15:
            risks.append({
                'risk_type': 'Budget Risk',
                'severity': 'High',
                'mitigation': [
                    'Lock in material pricing contracts',
                    'Set up change order approval process',
                    'Reserve 15% contingency fund'
                ],
                'impact': 'Limits overrun to planned contingency'
            })
        
        return risks
    
    def _prioritize_actions(self):
        """Prioritize all recommendations by impact and urgency."""
        all_actions = []
        
        # Collect all recommendations
        if 'permit_optimization' in self.recommendations:
            rec = self.recommendations['permit_optimization']['recommended']
            all_actions.append({
                'category': 'Permit',
                'action': f"Use {rec['permit_name']}",
                'impact_score': rec['score'],
                'time_saved': f"{rec['time_saved']} days",
                'cost_impact': f"${rec['cost']:,}",
                'priority': 'High' if rec['score'] > 70 else 'Medium'
            })
        
        for sched_rec in self.recommendations.get('schedule_optimization', []):
            all_actions.append({
                'category': 'Schedule',
                'action': sched_rec['action'],
                'impact_score': sched_rec.get('impact_days', 0) * 10,
                'time_saved': f"{sched_rec.get('impact_days', 0)} days",
                'cost_impact': sched_rec.get('impact_cost', 'N/A'),
                'priority': sched_rec.get('priority', 'Medium')
            })
        
        for worker_rec in self.recommendations.get('worker_allocation', []):
            all_actions.append({
                'category': 'Workers',
                'action': f"{worker_rec['phase']}: {worker_rec['change']}",
                'impact_score': 50,
                'time_saved': 'Varies',
                'cost_impact': worker_rec.get('benefit', 'N/A'),
                'priority': 'High'
            })
        
        # Sort by priority and impact
        all_actions.sort(key=lambda x: (x['priority'] == 'High', x['impact_score']), reverse=True)
        
        self.recommendations['priority_actions'] = all_actions[:5]  # Top 5
    
    def _generate_faq(self, project_data, delay_prediction, cost_prediction):
        """Generate FAQ responses based on project analysis."""
        self.faq_responses = {
            'biggest_risk': self._faq_biggest_risk(delay_prediction, cost_prediction),
            'reduce_delays': self._faq_reduce_delays(delay_prediction),
            'reduce_costs': self._faq_reduce_costs(cost_prediction),
            'worker_changes': self._faq_worker_changes(),
            'permit_choice': self._faq_permit_choice(),
            'timeline_realistic': self._faq_timeline(delay_prediction)
        }
    
    def _faq_biggest_risk(self, delay_pred, cost_pred):
        if delay_pred > 20:
            return f"Your biggest risk is schedule delays ({delay_pred:.0f} days predicted). Material delivery and weather are primary concerns."
        elif cost_pred > 15:
            return f"Your biggest risk is cost overrun ({cost_pred:.1f}% predicted). Focus on locking in material prices and optimizing labor."
        else:
            return "Your project has manageable risk levels. Main focus should be maintaining schedule adherence."
    
    def _faq_reduce_delays(self, delay_pred):
        if delay_pred > 15:
            return "To reduce delays: 1) Pre-order all materials 2 weeks early, 2) Schedule inspections in advance, 3) Add workers during critical phases, 4) Build in weather contingency."
        else:
            return "Your delay risk is moderate. Focus on proactive inspection scheduling and maintaining material lead times."
    
    def _faq_reduce_costs(self, cost_pred):
        if cost_pred > 10:
            return f"To reduce costs: 1) Negotiate bulk material pricing (save 20%), 2) Optimize worker allocation (save 15% labor), 3) Use faster permit (save on delay costs)."
        else:
            return "Your cost risk is low. Continue with current planning and monitor material prices."
    
    def _faq_worker_changes(self):
        worker_recs = self.recommendations.get('worker_allocation', [])
        if worker_recs:
            return f"Yes, consider: {worker_recs[0]['change']} during {worker_recs[0]['phase']} phase. This provides {worker_recs[0]['benefit']} in net benefit."
        return "Current worker allocation appears optimal."
    
    def _faq_permit_choice(self):
        permit_rec = self.recommendations.get('permit_optimization', {}).get('recommended', {})
        if permit_rec:
            return f"Recommended: {permit_rec['permit_name']} because it {', '.join(permit_rec['reasons'][:2])}."
        return "Standard permitting process is appropriate for your project."
    
    def _faq_timeline(self, delay_pred):
        if delay_pred > 20:
            return f"No, your timeline needs adjustment. Expect {delay_pred:.0f} additional days. Consider adding this buffer now to avoid client issues."
        elif delay_pred > 10:
            return f"Timeline is tight. {delay_pred:.0f} days of potential delay detected. Add contingency buffer."
        else:
            return f"Timeline is realistic with minor risk ({delay_pred:.0f} days potential delay). Maintain current schedule."
    
    def answer_question(self, question):
        """
        Answer PM's specific question based on analysis.
        
        Args:
            question: PM's question (string)
        
        Returns:
            Detailed answer based on project analysis
        """
        question_lower = question.lower()
        
        # Match question to FAQ
        if 'biggest risk' in question_lower or 'main concern' in question_lower:
            return self.faq_responses.get('biggest_risk', 'Analysis not yet complete.')
        
        elif 'reduce delay' in question_lower or 'prevent delay' in question_lower:
            return self.faq_responses.get('reduce_delays', 'Analysis not yet complete.')
        
        elif 'reduce cost' in question_lower or 'save money' in question_lower:
            return self.faq_responses.get('reduce_costs', 'Analysis not yet complete.')
        
        elif 'worker' in question_lower or 'crew' in question_lower:
            return self.faq_responses.get('worker_changes', 'Analysis not yet complete.')
        
        elif 'permit' in question_lower:
            return self.faq_responses.get('permit_choice', 'Analysis not yet complete.')
        
        elif 'timeline' in question_lower or 'schedule realistic' in question_lower:
            return self.faq_responses.get('timeline_realistic', 'Analysis not yet complete.')
        
        else:
            return "I can help with: biggest risks, reducing delays, reducing costs, worker allocation, permit selection, and timeline assessment. What would you like to know?"
    
    def get_full_report(self):
        """Generate complete prescriptive report."""
        return {
            'recommendations': self.recommendations,
            'faq': self.faq_responses
        }
    
    def get_priority_actions(self):
        """Get top priority actions only."""
        return self.recommendations.get('priority_actions', [])
    
    def get_permit_recommendation(self):
        """Get permit recommendation only."""
        return self.recommendations.get('permit_optimization', {})
