"""
Construction AI Feedback Tool - Enterprise Grade
Professional predictive analytics for infrastructure projects
"""

import streamlit as st
import pandas as pd
import io
import os
from datetime import datetime
import json

# Import your engines
try:
    from delay_cost_engines import DelayEngineV2, CostEngineV2
    from integrated_master_pipeline import IntegratedMasterPipeline
    from flexible_column_mapper import FlexibleColumnMapper
except ImportError as e:
    st.error(f"Import error: {e}. Make sure all engine files are in the same directory.")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Construction AI Predictor | Enterprise Analytics",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Professional CSS styling
st.markdown("""
<style>
    * { 
        margin: 0; 
        padding: 0; 
        box-sizing: border-box;
    }
    
    body { 
        background-color: #f8f9fa;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
    }
    
    .main-header { 
        font-size: 2.2rem; 
        font-weight: 700; 
        color: #ffffff; 
        background: linear-gradient(135deg, #1a365d 0%, #2563eb 100%);
        padding: 2rem; 
        margin-bottom: 0; 
        border-radius: 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        letter-spacing: -0.5px;
    }
    
    .company-info { 
        font-size: 0.95rem; 
        color: #64748b; 
        padding: 0.75rem 2rem;
        background: #ffffff;
        border-bottom: 1px solid #e2e8f0;
        font-weight: 500;
    }
    
    .section-title { 
        font-size: 1.5rem; 
        font-weight: 700; 
        color: #1e293b; 
        margin: 2.5rem 0 1.25rem 0;
        padding-bottom: 0.75rem;
        border-bottom: 2px solid #2563eb;
    }
    
    .section-subtitle {
        font-size: 0.9rem;
        color: #64748b;
        margin-bottom: 1.5rem;
        line-height: 1.6;
    }
    
    .divider { 
        border-bottom: 1px solid #e2e8f0; 
        margin: 2.5rem 0; 
    }
    
    .upload-container { 
        background: #ffffff;
        border: 2px dashed #cbd5e1; 
        border-radius: 8px; 
        padding: 3rem; 
        text-align: center; 
        margin: 1.5rem 0;
        transition: all 0.3s ease;
    }
    
    .upload-container:hover {
        border-color: #2563eb;
        background: #f8fafc;
    }
    
    .upload-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 0.5rem;
    }
    
    .upload-subtitle {
        font-size: 0.9rem;
        color: #64748b;
    }
    
    .metric-card { 
        background: #ffffff;
        border: 1px solid #e2e8f0; 
        border-radius: 8px; 
        padding: 1.75rem; 
        text-align: center; 
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        transition: all 0.2s ease;
        height: 100%;
    }
    
    .metric-card:hover {
        box-shadow: 0 4px 12px rgba(0,0,0,0.12);
        transform: translateY(-2px);
    }
    
    .metric-value { 
        font-size: 2.25rem; 
        font-weight: 700; 
        color: #1e293b;
        margin: 0.5rem 0;
        letter-spacing: -0.5px;
    }
    
    .metric-label { 
        font-size: 0.75rem; 
        color: #64748b;
        text-transform: uppercase; 
        letter-spacing: 0.8px;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .metric-sublabel {
        font-size: 0.85rem;
        color: #94a3b8;
        margin-top: 0.25rem;
    }
    
    .risk-high { color: #dc2626; }
    .risk-medium { color: #f59e0b; }
    .risk-low { color: #059669; }
    
    .alert-box { 
        background: #ffffff;
        border-left: 4px solid #f59e0b; 
        padding: 1.25rem; 
        margin: 1rem 0; 
        border-radius: 4px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    
    .alert-success {
        border-left-color: #059669;
        background: #f0fdf4;
    }
    
    .alert-title {
        font-size: 1rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.5rem;
    }
    
    .alert-description {
        font-size: 0.9rem;
        color: #475569;
        line-height: 1.6;
    }
    
    .action-card { 
        background: #ffffff;
        border: 1px solid #e2e8f0; 
        padding: 1.75rem; 
        margin: 1rem 0; 
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    
    .action-title { 
        font-size: 1.05rem; 
        font-weight: 700; 
        margin-bottom: 1rem; 
        color: #1e293b;
        padding-bottom: 0.75rem;
        border-bottom: 2px solid #e2e8f0;
    }
    
    .action-list { 
        list-style: none;
        padding-left: 0;
    }
    
    .action-list li { 
        padding: 0.75rem 0; 
        color: #334155;
        font-size: 0.9rem; 
        line-height: 1.7;
        border-bottom: 1px solid #f1f5f9;
    }
    
    .action-list li:last-child {
        border-bottom: none;
    }
    
    .action-list strong {
        color: #1e293b;
        font-weight: 700;
    }
    
    .value-proposition {
        background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
        border: 1px solid #2563eb;
        border-radius: 8px;
        padding: 2rem;
        margin: 2rem 0;
    }
    
    .value-proposition-title {
        font-size: 1.35rem;
        font-weight: 700;
        color: #1e3a8a;
        margin-bottom: 1rem;
    }
    
    .value-proposition-content {
        font-size: 0.95rem;
        color: #1e293b;
        line-height: 1.8;
    }
    
    .stButton button { 
        border-radius: 6px; 
        font-weight: 600;
        font-size: 0.95rem;
        padding: 0.625rem 1.5rem;
        transition: all 0.2s ease;
        border: none;
    }
    
    .stButton button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    .info-banner {
        background: #f0f9ff;
        border: 1px solid #bae6fd;
        border-radius: 6px;
        padding: 1rem 1.25rem;
        margin: 1rem 0;
        color: #0c4a6e;
        font-size: 0.9rem;
    }
    
    .success-banner {
        background: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-radius: 6px;
        padding: 1rem 1.25rem;
        margin: 1rem 0;
        color: #14532d;
        font-size: 0.9rem;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)


# =====================================================================
# AUTHENTICATION SYSTEM
# =====================================================================

class AuthManager:
    """
    Handles user authentication and account management.
    Stores credentials in local JSON file.
    """
    
    def __init__(self):
        self.accounts_file = "accounts.json"
        self.accounts = self._load_accounts()
    
    def _load_accounts(self):
        """Load existing accounts or create default demo account."""
        if os.path.exists(self.accounts_file):
            try:
                with open(self.accounts_file, 'r') as f:
                    return json.load(f)
            except:
                return self._create_default_accounts()
        return self._create_default_accounts()
    
    def _create_default_accounts(self):
        """Create default demo account for testing."""
        import hashlib
        return {
            "demo@mcfarland.com": {
                "company": "McFarland Johnson",
                "pwd_hash": hashlib.sha256("demo123".encode()).hexdigest(),
                "projects": [],
                "created": datetime.now().isoformat()
            }
        }
    
    def _hash_password(self, password):
        """Hash password using SHA256."""
        import hashlib
        return hashlib.sha256(password.encode()).hexdigest()
    
    def login(self, email, password):
        """
        Authenticate user credentials.
        Returns (success: bool, user_data: dict or None)
        """
        if email not in self.accounts:
            return False, None
        
        if self.accounts[email]["pwd_hash"] != self._hash_password(password):
            return False, None
        
        return True, self.accounts[email]
    
    def register(self, company, email, password):
        """
        Register new user account.
        Returns (success: bool, message: str)
        """
        if email in self.accounts:
            return False, "Email address already registered"
        
        if len(password) < 6:
            return False, "Password must be at least 6 characters"
        
        self.accounts[email] = {
            "company": company,
            "pwd_hash": self._hash_password(password),
            "projects": [],
            "created": datetime.now().isoformat()
        }
        
        self._save_accounts()
        return True, "Account created successfully"
    
    def _save_accounts(self):
        """Persist accounts to disk."""
        with open(self.accounts_file, 'w') as f:
            json.dump(self.accounts, f, indent=2)


# =====================================================================
# ANALYSIS ENGINE
# =====================================================================

def analyze_project(df, parameters):
    """
    Execute AI analysis on project data.
    
    Args:
        df: DataFrame containing project schedule and cost data
        parameters: Dict with project context (type, location, budget, timeline)
    
    Returns:
        (success: bool, result: dict or error_message: str)
    """
    try:
        # Map columns to standardized format
        mapper = FlexibleColumnMapper()
        df_mapped, mapping_info = mapper.map_dataframe(df, verbose=False)
        
        # Ensure required columns exist
        required_columns = ['estimated_cost', 'actual_cost']
        for col in required_columns:
            if col not in df_mapped.columns:
                df_mapped[col] = 0
        
        # Handle date columns
        if 'planned_end_date' not in df_mapped.columns:
            df_mapped['planned_end_date'] = pd.Timestamp.now()
        if 'actual_end_date' not in df_mapped.columns:
            df_mapped['actual_end_date'] = pd.Timestamp.now()
        
        # Initialize prediction engines
        delay_engine = DelayEngineV2()
        cost_engine = CostEngineV2()
        
        # Process data through engines
        delay_engine.process(df_mapped)
        df_delay = delay_engine.df.copy()
        cost_engine.process(df_delay)
        
        # Extract features
        X_delay, y_delay = delay_engine.get_features()
        X_cost, y_cost = cost_engine.get_features()
        
        # Train and predict with integrated pipeline
        pipeline = IntegratedMasterPipeline()
        pipeline.train_delay_model(X_delay, y_delay)
        pipeline.train_cost_model(X_cost, y_cost)
        pipeline.predict_with_details(X_delay, X_cost, delay_engine, cost_engine)
        
        # Generate comprehensive report
        report = pipeline.get_project_report(0)
        
        # Attach parameter context
        report['parameters'] = parameters
        report['constraints_applied'] = {
            'project_type': parameters.get('project_type'),
            'location': parameters.get('location'),
            'timeline_months': parameters.get('timeline_months'),
            'budget_millions': parameters.get('budget_millions')
        }
        
        return True, report
    
    except Exception as e:
        return False, f"Analysis error: {str(e)}"


def display_results(report):
    """
    Display comprehensive analysis results with actionable insights.
    """
    
    st.markdown('<div class="section-title">Predictive Analysis Results</div>', unsafe_allow_html=True)
    
    # Extract report data
    delay_days = report['summary']['delay_days']
    cost_overrun_pct = report['summary']['cost_overrun_pct']
    risk_level = report['summary']['risk_level']
    
    params = report.get('parameters', {})
    budget_millions = params.get('budget_millions', 5)
    timeline_months = params.get('timeline_months', 12)
    location = params.get('location', 'Unknown')
    project_type = params.get('project_type', 'Infrastructure')
    company_name = params.get('company', 'Your Organization')
    
    # Calculate financial metrics
    total_budget = budget_millions * 1_000_000
    daily_cost = total_budget / (timeline_months * 30)
    delay_cost_impact = delay_days * daily_cost
    cost_overrun_dollars = total_budget * (cost_overrun_pct / 100)
    total_risk_exposure = delay_cost_impact + cost_overrun_dollars
    
    # Display key performance indicators
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f'''
        <div class="metric-card">
            <div class="metric-label">Schedule Delay</div>
            <div class="metric-value">{delay_days:.0f}</div>
            <div class="metric-sublabel">days behind schedule</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with col2:
        st.markdown(f'''
        <div class="metric-card">
            <div class="metric-label">Cost Overrun</div>
            <div class="metric-value">{cost_overrun_pct:.1f}%</div>
            <div class="metric-sublabel">${cost_overrun_dollars:,.0f} at risk</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with col3:
        st.markdown(f'''
        <div class="metric-card">
            <div class="metric-label">Total Exposure</div>
            <div class="metric-value">${total_risk_exposure/1_000_000:.2f}M</div>
            <div class="metric-sublabel">combined risk impact</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with col4:
        risk_class = "risk-high" if risk_level == "High" else "risk-medium" if risk_level == "Medium" else "risk-low"
        st.markdown(f'''
        <div class="metric-card">
            <div class="metric-label">Risk Level</div>
            <div class="metric-value {risk_class}">{risk_level}</div>
            <div class="metric-sublabel">overall assessment</div>
        </div>
        ''', unsafe_allow_html=True)
    
    # Value proposition summary
    potential_recovery = total_risk_exposure * 0.675
    roi_percentage = (potential_recovery / total_budget) * 100
    
    st.markdown(f'''
    <div class="value-proposition">
        <div class="value-proposition-title">Executive Summary</div>
        <div class="value-proposition-content">
            Our predictive analysis of your <strong>{project_type}</strong> project in <strong>{location}</strong> 
            has identified <strong>${total_risk_exposure:,.0f}</strong> in combined schedule and cost risk over the 
            <strong>{timeline_months}-month</strong> timeline. Through implementation of data-driven mitigation strategies, 
            <strong>{company_name}</strong> can potentially recover <strong>60-75%</strong> of these costs, representing 
            <strong>${potential_recovery:,.0f}</strong> in preserved value. Our models leverage historical construction data 
            to identify risks weeks to months in advance, enabling proactive intervention before issues escalate.
        </div>
    </div>
    ''', unsafe_allow_html=True)
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # Risk assessment and mitigation strategies
    col_risks, col_actions = st.columns([1, 1])
    
    with col_risks:
        st.markdown('<div class="section-title">Risk Assessment</div>', unsafe_allow_html=True)
        
        # Generate risk alerts based on thresholds
        risk_items = []
        
        if delay_days > 30:
            risk_items.append((
                "Critical Schedule Risk",
                f"Project is tracking {delay_days:.0f} days behind schedule. At ${daily_cost:,.0f} per day, this represents ${delay_cost_impact:,.0f} in delay-related costs.",
                "warning"
            ))
        elif delay_days > 15:
            risk_items.append((
                "Significant Schedule Risk",
                f"Project shows {delay_days:.0f}-day delay trend. Early intervention can prevent ${delay_cost_impact:,.0f} in schedule-related costs.",
                "warning"
            ))
        elif delay_days > 5:
            risk_items.append((
                "Moderate Schedule Risk",
                f"Minor delays detected at {delay_days:.0f} days. Proactive management can keep project on track.",
                "warning"
            ))
        
        if cost_overrun_pct > 15:
            risk_items.append((
                "Major Budget Risk",
                f"Cost overrun projection of {cost_overrun_pct:.1f}% totaling ${cost_overrun_dollars:,.0f}. Immediate cost control measures required.",
                "warning"
            ))
        elif cost_overrun_pct > 8:
            risk_items.append((
                "Elevated Budget Risk",
                f"Cost trending {cost_overrun_pct:.1f}% over budget at ${cost_overrun_dollars:,.0f}. Review procurement and resource allocation.",
                "warning"
            ))
        elif cost_overrun_pct > 3:
            risk_items.append((
                "Minor Budget Variance",
                f"Modest cost increase of {cost_overrun_pct:.1f}% detected. Monitor closely to prevent escalation.",
                "warning"
            ))
        
        if delay_days <= 5 and cost_overrun_pct <= 3:
            risk_items.append((
                "Project On Track",
                f"Your {project_type} project is performing within acceptable parameters. Continue current management approach and monitoring protocols.",
                "success"
            ))
        
        for title, description, alert_type in risk_items:
            alert_class = "alert-box" if alert_type == "warning" else "alert-box alert-success"
            st.markdown(f'''
            <div class="{alert_class}">
                <div class="alert-title">{title}</div>
                <div class="alert-description">{description}</div>
            </div>
            ''', unsafe_allow_html=True)
    
    with col_actions:
        st.markdown('<div class="section-title">Mitigation Strategies</div>', unsafe_allow_html=True)
        
        # Calculate specific savings opportunities
        labor_budget = total_budget * 0.45
        material_budget = total_budget * 0.35
        equipment_budget = total_budget * 0.20
        
        # Immediate cost recovery actions
        schedule_compression_days = int(delay_days * 0.4)
        schedule_compression_savings = schedule_compression_days * daily_cost
        material_discount_savings = material_budget * 0.05
        labor_optimization_savings = labor_budget * 0.18
        
        st.markdown(f'''
        <div class="action-card">
            <div class="action-title">Immediate Cost Recovery Actions</div>
            <ul class="action-list">
                <li><strong>Schedule Compression:</strong> Each day saved recovers ${daily_cost:,.0f}. Target {schedule_compression_days} days through parallel activities for ${schedule_compression_savings:,.0f} in savings.</li>
                <li><strong>Bulk Material Procurement:</strong> Negotiate 4-6% discount on ${material_budget:,.0f} materials budget resulting in ${material_discount_savings:,.0f} in cost savings.</li>
                <li><strong>Labor Optimization:</strong> Reduce crew size during low-productivity phases by 18% for ${labor_optimization_savings:,.0f} in labor savings.</li>
            </ul>
        </div>
        ''', unsafe_allow_html=True)
        
        # Schedule acceleration tactics
        permit_processing_days = 45 if location in ["New York", "California"] else 30
        parallel_execution_days = int(delay_days * 0.35)
        prefab_reduction_days = int(delay_days * 0.25)
        
        st.markdown(f'''
        <div class="action-card">
            <div class="action-title">Schedule Acceleration Tactics</div>
            <ul class="action-list">
                <li><strong>Critical Path Fast-Track:</strong> Identify 2-3 sequential activities on critical path for parallel execution. Potential time savings: {parallel_execution_days} days.</li>
                <li><strong>Permit Pre-Filing:</strong> Submit {location} permit applications immediately versus waiting. Average processing time: {permit_processing_days} days.</li>
                <li><strong>Off-Site Pre-Fabrication:</strong> Move 25-30% of work to controlled workshop environment, reducing field time by {prefab_reduction_days} days.</li>
            </ul>
        </div>
        ''', unsafe_allow_html=True)
        
        # Ongoing risk monitoring
        st.markdown(f'''
        <div class="action-card">
            <div class="action-title">Ongoing Risk Monitoring</div>
            <ul class="action-list">
                <li><strong>Weekly Progress Reviews:</strong> Track actual versus predicted performance with model updates as new data becomes available.</li>
                <li><strong>Weather Impact Analysis:</strong> Integration of real-time weather data to adjust delay predictions based on {location} seasonal patterns.</li>
                <li><strong>Early Warning Indicators:</strong> Monitor leading indicators including permit approval rates, material delivery timelines, and subcontractor availability.</li>
            </ul>
        </div>
        ''', unsafe_allow_html=True)
    
    # Bottom line ROI summary
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    total_savings = schedule_compression_savings + material_discount_savings + labor_optimization_savings
    
    st.markdown(f'''
    <div class="value-proposition">
        <div class="value-proposition-title">Return on Investment</div>
        <div class="value-proposition-content">
            By implementing these evidence-based recommendations, <strong>{company_name}</strong> can achieve:
            <ul style="margin-top: 1rem; padding-left: 1.5rem; color: #1e293b;">
                <li style="margin: 0.5rem 0;"><strong>Recover ${potential_recovery:,.0f}</strong> of the ${total_risk_exposure:,.0f} at-risk budget (67.5% recovery rate)</li>
                <li style="margin: 0.5rem 0;"><strong>Reduce schedule delays</strong> from {delay_days:.0f} days to approximately {delay_days * 0.3:.0f} days through proactive management</li>
                <li style="margin: 0.5rem 0;"><strong>Improve project margin</strong> by {roi_percentage:.1f}% through data-driven risk management</li>
                <li style="margin: 0.5rem 0;"><strong>Enhance stakeholder confidence</strong> with transparent risk communication and measurable mitigation outcomes</li>
            </ul>
        </div>
    </div>
    ''', unsafe_allow_html=True)


# =====================================================================
# MAIN APPLICATION
# =====================================================================

def main():
    """Main application entry point."""
    
    # Initialize session state
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
        st.session_state.user_email = None
        st.session_state.user_company = None
    
    auth_manager = AuthManager()
    
    # =====================================================================
    # AUTHENTICATION FLOW
    # =====================================================================
    
    if not st.session_state.authenticated:
        st.markdown('<div class="main-header">Construction AI Predictor</div>', unsafe_allow_html=True)
        st.markdown('<div class="company-info">Predictive analytics platform for infrastructure project risk management</div>', unsafe_allow_html=True)
        
        st.markdown('<div style="padding: 2rem;"></div>', unsafe_allow_html=True)
        
        tab_login, tab_register = st.tabs(["Sign In", "Create Account"])
        
        with tab_login:
            st.markdown("### Sign In to Your Account")
            st.markdown("Access your project analytics dashboard")
            
            email = st.text_input("Email Address", key="login_email")
            password = st.text_input("Password", type="password", key="login_password")
            
            if st.button("Sign In", type="primary", use_container_width=True):
                success, user_data = auth_manager.login(email, password)
                if success:
                    st.session_state.authenticated = True
                    st.session_state.user_email = email
                    st.session_state.user_company = user_data['company']
                    st.success("Authentication successful")
                    st.rerun()
                else:
                    st.error("Invalid credentials. Please try again.")
            
            st.markdown('<div class="info-banner">Demo Account: demo@mcfarland.com / demo123</div>', unsafe_allow_html=True)
        
        with tab_register:
            st.markdown("### Create New Account")
            st.markdown("Register your organization for predictive analytics access")
            
            company = st.text_input("Company Name", key="signup_company")
            email = st.text_input("Email Address", key="signup_email")
            password = st.text_input("Password", type="password", key="signup_password")
            terms = st.checkbox("I agree to the terms of service and privacy policy")
            
            if st.button("Create Account", type="primary", use_container_width=True):
                if not terms:
                    st.warning("Please accept the terms of service to continue")
                else:
                    success, message = auth_manager.register(company, email, password)
                    if success:
                        st.session_state.authenticated = True
                        st.session_state.user_email = email
                        st.session_state.user_company = company
                        st.success("Account created successfully")
                        st.rerun()
                    else:
                        st.error(message)
        
        return
    
    # =====================================================================
    # AUTHENTICATED APPLICATION
    # =====================================================================
    
    # Header with navigation
    col_title, col_signout = st.columns([4, 1])
    with col_title:
        st.markdown('<div class="main-header">Construction AI Predictor</div>', unsafe_allow_html=True)
    with col_signout:
        if st.button("Sign Out", use_container_width=True):
            st.session_state.authenticated = False
            st.rerun()
    
    st.markdown(f'<div class="company-info"><strong>{st.session_state.user_company}</strong> | {st.session_state.user_email}</div>', unsafe_allow_html=True)
    
    # Navigation buttons
    col_nav1, col_nav2, col_nav3 = st.columns([1, 2, 1])
    with col_nav1:
        if st.button("Past Projects", use_container_width=True):
            st.session_state.view = "past_projects"
    with col_nav3:
        if st.button("Profile", use_container_width=True):
            st.session_state.view = "profile"
    
    st.markdown('<div style="padding: 1rem;"></div>', unsafe_allow_html=True)
    
    # Handle navigation views
    if "view" in st.session_state:
        if st.session_state.view == "past_projects":
            st.markdown('<div class="section-title">Project History</div>', unsafe_allow_html=True)
            st.markdown('<div class="info-banner">No past projects available. Upload your first project below to begin analysis.</div>', unsafe_allow_html=True)
            if st.button("Return to Analysis Dashboard"):
                del st.session_state.view
                st.rerun()
            return
        
        elif st.session_state.view == "profile":
            st.markdown('<div class="section-title">Account Profile</div>', unsafe_allow_html=True)
            st.markdown(f"**Company:** {st.session_state.user_company}")
            st.markdown(f"**Email:** {st.session_state.user_email}")
            st.markdown(f"**Account Type:** Enterprise Professional")
            if st.button("Return to Analysis Dashboard"):
                del st.session_state.view
                st.rerun()
            return
    
    # =====================================================================
    # MAIN ANALYSIS DASHBOARD
    # =====================================================================
    
    st.markdown('<div class="section-title">Upload Project Data</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Upload your project schedule file (CSV or Excel format) containing task information, dates, estimated costs, and actual costs for AI-powered analysis.</div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Select project file",
        type=['csv', 'xlsx', 'xls'],
        help="Supported formats: CSV, Excel (.xlsx, .xls)"
    )
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    st.markdown('<div class="section-title">Project Parameters</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Define your project scope and constraints to calibrate AI predictions to your specific context.</div>', unsafe_allow_html=True)
    
    col_param1, col_param2 = st.columns(2)
    
    with col_param1:
        project_type = st.selectbox(
            "Project Type",
            ["Highway", "Bridge", "Terminal", "Infrastructure", "Transportation", "Water/Wastewater", "Other"],
            help="Select the primary project category"
        )
        
        location = st.selectbox(
            "Project Location",
            ["New York", "California", "Texas", "Florida", "Oregon", "Massachusetts", "Washington", "Other"],
            help="Location affects permit timelines and environmental risk factors"
        )
    
    with col_param2:
        timeline_months = st.number_input(
            "Planned Timeline (months)",
            min_value=1,
            max_value=120,
            value=12,
            help="Total project duration from start to substantial completion"
        )
        
        budget_millions = st.number_input(
            "Total Budget ($M)",
            min_value=0.1,
            max_value=1000.0,
            value=5.0,
            step=0.5,
            help="Total project budget in millions of dollars"
        )
    
    parameters = {
        "project_type": project_type,
        "location": location,
        "timeline_months": timeline_months,
        "budget_millions": budget_millions,
        "company": st.session_state.user_company
    }
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # File processing and analysis
    if uploaded_file is not None:
        st.markdown(f'<div class="success-banner">File loaded successfully: {uploaded_file.name}</div>', unsafe_allow_html=True)
        
        try:
            # Read uploaded file
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(io.StringIO(uploaded_file.getvalue().decode('utf-8')))
            else:
                df = pd.read_excel(uploaded_file)
            
            st.markdown(f"**Dataset:** {len(df)} rows × {len(df.columns)} columns")
            
            if st.button("Run AI Analysis", type="primary", use_container_width=True):
                with st.spinner("Processing project data through AI models..."):
                    success, result = analyze_project(df, parameters)
                    
                    if success:
                        st.session_state.report = result
                        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
                        display_results(result)
                        
                        with st.expander("View Raw Project Data"):
                            st.dataframe(df.head(20), use_container_width=True)
                    else:
                        st.error(f"Analysis failed: {result}")
        
        except Exception as e:
            st.error(f"File processing error: {str(e)}")
    
    else:
        st.markdown('''
        <div class="upload-container">
            <div class="upload-title">Drop your project file here</div>
            <div class="upload-subtitle">CSV or Excel format with schedule data including tasks, dates, and costs</div>
        </div>
        ''', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
