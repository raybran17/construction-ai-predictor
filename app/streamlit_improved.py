"""
Construction AI Predictor - Production Ready
Uses YOUR trained models + ExecutiveSummary for real recommendations
"""

import streamlit as st
import pandas as pd
import io
import os
from datetime import datetime
import json

# Import your engines
from delay_cost_engines import DelayEngineV2, CostEngineV2
from integrated_master_pipeline import IntegratedMasterPipeline
from flexible_column_mapper import FlexibleColumnMapper
from executive_summary import ExecutiveSummary

# Page config
st.set_page_config(page_title="Construction AI Predictor", layout="wide")

# Professional CSS
st.markdown("""
<style>
    * { margin: 0; padding: 0; }
    body { background-color: #ffffff; }
    
    .main-title { font-size: 2.8rem; font-weight: 600; color: #1a1a1a; margin-bottom: 0.5rem; }
    .subtitle { font-size: 0.95rem; color: #666666; margin-bottom: 2rem; }
    
    .section-header { font-size: 1.5rem; font-weight: 600; color: #1a1a1a; margin: 2rem 0 1rem 0; }
    .divider { border-bottom: 1px solid #e8e8e8; margin: 2rem 0; }
    
    .metric-card { background: #ffffff; border: 1px solid #e8e8e8; border-radius: 10px; padding: 1.5rem; text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }
    .metric-value { font-size: 2rem; font-weight: 600; color: #1a1a1a; margin: 0.5rem 0; }
    .metric-label { font-size: 0.85rem; color: #888888; text-transform: uppercase; letter-spacing: 0.6px; }
    
    .issue-box { background-color: #fef9f0; border-left: 4px solid #d4a574; padding: 1rem; margin: 0.8rem 0; border-radius: 6px; }
    .warning-icon { color: #d4a574; font-weight: 600; margin-right: 0.5rem; }
    
    .action-box { background-color: #ffffff; border: 1px solid #e0e0e0; padding: 1rem; margin: 0.8rem 0; border-radius: 6px; }
    .action-title { font-weight: 600; color: #1a1a1a; }
    .action-content { color: #555555; font-size: 0.95rem; margin-top: 0.5rem; line-height: 1.6; }
    
    .stButton button { border-radius: 6px; font-weight: 500; }
</style>
""", unsafe_allow_html=True)


# =====================================================================
# AUTH SYSTEM
# =====================================================================

class AuthManager:
    def __init__(self):
        self.accounts_file = "accounts.json"
        self.accounts = self._load_accounts()
    
    def _load_accounts(self):
        if os.path.exists(self.accounts_file):
            try:
                return json.load(open(self.accounts_file))
            except:
                return self._defaults()
        return self._defaults()
    
    def _defaults(self):
        import hashlib
        return {
            "demo@mcfarland.com": {
                "company": "McFarland Johnson",
                "pwd_hash": hashlib.sha256("demo123".encode()).hexdigest(),
                "projects": []
            }
        }
    
    def _hash_password(self, pwd):
        import hashlib
        return hashlib.sha256(pwd.encode()).hexdigest()
    
    def login(self, email, pwd):
        if email not in self.accounts:
            return False, None
        if self.accounts[email]["pwd_hash"] != self._hash_password(pwd):
            return False, None
        return True, self.accounts[email]
    
    def register(self, company, email, pwd):
        if email in self.accounts:
            return False, "Email exists"
        if len(pwd) < 6:
            return False, "Password too short"
        
        self.accounts[email] = {
            "company": company,
            "pwd_hash": self._hash_password(pwd),
            "projects": []
        }
        self._save()
        return True, "Account created"
    
    def _save(self):
        with open(self.accounts_file, 'w') as f:
            json.dump(self.accounts, f)


# =====================================================================
# ANALYSIS & DISPLAY
# =====================================================================

def analyze_project(df, parameters):
    """Analyze project and return structured results."""
    try:
        # Map columns
        mapper = FlexibleColumnMapper()
        df_mapped, _ = mapper.map_dataframe(df, verbose=False)
        
        # Process through engines
        delay_engine = DelayEngineV2()
        cost_engine = CostEngineV2()
        
        delay_engine.process(df_mapped)
        df_delay = delay_engine.df.copy()
        cost_engine.process(df_delay)
        
        X_delay, y_delay = delay_engine.get_features()
        X_cost, y_cost = cost_engine.get_features()
        
        # Train models
        pipeline = IntegratedMasterPipeline()
        pipeline.train_delay_model(X_delay, y_delay)
        pipeline.train_cost_model(X_cost, y_cost)
        pipeline.predict_with_details(X_delay, X_cost, delay_engine, cost_engine)
        
        report = pipeline.get_project_report(0)
        
        return True, report, pipeline, delay_engine, cost_engine
    
    except Exception as e:
        return False, str(e), None, None, None


def display_results(report, delay_engine, cost_engine):
    """Display analysis results with ExecutiveSummary recommendations based on REAL project data."""
    
    st.markdown('<div class="section-header">Analysis Results</div>', unsafe_allow_html=True)
    
    # Key metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Predicted Delay</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{report["summary"]["delay_days"]:.1f}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">days</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Cost Overrun</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{report["summary"]["cost_overrun_pct"]:.1f}%</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Risk Level</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{report["summary"]["risk_level"]}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # Extract REAL project data from analyzed dataframes
    try:
        # Get first project's actual data from the analyzed engines
        if delay_engine.df is not None and len(delay_engine.df) > 0:
            project_row = delay_engine.df.iloc[0]
            
            # Build project_data from REAL CSV analysis
            project_data = {
                'project_type': project_row.get('project_type', 'Unknown'),
                'location': project_row.get('location', 'Unknown'),
                'estimated_cost': float(project_row.get('estimated_cost', 0)),
                'actual_cost': float(project_row.get('actual_cost', 0)),
                'crew_size_avg': int(project_row.get('crew_size_avg', 20)),
                'equipment_downtime_hours': float(project_row.get('equipment_downtime_hours', 0)),
                'total_labor_hours': float(project_row.get('total_labor_hours', 0))
            }
        else:
            project_data = {'project_type': 'Unknown', 'location': 'Unknown', 'estimated_cost': 0, 'crew_size_avg': 20, 'equipment_downtime_hours': 0}
        
        # Get phase and cost analysis
        phase_analysis = delay_engine.get_all_phase_analysis() if hasattr(delay_engine, 'get_all_phase_analysis') else {}
        cost_breakdown = cost_engine.get_all_cost_breakdowns() if hasattr(cost_engine, 'get_all_cost_breakdowns') else {}
        
        # Get first project's analysis
        phase_data = phase_analysis.get(0, {})
        cost_data = cost_breakdown.get(0, {})
        
        # Create ExecutiveSummary with REAL data
        exec_summary = ExecutiveSummary(
            project_data=project_data,
            delay_pred=report['summary']['delay_days'],
            cost_pred=report['summary']['cost_overrun_pct'],
            phase_analysis=phase_data,
            cost_breakdown=cost_data
        )
        
        summary_dict = exec_summary.export_to_dict()
        risks = summary_dict.get('top_risks', [])
        actions = summary_dict.get('recommended_actions', [])
    except Exception as e:
        st.warning(f"Could not generate ExecutiveSummary: {str(e)}")
        risks = []
        actions = []
    
    # Issues and Recommendations
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Identified Issues")
        
        delay = report['summary']['delay_days']
        cost = report['summary']['cost_overrun_pct']
        
        if delay > 5:
            st.markdown("""<div class="issue-box">
                <span class="warning-icon">⚠</span> Schedule Risk
                <div class="action-content">Delays detected in project timeline</div>
            </div>""", unsafe_allow_html=True)
        
        if cost > 5:
            st.markdown("""<div class="issue-box">
                <span class="warning-icon">⚠</span> Budget Risk
                <div class="action-content">Cost overruns identified</div>
            </div>""", unsafe_allow_html=True)
        
        if delay <= 5 and cost <= 5:
            st.markdown("""<div class="issue-box">
                <span class="warning-icon">✓</span> On Track
                <div class="action-content">Project within acceptable parameters</div>
            </div>""", unsafe_allow_html=True)
        
        # Top Risks
        if risks:
            st.markdown("### Top Risks")
            for risk in risks[:3]:
                st.markdown(f"""<div class="action-box">
                    <div class="action-title">{risk['title']}</div>
                    <div class="action-content">
                        Probability: {risk['probability']}%<br>
                        Impact: {risk['impact']}<br>
                        Financial Exposure: ${risk['cost']:,.0f}<br>
                        <strong>Action:</strong> {risk['action']}
                    </div>
                </div>""", unsafe_allow_html=True)
    
    with col2:
        st.markdown("### AI-Recommended Actions")
        
        if actions:
            for action in actions[:3]:
                priority_color = "#d4a574" if action['priority'] == 'CRITICAL' else "#4a90a4"
                st.markdown(f"""<div class="action-box" style="border-left: 4px solid {priority_color};">
                    <div class="action-title">[{action['priority']}] {action['title']}</div>
                    <div class="action-content">
                        Investment: ${action['cost']:,.0f}<br>
                        Potential Savings: ${action['saves']:,.0f}<br>
                        Time Saved: {action['time_saved']} days<br>
                        ROI: {action['roi']}
                    </div>
                </div>""", unsafe_allow_html=True)
        else:
            st.info("No specific actions needed at this time.")


# =====================================================================
# MAIN APP
# =====================================================================

def main():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
        st.session_state.user_email = None
        st.session_state.user_company = None
    
    auth_manager = AuthManager()
    
    # LOGIN
    if not st.session_state.authenticated:
        st.markdown('<div class="main-title">Construction AI Predictor</div>', unsafe_allow_html=True)
        st.markdown('<div class="subtitle">Infrastructure project analytics</div>', unsafe_allow_html=True)
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["Sign In", "Create Account"])
        
        with tab1:
            st.markdown("### Sign In")
            email = st.text_input("Email", key="login_email")
            password = st.text_input("Password", type="password", key="login_password")
            
            if st.button("Sign In", type="primary", use_container_width=True):
                success, user = auth_manager.login(email, password)
                if success:
                    st.session_state.authenticated = True
                    st.session_state.user_email = email
                    st.session_state.user_company = user['company']
                    st.success("Welcome!")
                    st.rerun()
                else:
                    st.error("Invalid credentials")
            
            st.info("Demo: demo@mcfarland.com / demo123")
        
        with tab2:
            st.markdown("### Create Account")
            company = st.text_input("Company Name", key="signup_company")
            email = st.text_input("Email", key="signup_email")
            password = st.text_input("Password", type="password", key="signup_password")
            terms = st.checkbox("Agree to terms of service")
            
            if st.button("Create Account", type="primary", use_container_width=True):
                if not terms:
                    st.warning("Accept terms first")
                else:
                    success, msg = auth_manager.register(company, email, password)
                    if success:
                        st.session_state.authenticated = True
                        st.session_state.user_email = email
                        st.session_state.user_company = company
                        st.success("Account created!")
                        st.rerun()
                    else:
                        st.error(msg)
        
        return
    
    # =====================================================================
    # AUTHENTICATED APP
    # =====================================================================
    
    st.markdown('<div class="main-title">Construction AI Predictor</div>', unsafe_allow_html=True)
    st.markdown(f"**{st.session_state.user_company}**", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("Past Projects", use_container_width=True):
            st.session_state.view = "past_projects"
    with col2:
        if st.button("Sign Out", use_container_width=True):
            st.session_state.authenticated = False
            st.rerun()
    with col3:
        if st.button("Profile", use_container_width=True):
            st.session_state.view = "profile"
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # Navigation
    if "view" in st.session_state:
        if st.session_state.view == "past_projects":
            st.markdown("### Past Projects")
            st.info("No past projects yet. Upload your first project to get started.")
            if st.button("← Back to Analysis"):
                del st.session_state.view
                st.rerun()
            return
        elif st.session_state.view == "profile":
            st.markdown("### Profile Settings")
            st.markdown(f"**Company:** {st.session_state.user_company}")
            st.markdown(f"**Email:** {st.session_state.user_email}")
            if st.button("← Back to Analysis"):
                del st.session_state.view
                st.rerun()
            return
    
    # =====================================================================
    # MAIN ANALYSIS PAGE
    # =====================================================================
    
    st.markdown("### Upload Project Data")
    uploaded_file = st.file_uploader("Select CSV or Excel file", type=['csv', 'xlsx', 'xls'])
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    if uploaded_file is not None:
        st.success(f"File loaded: {uploaded_file.name}")
        
        try:
            # Read file
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(io.StringIO(uploaded_file.getvalue().decode('utf-8')))
            else:
                df = pd.read_excel(uploaded_file)
            
            st.markdown(f"Rows: {len(df)} | Columns: {len(df.columns)}")
            
            if st.button("Analyze Project", type="primary", use_container_width=True):
                with st.spinner("Analyzing with your trained models..."):
                    success, result, pipeline, delay_engine, cost_engine = analyze_project(df, {})
                    
                    if success:
                        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
                        display_results(result, delay_engine, cost_engine)
                        
                        with st.expander("View raw data (first 10 rows)"):
                            st.dataframe(df.head(10))
                    else:
                        st.error(f"Analysis failed: {result}")
        
        except Exception as e:
            st.error(f"File error: {str(e)}")


if __name__ == "__main__":
    main()
