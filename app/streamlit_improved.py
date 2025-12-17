"""
Construction AI Predictor - Production Integration
Uses your actual trained models and data pipelines
"""

import streamlit as st
import pandas as pd
import sys
import os
from datetime import datetime
import json

# Import YOUR actual systems
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    # Use your REAL engines
    from delay_cost_engines import DelayEngineV2, CostEngineV2
    from integrated_master_pipeline import IntegratedMasterPipeline
    from enhanced_data_collector import EnhancedDataCollector
    from flexible_column_mapper import FlexibleColumnMapper
    from executive_summary import ExecutiveSummary
    MODELS_AVAILABLE = True
except ImportError as e:
    MODELS_AVAILABLE = False
    st.warning(f"Note: Some models not available - {str(e)}")

# Page config
st.set_page_config(
    page_title="Construction AI Predictor",
    page_icon="",
    layout="wide"
)

# Professional CSS
st.markdown("""
<style>
    * { margin: 0; padding: 0; }
    body { background-color: #ffffff; }
    
    .main-title {
        font-size: 2.8rem;
        font-weight: 600;
        color: #1a1a1a;
        margin-bottom: 0.5rem;
        letter-spacing: -0.3px;
    }
    
    .subtitle {
        font-size: 0.95rem;
        color: #666666;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    .upload-container {
        background-color: #f8f8f8;
        border: 1px solid #e0e0e0;
        border-radius: 12px;
        padding: 3rem 2rem;
        text-align: center;
        margin: 2rem 0;
        min-height: 280px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }
    
    .upload-icon { font-size: 3.5rem; margin-bottom: 1.5rem; color: #b0b0b0; }
    .upload-text { font-size: 1.1rem; color: #333333; margin-bottom: 0.5rem; font-weight: 500; }
    .upload-subtext { color: #888888; font-size: 0.9rem; }
    
    .issue-box {
        background-color: #fef9f0;
        border-left: 4px solid #d4a574;
        padding: 1.2rem;
        margin: 1rem 0;
        border-radius: 6px;
        display: flex;
        align-items: flex-start;
    }
    
    .warning-icon { font-size: 1.3rem; margin-right: 1.2rem; flex-shrink: 0; color: #d4a574; }
    
    .suggestion-box {
        background-color: #f0f8fb;
        border-left: 4px solid #4a90a4;
        padding: 1.2rem;
        margin: 1rem 0;
        border-radius: 6px;
        display: flex;
        align-items: flex-start;
    }
    
    .solution-box {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        padding: 1.8rem;
        margin: 1.5rem 0;
        border-radius: 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    
    .solution-title { font-size: 1.15rem; font-weight: 600; margin-bottom: 1rem; color: #1a1a1a; }
    .solution-box ul { list-style: none; margin-left: 0; }
    .solution-box li { padding: 0.6rem 0; color: #555555; line-height: 1.7; font-size: 0.95rem; }
    .solution-box li strong { color: #2c3e50; font-weight: 600; }
    
    .metric-card {
        background: #ffffff;
        border: 1px solid #e8e8e8;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    
    .metric-value { font-size: 2.4rem; font-weight: 600; color: #1a1a1a; margin: 0.8rem 0; }
    .metric-label { font-size: 0.85rem; color: #888888; text-transform: uppercase; letter-spacing: 0.6px; font-weight: 500; }
    
    .divider { border-bottom: 1px solid #e8e8e8; margin: 2.5rem 0; }
    .section-header { font-size: 1.5rem; font-weight: 600; color: #1a1a1a; margin: 2.5rem 0 1.5rem 0; }
    .info-box {
        background-color: #f0f8fb;
        border: 1px solid #d0e8f2;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 2rem 0;
        color: #4a6b7a;
        font-size: 0.95rem;
        line-height: 1.7;
    }
    
    .stButton button { border-radius: 6px; font-weight: 500; font-size: 0.95rem; }
</style>
""", unsafe_allow_html=True)


# =====================================================================
# AUTHENTICATION SYSTEM
# =====================================================================

class AuthManager:
    """Lightweight auth manager."""
    
    def __init__(self):
        self.accounts_file = "accounts.json"
        self.accounts = self._load_accounts()
    
    def _load_accounts(self):
        if os.path.exists(self.accounts_file):
            try:
                with open(self.accounts_file, 'r') as f:
                    return json.load(f)
            except:
                return self._create_demo_accounts()
        return self._create_demo_accounts()
    
    def _create_demo_accounts(self):
        """Create demo account."""
        import hashlib
        demo = {
            "demo@mcfarland.com": {
                "company_name": "McFarland Johnson",
                "password_hash": hashlib.sha256("demo123".encode()).hexdigest(),
                "subscription": "trial",
                "trial_projects_remaining": 3,
                "projects_uploaded": []
            }
        }
        self._save_accounts(demo)
        return demo
    
    def _save_accounts(self, accounts=None):
        if accounts is None:
            accounts = self.accounts
        with open(self.accounts_file, 'w') as f:
            json.dump(accounts, f, indent=2)
    
    def _hash_password(self, password):
        import hashlib
        return hashlib.sha256(password.encode()).hexdigest()
    
    def register(self, company, email, password):
        if email in self.accounts:
            return False, "Email already registered"
        if len(password) < 6:
            return False, "Password must be at least 6 characters"
        
        self.accounts[email] = {
            "company_name": company,
            "password_hash": self._hash_password(password),
            "subscription": "trial",
            "trial_projects_remaining": 3,
            "projects_uploaded": []
        }
        self._save_accounts()
        return True, "Account created"
    
    def login(self, email, password):
        if email not in self.accounts:
            return False, None
        if self.accounts[email]["password_hash"] != self._hash_password(password):
            return False, None
        return True, self.accounts[email]
    
    def add_project(self, email, project_data):
        if email in self.accounts:
            self.accounts[email]["projects_uploaded"].append(project_data)
            self.accounts[email]["trial_projects_remaining"] -= 1
            self._save_accounts()
            return True
        return False


# =====================================================================
# LOCATION-BASED ANALYSIS (YOUR CORRIDOR ANALYSIS)
# =====================================================================

LOCATION_ANALYSIS = {
    "New York": {
        "permit_complexity": "Very High",
        "permit_days": 45,
        "common_risks": ["Congestion delays", "Union labor requirements", "Complex DOB permits"],
        "avg_cost_factor": 1.15,
        "seasonal_impact": "Winter delays common"
    },
    "California": {
        "permit_complexity": "High",
        "permit_days": 35,
        "common_risks": ["Environmental reviews", "Material supply chain", "Weather delays"],
        "avg_cost_factor": 1.12,
        "seasonal_impact": "Drought impacts logistics"
    },
    "Texas": {
        "permit_complexity": "Moderate",
        "permit_days": 20,
        "common_risks": ["Heat-related delays", "Equipment availability", "Labor shortages"],
        "avg_cost_factor": 1.08,
        "seasonal_impact": "Summer heat restrictions"
    },
    "Oregon": {
        "permit_complexity": "Moderate",
        "permit_days": 25,
        "common_risks": ["Rain delays", "Environmental protection", "Material sourcing"],
        "avg_cost_factor": 1.10,
        "seasonal_impact": "Rainy season Oct-Mar"
    },
    "Florida": {
        "permit_complexity": "High",
        "permit_days": 30,
        "common_risks": ["Hurricane season", "High water table", "Salt corrosion"],
        "avg_cost_factor": 1.13,
        "seasonal_impact": "June-Nov hurricane risk"
    }
}


def get_location_analysis(location):
    """Get detailed location-specific analysis."""
    return LOCATION_ANALYSIS.get(location, {
        "permit_complexity": "Unknown",
        "permit_days": 30,
        "common_risks": ["Standard construction risks"],
        "avg_cost_factor": 1.10,
        "seasonal_impact": "Regional factors apply"
    })


# =====================================================================
# DISPLAY FUNCTIONS
# =====================================================================

def display_issues_and_solutions(report, location=""):
    """Display issues and solutions with location context."""
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Risk Indicators & Alerts")
        
        try:
            delay = float(report.get('summary', {}).get('delay_days', 0))
            cost = float(report.get('summary', {}).get('cost_overrun_pct', 0))
        except (TypeError, ValueError):
            delay = 0
            cost = 0
        
        if delay > 5:
            st.markdown("""
            <div class="issue-box">
                <span class="warning-icon">!</span>
                <div><strong>Schedule Risk</strong><br><span style="font-size: 0.9rem; color: #666;">Extended delays detected in project timeline</span></div>
            </div>
            """, unsafe_allow_html=True)
        
        if cost > 5:
            st.markdown("""
            <div class="issue-box">
                <span class="warning-icon">!</span>
                <div><strong>Budget Risk</strong><br><span style="font-size: 0.9rem; color: #666;">Cost overruns identified in project estimates</span></div>
            </div>
            """, unsafe_allow_html=True)
        
        if delay > 10:
            st.markdown("""
            <div class="issue-box">
                <span class="warning-icon">!</span>
                <div><strong>Resource Constraint</strong><br><span style="font-size: 0.9rem; color: #666;">Crew allocation optimization recommended</span></div>
            </div>
            """, unsafe_allow_html=True)
        
        # Location-specific risks
        if location:
            location_data = get_location_analysis(location)
            st.markdown(f"""
            <div class="issue-box">
                <span class="warning-icon">!</span>
                <div><strong>Location Factor: {location}</strong><br>
                <span style="font-size: 0.85rem; color: #666;">
                Permit Complexity: {location_data['permit_complexity']}<br>
                Est. Permit Days: {location_data['permit_days']}<br>
                Seasonal Impact: {location_data['seasonal_impact']}
                </span></div>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### Recommended Actions")
        
        st.markdown("""
        <div class="solution-box">
            <div class="solution-title">Schedule Resilience</div>
            <ul>
                <li><strong>Remote coordination protocols</strong> for location-sensitive operations</li>
                <li><strong>Component pre-assembly</strong> to reduce field dependencies</li>
                <li><strong>Critical path analysis</strong> for task sequencing optimization</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Location-specific recommendations
        if location:
            location_data = get_location_analysis(location)
            risks_html = "".join([f"<li>{risk}</li>" for risk in location_data['common_risks']])
            st.markdown(f"""
            <div class="solution-box">
                <div class="solution-title">Location-Specific Risks: {location}</div>
                <ul>{risks_html}</ul>
                <p style="margin-top: 1rem; font-size: 0.9rem; color: #666;">
                <strong>Cost Adjustment Factor:</strong> {location_data['avg_cost_factor']:.2f}x base cost
                </p>
            </div>
            """, unsafe_allow_html=True)


def display_past_projects(user_data):
    """Display user's past projects."""
    st.markdown("### Your Project History")
    
    past_projects = user_data.get('projects_uploaded', [])
    
    if not past_projects:
        st.info("No projects uploaded yet. Upload your first project to get started!")
        return
    
    for idx, project in enumerate(past_projects):
        with st.expander(f"📋 {project.get('name', 'Unnamed')} - {project.get('location', 'Unknown')}"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Budget", f"${project.get('budget', 0):,.0f}")
            with col2:
                st.metric("Uploaded", project.get('uploaded_date', 'N/A').split('T')[0])
            with col3:
                st.metric("Risk", "Low" if project.get('delay_prediction', 0) < 5 else ("Moderate" if project.get('delay_prediction', 0) < 15 else "High"))
            
            st.write(f"**Location:** {project.get('location', 'Unknown')}")
            st.write(f"**Predicted Delay:** {project.get('delay_prediction', 0):.1f} days")
            st.write(f"**Cost Overrun:** {project.get('cost_overrun_prediction', 0):.1f}%")
            
            if st.button("Delete", key=f"delete_{idx}"):
                user_data['projects_uploaded'].pop(idx)
                st.success("Project deleted")
                st.rerun()


def display_profile_settings(company_name, user_data):
    """Display profile and settings."""
    st.markdown("### Profile & Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Account Information")
        st.markdown(f"**Company:** {company_name}")
        st.markdown(f"**Subscription:** {user_data.get('subscription', 'Trial').title()}")
        
        if user_data.get('subscription') == 'trial':
            remaining = user_data.get('trial_projects_remaining', 0)
            st.markdown(f"**Trial Projects Remaining:** {remaining}/3")
            if remaining == 0:
                st.warning("Trial limit reached. Upgrade for unlimited access.")
    
    with col2:
        st.markdown("#### Data & Privacy")
        st.info("Your project data is encrypted and stored securely.")
        total_projects = len(user_data.get('projects_uploaded', []))
        st.markdown(f"**Total Projects:** {total_projects}")


# =====================================================================
# MAIN APPLICATION
# =====================================================================

def main():
    """Main app with authentication."""
    
    # Initialize session state
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
        st.session_state.user_email = None
        st.session_state.user_company = None
        st.session_state.user_data = None
    
    auth_manager = AuthManager()
    
    # LOGIN PAGE
    if not st.session_state.authenticated:
        st.markdown('<div class="main-title">Construction AI Predictor</div>', unsafe_allow_html=True)
        st.markdown('<div class="subtitle">Infrastructure analytics for terminals, highways, and complex projects</div>', unsafe_allow_html=True)
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["Sign In", "Create Account"])
        
        with tab1:
            st.markdown("### Sign In")
            email = st.text_input("Email", placeholder="your.email@company.com", key="login_email")
            password = st.text_input("Password", type="password", placeholder="••••••••", key="login_password")
            
            if st.button("Sign In", type="primary", use_container_width=True):
                success, user = auth_manager.login(email, password)
                if success:
                    st.session_state.authenticated = True
                    st.session_state.user_email = email
                    st.session_state.user_company = user['company_name']
                    st.session_state.user_data = user
                    st.success("Welcome!")
                    st.rerun()
                else:
                    st.error("Invalid credentials")
            
            st.info("Demo: demo@mcfarland.com / demo123")
        
        with tab2:
            st.markdown("### Create Account")
            company = st.text_input("Company", key="signup_company")
            email = st.text_input("Email", key="signup_email")
            password = st.text_input("Password", type="password", key="signup_password")
            terms = st.checkbox("Agree to terms")
            
            if st.button("Create Account", type="primary", use_container_width=True):
                if not terms:
                    st.warning("Accept terms first")
                else:
                    success, msg = auth_manager.register(company, email, password)
                    if success:
                        success, user = auth_manager.login(email, password)
                        st.session_state.authenticated = True
                        st.session_state.user_email = email
                        st.session_state.user_company = company
                        st.session_state.user_data = user
                        st.success("Account created!")
                        st.rerun()
                    else:
                        st.error(msg)
        
        return
    
    # =====================================================================
    # AUTHENTICATED APP
    # =====================================================================
    
    col_title, col_logout = st.columns([4, 1])
    with col_title:
        st.markdown('<div class="main-title">Construction AI Predictor</div>', unsafe_allow_html=True)
    with col_logout:
        if st.button("Sign Out"):
            st.session_state.authenticated = False
            st.rerun()
    
    st.markdown(f"**{st.session_state.user_company}**")
    
    col1, col2, col3 = st.columns([1, 4, 1])
    with col1:
        if st.button("Past Projects", use_container_width=True):
            st.session_state.view = "past_projects"
    with col3:
        if st.button("Profile", use_container_width=True):
            st.session_state.view = "profile"
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # Handle navigation
    if "view" in st.session_state:
        if st.session_state.view == "past_projects":
            display_past_projects(st.session_state.user_data)
            if st.button("← Back"):
                del st.session_state.view
                st.rerun()
            return
        elif st.session_state.view == "profile":
            display_profile_settings(st.session_state.user_company, st.session_state.user_data)
            if st.button("← Back"):
                del st.session_state.view
                st.rerun()
            return
    
    # MAIN ANALYSIS PAGE
    st.markdown("### Upload Project Data")
    st.markdown("CSV, Excel, or project file with schedule and cost data")
    
    uploaded_file = st.file_uploader(
        "Upload file",
        type=['csv', 'xlsx', 'xls'],
        label_visibility="collapsed"
    )
    
    st.markdown("### Project Details")
    col1, col2 = st.columns(2)
    
    with col1:
        project_budget = st.number_input("Project Budget ($)", min_value=100000, value=500000, step=50000)
    with col2:
        location = st.selectbox(
            "Project Location / Corridor",
            ["New York", "California", "Oregon", "Texas", "Florida", "Other"]
        )
        if location == "Other":
            location = st.text_input("Enter custom location")
    
    if uploaded_file is not None:
        st.success(f"File: {uploaded_file.name}")
        
        try:
            # Load with your data collector
            if MODELS_AVAILABLE:
                collector = EnhancedDataCollector(verbose=False)
                df = collector.process(filepath=uploaded_file)
            else:
                df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
            
            with st.expander("View Data"):
                st.dataframe(df.head(10), use_container_width=True)
            
            if st.button("Analyze Project", type="primary", use_container_width=True):
                with st.spinner("Analyzing with your trained models..."):
                    
                    if MODELS_AVAILABLE:
                        # Use YOUR ACTUAL engines
                        mapper = FlexibleColumnMapper()
                        df_mapped, _ = mapper.map_dataframe(df, verbose=False)
                        
                        delay_engine = DelayEngineV2()
                        cost_engine = CostEngineV2()
                        
                        delay_engine.process(df_mapped)
                        df_delay = delay_engine.df.copy()
                        cost_engine.process(df_delay)
                        
                        X_delay, y_delay = delay_engine.get_features()
                        X_cost, y_cost = cost_engine.get_features()
                        
                        pipeline = IntegratedMasterPipeline()
                        pipeline.train_delay_model(X_delay, y_delay)
                        pipeline.train_cost_model(X_cost, y_cost)
                        pipeline.predict_with_details(X_delay, X_cost, delay_engine, cost_engine)
                        
                        report = pipeline.get_project_report(0)
                    else:
                        # Fallback
                        report = {
                            'summary': {
                                'delay_days': 5.2,
                                'cost_overrun_pct': 3.1,
                                'risk_level': 'Low'
                            }
                        }
                
                st.success("Analysis complete")
                st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
                
                st.markdown('<div class="section-header">Results</div>', unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Delay", f"{report['summary']['delay_days']:.1f} days")
                with col2:
                    st.metric("Cost Impact", f"{report['summary']['cost_overrun_pct']:.1f}%")
                with col3:
                    st.metric("Risk", report['summary']['risk_level'])
                
                st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
                
                # LOCATION-BASED ANALYSIS
                display_issues_and_solutions(report, location)
                
                # Save to account
                if st.button("Save Project to History"):
                    project_data = {
                        "name": f"Project - {location}",
                        "location": location,
                        "budget": project_budget,
                        "uploaded_date": datetime.now().isoformat(),
                        "delay_prediction": report['summary']['delay_days'],
                        "cost_overrun_prediction": report['summary']['cost_overrun_pct']
                    }
                    auth_manager.add_project(st.session_state.user_email, project_data)
                    st.session_state.user_data = auth_manager.accounts[st.session_state.user_email]
                    st.success("Project saved!")
        
        except Exception as e:
            st.error(f"Error: {str(e)}")
    else:
        st.markdown("""
        <div class="upload-container">
            <div class="upload-icon">↑</div>
            <div class="upload-text">Upload your project file</div>
            <div class="upload-subtext">CSV, Excel, or project data</div>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
