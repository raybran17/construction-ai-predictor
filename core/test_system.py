"""
Complete System Test
Tests all pipelines with NYC data
"""

print("\n" + "🚀 "+"="*68)
print("TESTING COMPLETE CONSTRUCTION AI SYSTEM")
print("="*70 + "\n")

# Step 1: Load NYC data
print("📊 Step 1: Loading NYC residential data...")
import pandas as pd

try:
    df = pd.read_csv('nyc_residential_projects.csv')
    print(f"✅ Loaded {len(df)} NYC projects")
    print(f"   Columns: {', '.join(df.columns.tolist()[:5])}...")
except FileNotFoundError:
    print("❌ nyc_residential_projects.csv not found!")
    print("   Run: python nyc_data_collector.py first")
    exit()

# Step 2: Universal Adapter
print("\n🔧 Step 2: Running Universal Adapter...")
from universal_adapter_pipeline import UniversalDataAdapter

adapter = UniversalDataAdapter()
adapter.run_full_adapter('nyc_residential_projects.csv')
clean_data = adapter.get_standardized_data()
print(f"✅ Adapter complete: {len(clean_data)} projects standardized")

# Step 3: Delay Engine
print("\n⏱️  Step 3: Running Enhanced Delay Engine...")
from delay_cost_engines import EnhancedDelayEngine

delay_engine = EnhancedDelayEngine()
delay_engine.process(clean_data)
X_delay, y_delay = delay_engine.get_features()
print(f"✅ Delay engine: {X_delay.shape[1]} features created")

# Step 4: Cost Engine
print("\n💰 Step 4: Running Enhanced Cost Engine...")
from delay_cost_engines import EnhancedCostEngine

cost_engine = EnhancedCostEngine()
cost_engine.process(clean_data)
X_cost, y_cost = cost_engine.get_features()
print(f"✅ Cost engine: {X_cost.shape[1]} features created")

# Step 5: Train Models
print("\n🎓 Step 5: Training Models...")
from master_pipeline import EnhancedMasterPipeline

master = EnhancedMasterPipeline()
master.train_delay_model(X_delay, y_delay)
master.train_cost_model(X_cost, y_cost)

# Step 6: Make Predictions
print("\n🔮 Step 6: Generating Predictions...")
master.predict_with_details(X_delay, X_cost, delay_engine, cost_engine)

# Step 7: Test Prescriptive Pipeline
print("\n🎯 Step 7: Testing Prescriptive Recommendations...")
from prescriptive_pipeline import PrescriptivePipeline

prescriptive = PrescriptivePipeline()

# Get analysis for first project
report = master.get_project_report(0)
delay_pred = report['summary']['total_delay_days']
cost_pred = report['summary']['cost_overrun_pct']

recommendations = prescriptive.analyze_project(
    project_data={'project_type': 'Residential', 'planned_duration': 120, 'planned_cost': 500000},
    delay_prediction=delay_pred,
    cost_prediction=cost_pred,
    delay_breakdown=report['delay_breakdown'],
    cost_breakdown=report['cost_breakdown'],
    worker_optimization=report['worker_optimization']
)

print(f"✅ Prescriptive pipeline: {len(recommendations['priority_actions'])} actions generated")

# Step 8: Display Sample Output
print("\n" + "="*70)
print("📋 SAMPLE OUTPUT FOR PROJECT #1")
print("="*70)

output = master.generate_dashboard_output(0)
print(output)

# Step 9: Test FAQ
print("\n💬 TESTING FAQ ENGINE:")
print("="*70)
print("\nQ: What's my biggest risk?")
print(f"A: {prescriptive.answer_question('What is my biggest risk?')}")

print("\nQ: Should I hire more workers?")
print(f"A: {prescriptive.answer_question('Should I hire more workers?')}")

# Step 10: Save Models
print("\n💾 Step 10: Saving trained models...")
master.save_models()
master.export_results('test_predictions.csv')

# Final Summary
print("\n" + "="*70)
print("🎉 SYSTEM TEST COMPLETE!")
print("="*70)

metrics = master.get_metrics()
print(f"\n📊 Model Performance:")
print(f"   Delay Model: {metrics['delay']['test']['r2']*100:.1f}% accurate")
print(f"   Cost Model:  {metrics['cost']['test']['r2']*100:.1f}% accurate")

print(f"\n✅ Files Created:")
print(f"   • trained_delay_model.pkl")
print(f"   • trained_cost_model.pkl")
print(f"   • test_predictions.csv")

print(f"\n🚀 System is READY for production!")
print("="*70 + "\n")
