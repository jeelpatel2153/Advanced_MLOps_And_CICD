import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

# Load datasets
reference_data = pd.read_csv("data/train.csv")
current_data = pd.read_csv("data/test.csv")

# Create report
report = Report(metrics=[DataDriftPreset()])

# Run report
report.run(
    reference_data=reference_data,
    current_data=current_data
)

# Save report
report.save_html("drift_report.html")

print("Drift report generated successfully!")