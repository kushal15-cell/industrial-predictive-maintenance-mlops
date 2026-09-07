"""Review persisted evidence without loading the model or running monitoring."""
import json
from pathlib import Path
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
st.set_page_config(page_title="Agent decision review", layout="wide")
st.title("Agent decision review")
st.caption("Approval permits training. Only evaluated candidates are eligible for a separate deployment review.")
paths = sorted((ROOT / "monitoring/reports/decisions").glob("*/record.json"),
               key=lambda p: p.stat().st_mtime, reverse=True)
if not paths:
    example = ROOT / 'docs/examples/borderline-decision.json'
    if example.exists():
        paths = [example]
        st.warning('Injected demonstration: borderline recommendation and blocked promotion. This is not a production incident.')
    else:
        st.info("No decision history yet. Run python -m src.agent.agent after generating fresh monitoring evidence.")
        st.stop()
selected = st.selectbox("Decision", paths, format_func=lambda p: p.parent.name)
try:
    record = json.loads(selected.read_text(encoding="utf-8"))
    st.subheader({"RETRAIN_RECOMMENDED": "Retraining recommended", "NO_RETRAINING": "Keep the current model", "WAIT_FOR_DATA": "Wait for better evidence"}.get(record["decision"]["decision"], record["decision"]["decision"]))
    for reason in record["decision"]["reasoning"]:
        st.write(reason)
    st.write(record["decision"].get("uncertainty", ""))
    st.subheader("Metrics and thresholds")
    evidence = record["decision"]["evidence"]
    st.caption(f"Evidence generated: {(record.get('monitoring_snapshot') or {}).get('generated_at', 'Unknown')} | Engine: {record.get('engine', 'Unknown')}")
    st.caption("Saved batch evaluation; opening this page does not recompute metrics. Confidence has not been calibrated.")
    for metric, label in (("mae", "Average prediction error"), ("rmse", "Error emphasizing large misses")):
        current, baseline, threshold = evidence.get(f"current_{metric}"), evidence.get(f"baseline_{metric}"), evidence.get(f"{metric}_threshold")
        if all(isinstance(value, (float, int)) for value in (current, baseline, threshold)):
            change = f"{(current / baseline - 1) * 100:+.1f}%" if baseline > 0 else "not available"
            st.write(f"{label}: {current:.2f} cycles versus {baseline:.2f} at baseline ({change}). Review threshold: {threshold:.2f} cycles.")
    st.write(f"Drifted features: {evidence.get('drifted_feature_count', 'Unknown')}; required: {evidence.get('minimum_drift_features', 'Unknown')}.")
    snapshot = record.get("monitoring_snapshot") or {}
    st.write(f"Labeled observations: {snapshot.get('sample_count', 'Unknown')}")
    if snapshot.get("drifted_features"):
        st.write("Features flagged: " + ", ".join(snapshot["drifted_features"]))
    st.subheader("Lifecycle")
    for entry in record["lifecycle"]:
        st.write(f"**{entry['status'].replace('_', ' ').title()}** - {entry['timestamp']}")
        if entry.get("reason"):
            st.write(entry["reason"])
        gate = entry.get("evaluation", {}).get("gate", entry.get("gate", {}))
        if gate:
            st.write("Candidate passed evaluation." if gate.get("eligible") else "Candidate blocked; incumbent retained.")
            if gate.get("incumbent") and gate.get("candidate"):
                st.table({"Incumbent": gate["incumbent"], "Candidate": gate["candidate"]})
        if entry.get("workflow", {}).get("run_url"):
            st.write("CI run: " + entry["workflow"]["run_url"])
    with st.expander("Full evidence snapshot and model identity"):
        st.json(record)
    st.download_button("Download decision record", json.dumps(record, indent=2),
                       file_name=record["decision_id"] + ".json", mime="application/json")
except (ValueError, KeyError, OSError) as error:
    st.error(f"Cannot read decision record: {error}")
st.caption("CI results are stored in the GitHub Actions retraining-audit artifact with the same decision ID.")
