import React from "react";

export default function ResultCard({ bundle }) {
    return (
        <div style={cardStyle}>
            <h3>{bundle.options}</h3>
            <p><strong>Cost (in GBP):</strong> £{bundle.total_cost?.toFixed(2)}</p>
            <p><strong>Turnaround (in days):</strong> {bundle.turnaround_time} days</p>
            <p><strong>What it Covers?:</strong> {bundle.insights}</p>
            <hr />
            <p style={{ fontStyle: "italic", color: "#555" }}>
                {bundle.agent_comparison}
            </p>
        </div>
    );
}

const cardStyle = {
    border: "1px solid #ddd",
    borderRadius: 12,
    padding: 20,
    marginBottom: 16,
    background: "#fafafa",
    boxShadow: "0 2px 8px rgba(0,0,0,0.08)",
};
