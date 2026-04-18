import React, { useEffect, useState } from "react";
import { fetchInsights, fetchRecommendations } from "./api/apiClient";
import KeywordSelector from "./components/KeywordSelector";
import ResultCard from "./components/ResultCard";
import "./App.css";

export default function App() {
  const [keywords, setKeywords] = useState([]);
  const [selected, setSelected] = useState([]);
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [sortBy, setSortBy] = useState("cost"); // New state for sorting 

  useEffect(() => {
    fetchInsights()
      .then(setKeywords)
      .catch(() => setError("Could not load keywords — is the backend running?"));
  }, []);

  const handleSearch = async () => {
    if (selected.length === 0) return;
    setLoading(true);
    setError(null);
    try {
      // Pass both selected keywords and the sort preference 
      const data = await fetchRecommendations(selected, sortBy);
      setResults(data);
      if (data.length === 0) setError("No matching bundles found.");
    } catch {
      setError("Failed to fetch recommendations.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ maxWidth: 800, margin: "40px auto", padding: "0 20px", fontFamily: "sans-serif" }}>
      <h1>GuildHall Test Recommender</h1>
      <p>Select biomarkers to find the best diagnostic bundle.</p>

      <KeywordSelector options={keywords} onChange={setSelected} />

      <div style={{ display: 'flex', alignItems: 'center', gap: '15px', marginTop: '16px' }}>
        {/* Sort Dropdown  */}
        <select
          value={sortBy}
          onChange={(e) => setSortBy(e.target.value)}
          style={selectStyle}
        >
          <option value="cost">Sort by: Lowest Price</option>
          <option value="turnaround">Sort by: Fastest Results</option>
        </select>

        <button
          onClick={handleSearch}
          disabled={loading || selected.length === 0}
          style={buttonStyle}
        >
          {loading ? "Medical Agent Reasoning..." : "Find Tests"}
        </button>
      </div>

      {error && <p style={{ color: "red" }}>{error}</p>}

      <div style={{ marginTop: 24 }}>
        {results.map((bundle, i) => (
          <ResultCard key={i} bundle={bundle} />
        ))}
      </div>
    </div>
  );
}

const selectStyle = {
  padding: "10px",
  borderRadius: "8px",
  border: "1px solid #ccc",
  background: "#fff",
  fontSize: "14px",
  cursor: "pointer"
};

const buttonStyle = {
  padding: "10px 24px",
  background: "#0066cc",
  color: "#fff",
  border: "none",
  borderRadius: 8,
  fontSize: 16,
  cursor: "pointer",
};