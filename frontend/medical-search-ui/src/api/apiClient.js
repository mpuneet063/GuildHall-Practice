import axios from "axios";

const BASE_URL = "http://localhost:8000"; // Ensure this matches your FastAPI port

export const fetchInsights = () =>
    axios.get(`${BASE_URL}/insights`).then((r) => r.data.insights);

export const fetchRecommendations = (selectedKeywords, sortBy) =>
    axios
        .post(`${BASE_URL}/recommend`, {
            selected_keywords: selectedKeywords,
            sort_by: sortBy // New parameter passed to backend [cite: 1104, 1106]
        })
        .then((r) => r.data);