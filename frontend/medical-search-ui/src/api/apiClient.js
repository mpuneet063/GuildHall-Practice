// ./api/apiClient.js
import axios from 'axios';

const api = axios.create({ baseURL: 'http://127.0.0.1:8000' });

export const fetchInsights = async () => {
    const res = await api.get('/insights');
    return res.data.insights;
};

export const fetchRecommendations = async (selectedKeywords, sortBy) => {
    // Matches the FastAPI Pydantic model: { "selected_keywords": [...] }
    const res = await api.post('/recommend', {
        selected_keywords: selectedKeywords,
        sort_by: sortBy // Make sure FastAPI expects this field
    });
    return res.data;
};