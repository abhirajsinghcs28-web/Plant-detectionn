import axios from 'axios';

// Base API instance
const apiClient = axios.create({
  baseURL: '/api',
  timeout: 30000, // 30s timeout for model inference
});

/**
 * Predict crop leaf disease and severity
 * @param {FormData} formData - Contains 'image' file and 'crop' string
 */
export async function predictLeaf(formData) {
  try {
    const response = await apiClient.post('/predict', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  } catch (error) {
    const message =
      error.response?.data?.error ||
      error.message ||
      'Prediction failed. Please ensure the backend server is running and try again.';
    throw new Error(message);
  }
}

/**
 * Fetch past prediction history from MongoDB
 */
export async function fetchPredictions() {
  try {
    const response = await apiClient.get('/predictions');
    return response.data;
  } catch (error) {
    const message =
      error.response?.data?.error ||
      error.message ||
      'Could not retrieve predictions from database.';
    throw new Error(message);
  }
}

/**
 * Delete a prediction history item
 * @param {string} id - MongoDB ObjectId
 */
export async function deletePrediction(id) {
  try {
    const response = await apiClient.delete(`/predictions/${id}`);
    return response.data;
  } catch (error) {
    const message =
      error.response?.data?.error ||
      error.message ||
      'Could not delete prediction record.';
    throw new Error(message);
  }
}

/**
 * Check backend and database health status
 */
export async function checkHealth() {
  try {
    const response = await apiClient.get('/health');
    return response.data;
  } catch (error) {
    return {
      status: 'offline',
      database: 'disconnected',
      error: error.message,
    };
  }
}
