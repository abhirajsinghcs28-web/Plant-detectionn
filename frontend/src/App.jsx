import React, { useState, useEffect } from 'react';
import Navbar from './components/Navbar';
import Home from './pages/Home';
import Results from './pages/Results';
import History from './pages/History';
import { checkHealth } from './services/api';
import './App.css';

/**
 * Root Application Component
 * Agritech Self-Supervised Cross-Domain Vision Transformers
 * for Fine-Grained Crop Leaf Disease Severity Estimation
 */
export default function App() {
  const [activeTab, setActiveTab] = useState('analyze'); // 'analyze', 'history', 'results'
  const [currentResult, setCurrentResult] = useState(null);
  const [systemStatus, setSystemStatus] = useState({
    status: 'checking',
    database: 'checking',
  });

  // Check backend server and database connectivity on load
  const verifySystemHealth = async () => {
    try {
      const health = await checkHealth();
      setSystemStatus(health);
    } catch {
      setSystemStatus({ status: 'offline', database: 'disconnected' });
    }
  };

  useEffect(() => {
    verifySystemHealth();
    // Periodically verify connection every 30 seconds
    const interval = setInterval(verifySystemHealth, 30000);
    return () => clearInterval(interval);
  }, []);

  // When a prediction is received, we keep currentResult and stay on analyze/home with results below,
  // or user can inspect detailed results
  const handlePredictionCompleted = (result) => {
    setCurrentResult(result);
  };

  // Select prediction from history tab to view details
  const handleSelectPrediction = (item) => {
    setCurrentResult(item);
    setActiveTab('results');
  };

  return (
    <div className="app-container">
      {/* Top Academic Navbar */}
      <Navbar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        systemStatus={systemStatus}
      />

      {/* Main Content Workspace */}
      <main className="main-content">
        {activeTab === 'analyze' && (
          <Home
            onPredictionCompleted={handlePredictionCompleted}
            currentResult={currentResult}
            setCurrentResult={setCurrentResult}
          />
        )}

        {activeTab === 'history' && (
          <History onSelectPrediction={handleSelectPrediction} />
        )}

        {activeTab === 'results' && (
          <Results
            result={currentResult}
            onBackToAnalyze={() => setActiveTab('analyze')}
          />
        )}
      </main>

      {/* Academic Research Footer */}
      <footer className="app-footer">
        <p>
          Research Project: <strong>AI-Powered Crop Leaf Severity Analysis using Vision Transformers (ViT)</strong>
        </p>
        <p style={{ marginTop: '4px' }}>
          Architecture: Vision Transformer (ViT) • Input: 224 × 224 RGB • 4 Severity Levels (Healthy, Mild, Moderate, Severe)
        </p>
      </footer>
    </div>
  );
}
