import React from 'react';

/**
 * Clean Academic Navbar
 * Shows active tabs and honest system / AI engine connectivity status.
 */
export default function Navbar({ activeTab, setActiveTab, systemStatus }) {
  const isOnline = systemStatus?.status === 'online';
  const isDbConnected = systemStatus?.database === 'connected';
  const aiMode = systemStatus?.aiService?.mode || 'mock';
  const aiStatus = systemStatus?.aiService?.status || 'mock_active';

  // Construct honest status label
  let statusText = 'Server Offline';
  let dotClass = 'offline';

  if (isOnline) {
    if (aiMode === 'real') {
      if (aiStatus === 'connected') {
        statusText = `Server Online (DB OK • Real AI Connected)`;
        dotClass = 'online';
      } else {
        statusText = `Server Online (DB OK • AI Service Offline)`;
        dotClass = 'warning';
      }
    } else {
      statusText = `Server Online (DB OK • Mock AI Mode)`;
      dotClass = 'mock';
    }
  }

  return (
    <header className="research-navbar">
      <div className="navbar-container">
        <div className="navbar-brand">
          <div className="brand-badge">VISION TRANSFORMER (ViT) • SEVERITY ESTIMATION</div>
          <h1 className="brand-title">AI-Powered Crop Leaf Severity Analysis</h1>
          <p className="brand-subtitle">
            Upload a crop leaf image to estimate the severity of visible symptoms using our Vision Transformer-based AI model.
          </p>
        </div>

        <div className="navbar-controls">
          <nav className="nav-tabs">
            <button
              className={`nav-btn ${activeTab === 'analyze' ? 'active' : ''}`}
              onClick={() => setActiveTab('analyze')}
            >
              Analyze Severity
            </button>
            <button
              className={`nav-btn ${activeTab === 'history' ? 'active' : ''}`}
              onClick={() => setActiveTab('history')}
            >
              Analysis History
            </button>
          </nav>

          <div className="system-status-indicator">
            <span
              className={`status-dot ${dotClass}`}
              title={statusText}
            ></span>
            <span className="status-text">{statusText}</span>
          </div>
        </div>
      </div>
    </header>
  );
}
