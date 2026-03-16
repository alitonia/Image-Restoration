import React, { useState } from 'react';
import { ReactCompareSlider, ReactCompareSliderImage } from 'react-compare-slider';
import './App.css';

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [results, setResults] = useState({});
  const [activeMethod, setActiveMethod] = useState('swinir');
  const [isRestoring, setIsRestoring] = useState(false);
  const [restorationMode, setRestorationMode] = useState('swinir'); // swinir, sd, both

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setResults({});
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) return;

    setIsRestoring(true);
    const formData = new FormData();
    formData.append('image', selectedFile);
    formData.append('method', restorationMode);

    const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:5000';

    try {
      const response = await fetch(`${apiUrl}/api/restore`, {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();
      if (data.success) {
        setResults(data.results);
        setActiveMethod(restorationMode === 'both' ? 'swinir' : restorationMode);
      } else {
        alert('Restoration failed: ' + data.error);
      }
    } catch (error) {
      console.error('Error:', error);
      alert('Error connecting to server.');
    } finally {
      setIsRestoring(false);
    }
  };

  const getActiveUrl = () => {
    if (!results[activeMethod]) return null;
    const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:5000';
    return `${apiUrl}${results[activeMethod]}`;
  };

  return (
    <div className="dashboard">
      <aside className="sidebar">
        <div className="logo">RESTORIFY AI</div>
        <nav>
          <div className="nav-item active">Dashboard</div>
          <div className="nav-item">Ablation Study</div>
          <div className="nav-item">History</div>
          <div className="nav-item">Settings</div>
        </nav>
      </aside>

      <main className="main-content">
        <header className="header">
          <h1>Image Restoration</h1>
          <div className="user-profile">
            <div className="btn btn-primary" onClick={() => document.getElementById('file-upload').click()}>
              Upload New Image
            </div>
            <input 
              id="file-upload" 
              type="file" 
              hidden 
              onChange={handleFileChange} 
              accept="image/*"
            />
          </div>
        </header>

        <section className="card">
          {!previewUrl ? (
            <div className="upload-zone" onClick={() => document.getElementById('file-upload').click()}>
              <p>Drag and drop your damaged photo here</p>
              <p style={{ color: '#94a3b8', fontSize: '0.875rem' }}>or click to browse files</p>
            </div>
          ) : (
            <div className="comparison-container">
              <div className="method-selector">
                 <button className={`tab-btn ${restorationMode === 'swinir' ? 'active' : ''}`} onClick={() => setRestorationMode('swinir')}>SwinIR (Deterministic)</button>
                 <button className={`tab-btn ${restorationMode === 'sd' ? 'active' : ''}`} onClick={() => setRestorationMode('sd')}>Stable Diffusion (Generative)</button>
                 <button className={`tab-btn ${restorationMode === 'both' ? 'active' : ''}`} onClick={() => setRestorationMode('both')}>Compare Both</button>
              </div>

              {getActiveUrl() ? (
                <div style={{ width: '100%' }}>
                  {restorationMode === 'both' && (
                    <div className="view-toggle">
                       <button className={activeMethod === 'swinir' ? 'active' : ''} onClick={() => setActiveMethod('swinir')}>Show SwinIR</button>
                       <button className={activeMethod === 'sd' ? 'active' : ''} onClick={() => setActiveMethod('sd')}>Show Stable Diffusion</button>
                    </div>
                  )}
                  <ReactCompareSlider
                    itemOne={<ReactCompareSliderImage src={previewUrl} alt="Original" />}
                    itemTwo={<ReactCompareSliderImage src={getActiveUrl()} alt="Restored" />}
                    style={{ borderRadius: '1rem', overflow: 'hidden', height: '400px' }}
                  />
                </div>
              ) : (
                <div style={{ position: 'relative', width: '100%' }}>
                  <img src={previewUrl} alt="Preview" style={{ width: '100%', borderRadius: '1rem', maxHeight: '400px', objectFit: 'contain' }} />
                  {isRestoring && (
                    <div style={{
                      position: 'absolute', inset: 0, background: 'rgba(0,0,0,0.5)', 
                      display: 'flex', alignItems: 'center', justifyContent: 'center', borderRadius: '1rem'
                    }}>
                      <div className="loader">Restoring using {restorationMode === 'both' ? 'SwinIR & SD' : restorationMode.toUpperCase()}...</div>
                    </div>
                  )}
                </div>
              )}
              
              <div className="controls">
                <button className="btn btn-primary" onClick={handleUpload} disabled={isRestoring}>
                  {isRestoring ? 'Processing...' : 'Start Restoration'}
                </button>
                <button className="btn" onClick={() => { setPreviewUrl(null); setResults({}); }} style={{ background: 'var(--glass)', color: 'white' }}>
                  Clear
                </button>
              </div>
            </div>
          )}
        </section>

        <section className="stats-grid">
          <div className="stat-card">
            <div className="stat-label">Model Strength</div>
            <div className="stat-value">{activeMethod === 'swinir' ? 'Pixel Perfect' : 'Creative Detail'}</div>
          </div>
          <div className="stat-card">
            <div className="stat-label">Method</div>
            <div className="stat-value" style={{ textTransform: 'uppercase' }}>{activeMethod}</div>
          </div>
          <div className="stat-card">
            <div className="stat-label">VRAM Target</div>
            <div className="stat-value">4.0 GB</div>
          </div>
        </section>
      </main>
    </div>
  );
}

export default App;
