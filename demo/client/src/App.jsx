import React, { useState } from 'react';
import { ReactCompareSlider, ReactCompareSliderImage } from 'react-compare-slider';
import './App.css';

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [restoredUrl, setRestoredUrl] = useState(null);
  const [isRestoring, setIsRestoring] = useState(false);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setRestoredUrl(null);
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) return;

    setIsRestoring(true);
    const formData = new FormData();
    formData.append('image', selectedFile);

    try {
      const response = await fetch('http://localhost:5000/api/restore', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();
      if (data.success) {
        setRestoredUrl(`http://localhost:5000${data.restoredImage}`);
      } else {
        alert('Restoration failed.');
      }
    } catch (error) {
      console.error('Error:', error);
      alert('Error connecting to server.');
    } finally {
      setIsRestoring(false);
    }
  };

  return (
    <div className="dashboard">
      <aside className="sidebar">
        <div className="logo">RESTORIFY AI</div>
        <nav>
          <div className="nav-item active">Dashboard</div>
          <div className="nav-item">History</div>
          <div className="nav-item">Models</div>
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
              {restoredUrl ? (
                <ReactCompareSlider
                  itemOne={<ReactCompareSliderImage src={previewUrl} alt="Original" />}
                  itemTwo={<ReactCompareSliderImage src={restoredUrl} alt="Restored" />}
                  style={{ borderRadius: '1rem', overflow: 'hidden' }}
                />
              ) : (
                <div style={{ position: 'relative' }}>
                  <img src={previewUrl} alt="Preview" style={{ width: '100%', borderRadius: '1rem' }} />
                  {isRestoring && (
                    <div style={{
                      position: 'absolute', inset: 0, background: 'rgba(0,0,0,0.5)', 
                      display: 'flex', alignItems: 'center', justifyContent: 'center', borderRadius: '1rem'
                    }}>
                      <div className="loader">Restoring...</div>
                    </div>
                  )}
                </div>
              )}
              
              <div className="controls">
                <button className="btn btn-primary" onClick={handleUpload} disabled={isRestoring}>
                  {isRestoring ? 'Processing...' : 'Start Restoration'}
                </button>
                <button className="btn" onClick={() => { setPreviewUrl(null); setRestoredUrl(null); }} style={{ background: 'var(--glass)', color: 'white' }}>
                  Clear
                </button>
              </div>
            </div>
          )}
        </section>

        <section className="stats-grid">
          <div className="stat-card">
            <div className="stat-label">PSNR</div>
            <div className="stat-value">32.7 dB</div>
          </div>
          <div className="stat-card">
            <div className="stat-label">SSIM</div>
            <div className="stat-value">0.945</div>
          </div>
          <div className="stat-card">
            <div className="stat-label">VRAM Usage</div>
            <div className="stat-value">3.2 / 4.0 GB</div>
          </div>
        </section>
      </main>
    </div>
  );
}

export default App;
