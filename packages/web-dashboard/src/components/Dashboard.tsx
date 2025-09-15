import dayjs from 'dayjs';
export const Dashboard: React.FC = () => {
  const [isManualRefresh, setIsManualRefresh] = useState(false);
  const liveMetrics = useLiveData(5000); // تحديث كل 5 ثواني

  const handleRefresh = () => {
    setIsManualRefresh(true);
    setTimeout(() => setIsManualRefresh(false), 1000);
  };

  return (
    <div className="dashboard">
      <header className="dashboard-header">
        <div>
          <h1>AI Model Sentinel Dashboard</h1>
          <p>Real-time monitoring and governance</p>
        </div>
        <RefreshIndicator
          lastUpdated={liveMetrics.lastUpdated}
          isUpdating={isManualRefresh}
        />
      </header>

      <div className="dashboard-content">
        <div className="charts-section">
          <PerformanceChart metrics={liveMetrics} />
        </div>

        <div className="status-cards">
          <div className="status-card">
            <h3>📊 Model Accuracy</h3>
            <p className="metric-value">{liveMetrics.accuracy.toFixed(1)}%</p>
            <p className="metric-trend">
              {liveMetrics.accuracy > 92 ? '↑' : '↓'}
              {Math.abs(liveMetrics.accuracy - 92).toFixed(1)}% from baseline
            </p>
          </div>

          <div className="status-card">
            <h3>⚠️ Active Alerts</h3>
            <p className="metric-value">{liveMetrics.activeAlerts}</p>
            <p className="metric-trend">
              {liveMetrics.activeAlerts > 0 ? 'Needs attention' : 'All systems normal'}
            </p>
          </div>

          <div className="status-card">
            <h3>🔄 Data Drift</h3>
            <p className="metric-value">{liveMetrics.dataDrift.toFixed(2)}</p>
            <p className="metric-trend">
              {liveMetrics.dataDrift > 0.2 ? 'High' : liveMetrics.dataDrift > 0.1 ? 'Medium' : 'Low'}
            </p>
          </div>

          <div className="status-card">
            <h3>⚡ Inference Time</h3>
            <p className="metric-value">{liveMetrics.inferenceTime}ms</p>
            <p className="metric-trend">
              {liveMetrics.inferenceTime > 50 ? 'Slow' : 'Optimal'}
            </p>
          </div>
        </div>
      </div>

      <button className="refresh-button" onClick={handleRefresh}>
        Refresh Data
      </button>
    </div>
  );
};