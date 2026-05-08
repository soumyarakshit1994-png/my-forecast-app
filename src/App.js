import React, { useState } from 'react';
import { Download, Upload, TrendingUp, AlertCircle, BarChart3, Settings } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ComposedChart } from 'recharts';

const EnhancedForecastingApp = () => {
  const [data, setData] = useState([]);
  const [fileName, setFileName] = useState('');
  const [selectedModels, setSelectedModels] = useState(['exponential_smoothing', 'linear_regression']);
  const [forecastHorizon, setForecastHorizon] = useState(12);
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [dataPreview, setDataPreview] = useState([]);
  const [transformation, setTransformation] = useState('none');
  const [useEnsemble, setUseEnsemble] = useState(true);
  const [ensembleMethod, setEnsembleMethod] = useState('mean');

  const models = [
    { id: 'simple_moving_avg', name: 'Simple Moving Average', description: 'Uses average of past values' },
    { id: 'exponential_smoothing', name: 'Exponential Smoothing', description: 'Weights recent values more heavily' },
    { id: 'linear_regression', name: 'Linear Regression', description: 'Fits a linear trend' },
    { id: 'polynomial_regression', name: 'Polynomial Regression', description: 'Fits a polynomial trend' },
    { id: 'seasonal_naive', name: 'Seasonal Naive', description: 'Uses seasonal patterns' },
    { id: 'arima_simple', name: 'ARIMA (Simplified)', description: 'Autoregressive model' }
  ];

  const transformData = (values, type) => {
    if (type === 'none') return values;
    if (type === 'log') return values.map(v => Math.log(Math.max(v, 0.01)));
    if (type === 'diff') {
      const diff = [];
      for (let i = 1; i < values.length; i++) {
        diff.push(values[i] - values[i-1]);
      }
      return diff;
    }
    if (type === 'pct_change') {
      const pct = [];
      for (let i = 1; i < values.length; i++) {
        pct.push((values[i] - values[i-1]) / Math.abs(values[i-1]));
      }
      return pct;
    }
    return values;
  };

  const inverseTransform = (values, type, originalData) => {
    if (type === 'none') return values;
    if (type === 'log') return values.map(v => Math.exp(v));
    if (type === 'diff') {
      const result = [originalData[originalData.length - 1]];
      for (let i = 0; i < values.length; i++) {
        result.push(result[result.length - 1] + values[i]);
      }
      return result.slice(1);
    }
    if (type === 'pct_change') {
      const result = [originalData[originalData.length - 1]];
      for (let i = 0; i < values.length; i++) {
        result.push(result[result.length - 1] * (1 + values[i]));
      }
      return result.slice(1);
    }
    return values;
  };

  const calculateMetrics = (actual, predicted) => {
    const n = actual.length;
    const errors = actual.map((a, i) => a - (predicted[i] || 0));
    const mae = errors.reduce((sum, e) => sum + Math.abs(e), 0) / n;
    const mse = errors.reduce((sum, e) => sum + e * e, 0) / n;
    const rmse = Math.sqrt(mse);
    const mape = 100 * errors.reduce((sum, e, i) => sum + Math.abs(e / Math.max(Math.abs(actual[i]), 0.001)), 0) / n;
    const ssRes = errors.reduce((sum, e) => sum + e * e, 0);
    const meanActual = actual.reduce((s, a) => s + a) / n;
    const ssMean = actual.reduce((sum, a) => sum + (a - meanActual) ** 2, 0);
    const r2 = Math.max(-1, Math.min(1, 1 - (ssRes / (ssMean || 1))));
    
    return { mae, rmse, mape, r2 };
  };

  const forecasters = {
    simple_moving_avg: (timeSeries, horizon, window = 3) => {
      const forecast = [];
      let lastWindow = timeSeries.slice(-window);
      const residuals = [];
      
      for (let i = 0; i < timeSeries.length - window; i++) {
        const predicted = timeSeries.slice(i, i + window).reduce((a, b) => a + b) / window;
        residuals.push(Math.abs(timeSeries[i + window] - predicted));
      }
      const stdDev = Math.sqrt(residuals.reduce((a, b) => a + b * b, 0) / Math.max(residuals.length, 1));
      
      for (let i = 0; i < horizon; i++) {
        const avg = lastWindow.reduce((a, b) => a + b, 0) / lastWindow.length;
        forecast.push({
          point: avg,
          lower: avg - 1.96 * stdDev,
          upper: avg + 1.96 * stdDev
        });
        lastWindow = [...lastWindow.slice(1), avg];
      }
      return forecast;
    },

    exponential_smoothing: (timeSeries, horizon, alpha = 0.3) => {
      const forecast = [];
      let s = timeSeries[0];
      const residuals = [];
      
      for (let t = 1; t < timeSeries.length; t++) {
        const predicted = s;
        residuals.push(Math.abs(timeSeries[t] - predicted));
        s = alpha * timeSeries[t] + (1 - alpha) * s;
      }
      const stdDev = Math.sqrt(residuals.reduce((a, b) => a + b * b, 0) / Math.max(residuals.length, 1));
      
      for (let i = 0; i < horizon; i++) {
        forecast.push({
          point: s,
          lower: s - 1.96 * stdDev * (1 + i * 0.1),
          upper: s + 1.96 * stdDev * (1 + i * 0.1)
        });
      }
      return forecast;
    },

    linear_regression: (timeSeries, horizon) => {
      const n = timeSeries.length;
      const x = Array.from({ length: n }, (_, i) => i);
      const meanX = x.reduce((a, b) => a + b) / n;
      const meanY = timeSeries.reduce((a, b) => a + b) / n;
      
      let slope = 0, denominator = 0;
      for (let i = 0; i < n; i++) {
        slope += (x[i] - meanX) * (timeSeries[i] - meanY);
        denominator += (x[i] - meanX) ** 2;
      }
      slope /= (denominator || 1);
      const intercept = meanY - slope * meanX;
      
      const residuals = timeSeries.map((y, i) => Math.abs(y - (intercept + slope * i)));
      const stdDev = Math.sqrt(residuals.reduce((a, b) => a + b * b, 0) / Math.max(residuals.length, 1));
      
      const forecast = [];
      for (let i = 0; i < horizon; i++) {
        const point = intercept + slope * (n + i);
        forecast.push({
          point,
          lower: point - 1.96 * stdDev,
          upper: point + 1.96 * stdDev
        });
      }
      return forecast;
    },

    polynomial_regression: (timeSeries, horizon) => {
      const n = timeSeries.length;
      const x = Array.from({ length: n }, (_, i) => i);
      
      const sumX = x.reduce((a, b) => a + b);
      const sumY = timeSeries.reduce((a, b) => a + b);
      const sumX2 = x.reduce((a, xi) => a + xi * xi);
      const sumX3 = x.reduce((a, xi) => a + xi * xi * xi);
      const sumX4 = x.reduce((a, xi) => a + xi * xi * xi * xi);
      const sumXY = x.reduce((a, xi, i) => a + xi * timeSeries[i]);
      const sumX2Y = x.reduce((a, xi, i) => a + xi * xi * timeSeries[i]);
      
      const A = [[n, sumX, sumX2], [sumX, sumX2, sumX3], [sumX2, sumX3, sumX4]];
      const b = [sumY, sumXY, sumX2Y];
      
      for (let i = 0; i < 3; i++) {
        for (let j = i + 1; j < 3; j++) {
          const factor = A[j][i] / (A[i][i] || 0.0001);
          for (let k = i; k < 3; k++) A[j][k] -= factor * A[i][k];
          b[j] -= factor * b[i];
        }
      }
      
      const coeff = [0, 0, 0];
      for (let i = 2; i >= 0; i--) {
        coeff[i] = b[i];
        for (let j = i + 1; j < 3; j++) coeff[i] -= A[i][j] * coeff[j];
        coeff[i] /= (A[i][i] || 0.0001);
      }
      
      const residuals = timeSeries.map((y, i) => Math.abs(y - (coeff[0] + coeff[1] * i + coeff[2] * i * i)));
      const stdDev = Math.sqrt(residuals.reduce((a, b) => a + b * b, 0) / Math.max(residuals.length, 1));
      
      const forecast = [];
      for (let i = 0; i < horizon; i++) {
        const xi = n + i;
        const point = coeff[0] + coeff[1] * xi + coeff[2] * xi * xi;
        forecast.push({
          point,
          lower: point - 1.96 * stdDev,
          upper: point + 1.96 * stdDev
        });
      }
      return forecast;
    },

    seasonal_naive: (timeSeries, horizon, season = 12) => {
      const residuals = [];
      for (let i = season; i < timeSeries.length; i++) {
        residuals.push(Math.abs(timeSeries[i] - timeSeries[i - season]));
      }
      const stdDev = Math.sqrt(residuals.reduce((a, b) => a + b * b, 0) / Math.max(residuals.length, 1));
      
      const forecast = [];
      for (let i = 0; i < horizon; i++) {
        const idx = (timeSeries.length - season + (i % season)) % timeSeries.length;
        const point = timeSeries[idx];
        forecast.push({
          point,
          lower: point - 1.96 * stdDev,
          upper: point + 1.96 * stdDev
        });
      }
      return forecast;
    },

    arima_simple: (timeSeries, horizon) => {
      const n = timeSeries.length;
      const mean = timeSeries.reduce((a, b) => a + b) / n;
      
      let numerator = 0, denominator = 0;
      for (let i = 1; i < n; i++) {
        numerator += (timeSeries[i] - mean) * (timeSeries[i - 1] - mean);
        denominator += (timeSeries[i] - mean) ** 2;
      }
      const phi = denominator ? numerator / denominator : 0;
      
      const residuals = [];
      let predicted = timeSeries[0];
      for (let i = 1; i < n; i++) {
        predicted = mean + phi * (timeSeries[i - 1] - mean);
        residuals.push(Math.abs(timeSeries[i] - predicted));
      }
      const stdDev = Math.sqrt(residuals.reduce((a, b) => a + b * b, 0) / Math.max(residuals.length, 1));
      
      const forecast = [];
      let lastValue = timeSeries[n - 1];
      for (let i = 0; i < horizon; i++) {
        const next = mean + phi * (lastValue - mean);
        forecast.push({
          point: next,
          lower: next - 1.96 * stdDev * (1 + i * 0.05),
          upper: next + 1.96 * stdDev * (1 + i * 0.05)
        });
        lastValue = next;
      }
      return forecast;
    }
  };

  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setFileName(file.name);
    const reader = new FileReader();
    
    reader.onload = (event) => {
      try {
        const csv = event.target.result;
        const lines = csv.split('\n').filter(line => line.trim());
        
        const headers = lines[0].split(',').map(h => h.trim());
        const valueColumnIndex = headers.length > 1 ? 1 : 0;
        
        const parsedData = [];
        for (let i = 1; i < lines.length; i++) {
          const values = lines[i].split(',').map(v => v.trim());
          if (values[valueColumnIndex]) {
            parsedData.push({
              timestamp: values[0] || `Period ${i}`,
              value: parseFloat(values[valueColumnIndex])
            });
          }
        }

        if (parsedData.length < 3) {
          setError('Need at least 3 data points to forecast');
          return;
        }

        setData(parsedData);
        setDataPreview(parsedData.slice(0, 5));
        setError('');
        setResults(null);
      } catch (err) {
        setError('Failed to parse file. Ensure it\'s a valid CSV.');
      }
    };
    
    reader.readAsText(file);
  };

  const handleModelToggle = (modelId) => {
    setSelectedModels(prev => 
      prev.includes(modelId)
        ? prev.filter(m => m !== modelId)
        : [...prev, modelId]
    );
  };

  const generateForecasts = async () => {
    if (data.length === 0) {
      setError('Please upload data first');
      return;
    }
    if (selectedModels.length === 0) {
      setError('Select at least one model');
      return;
    }

    setLoading(true);
    setError('');

    try {
      await new Promise(resolve => setTimeout(resolve, 800));

      const timeSeries = data.map(d => d.value);
      const transformedSeries = transformData(timeSeries, transformation);
      
      const forecastsObj = {};
      const metricsObj = {};

      selectedModels.forEach(modelId => {
        const rawForecast = forecasters[modelId](transformedSeries, forecastHorizon);
        const forecast = rawForecast.map(f => ({
          point: inverseTransform([f.point], transformation, timeSeries)[0],
          lower: inverseTransform([f.lower], transformation, timeSeries)[0],
          upper: inverseTransform([f.upper], transformation, timeSeries)[0]
        }));
        forecastsObj[modelId] = forecast;

        const testSize = Math.floor(transformedSeries.length / 3);
        const predictions = transformedSeries.slice(0, -testSize).map((_, i) => {
          return forecasters[modelId](transformedSeries.slice(0, transformedSeries.length - testSize + i), 1)[0].point;
        });
        const actual = transformedSeries.slice(-testSize);
        metricsObj[modelId] = calculateMetrics(actual, predictions);
      });

      let ensembleForecast = null;
      if (useEnsemble && selectedModels.length > 1) {
        ensembleForecast = [];
        for (let i = 0; i < forecastHorizon; i++) {
          const points = selectedModels.map(m => forecastsObj[m][i].point);
          const lowers = selectedModels.map(m => forecastsObj[m][i].lower);
          const uppers = selectedModels.map(m => forecastsObj[m][i].upper);

          let point, lower, upper;
          if (ensembleMethod === 'mean') {
            point = points.reduce((a, b) => a + b) / points.length;
            lower = lowers.reduce((a, b) => a + b) / lowers.length;
            upper = uppers.reduce((a, b) => a + b) / uppers.length;
          } else if (ensembleMethod === 'median') {
            const sorted = [...points].sort((a, b) => a - b);
            point = sorted[Math.floor(sorted.length / 2)];
            const sortedL = [...lowers].sort((a, b) => a - b);
            lower = sortedL[Math.floor(sortedL.length / 2)];
            const sortedU = [...uppers].sort((a, b) => a - b);
            upper = sortedU[Math.floor(sortedU.length / 2)];
          } else if (ensembleMethod === 'weighted') {
            const weights = selectedModels.map(m => Math.max(metricsObj[m].r2, 0.1));
            const totalWeight = weights.reduce((a, b) => a + b);
            point = points.reduce((sum, p, i) => sum + p * (weights[i] / totalWeight), 0);
            lower = lowers.reduce((sum, l, i) => sum + l * (weights[i] / totalWeight), 0);
            upper = uppers.reduce((sum, u, i) => sum + u * (weights[i] / totalWeight), 0);
          }

          ensembleForecast.push({ point, lower, upper });
        }
      }

      const combinedForecast = [];
      for (let i = 0; i < forecastHorizon; i++) {
        const point = {
          period: data.length + i + 1,
          timestamp: `Forecast ${i + 1}`
        };
        selectedModels.forEach(modelId => {
          point[modelId] = forecastsObj[modelId][i].point;
          point[`${modelId}_lower`] = forecastsObj[modelId][i].lower;
          point[`${modelId}_upper`] = forecastsObj[modelId][i].upper;
        });
        if (ensembleForecast) {
          point.ensemble = ensembleForecast[i].point;
          point.ensemble_lower = ensembleForecast[i].lower;
          point.ensemble_upper = ensembleForecast[i].upper;
        }
        combinedForecast.push(point);
      }

      const chartData = data.map((d, i) => ({
        period: i + 1,
        timestamp: d.timestamp,
        value: d.value
      }));

      selectedModels.forEach(modelId => {
        forecastsObj[modelId].forEach((val, i) => {
          if (!chartData[data.length + i]) {
            chartData[data.length + i] = {
              period: data.length + i + 1,
              timestamp: `Forecast ${i + 1}`
            };
          }
          chartData[data.length + i][modelId] = val.point;
        });
      });

      if (ensembleForecast) {
        ensembleForecast.forEach((val, i) => {
          chartData[data.length + i].ensemble = val.point;
        });
      }

      setResults({
        forecasts: combinedForecast,
        chartData: chartData,
        metrics: metricsObj,
        ensemble: !!ensembleForecast,
        modelCount: selectedModels.length
      });
    } catch (err) {
      setError('Error generating forecasts: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  const downloadResults = () => {
    if (!results) return;

    const headers = ['Period', 'Timestamp', ...selectedModels.flatMap(m => [
      models.find(mo => mo.id === m)?.name,
      `${models.find(mo => mo.id === m)?.name} Lower`,
      `${models.find(mo => mo.id === m)?.name} Upper`
    ])];
    
    if (results.ensemble) {
      headers.push('Ensemble', 'Ensemble Lower', 'Ensemble Upper');
    }
    
    let csv = headers.join(',') + '\n';
    results.forecasts.forEach(row => {
      const values = [
        row.period,
        row.timestamp,
        ...selectedModels.flatMap(modelId => [
          row[modelId].toFixed(4),
          row[`${modelId}_lower`].toFixed(4),
          row[`${modelId}_upper`].toFixed(4)
        ])
      ];
      if (results.ensemble) {
        values.push(
          row.ensemble.toFixed(4),
          row.ensemble_lower.toFixed(4),
          row.ensemble_upper.toFixed(4)
        );
      }
      csv += values.join(',') + '\n';
    });

    const blob = new Blob([csv], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `forecasts_${new Date().toISOString().split('T')[0]}.csv`;
    a.click();
  };

  const colors = ['#3b82f6', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6', '#ec4899'];

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 to-slate-800 p-6 md:p-8">
      <div className="max-w-7xl mx-auto">
        <div className="mb-8">
          <div className="flex items-center gap-3 mb-2">
            <TrendingUp className="w-8 h-8 text-blue-400" />
            <h1 className="text-3xl md:text-4xl font-bold text-white">Advanced Time Series Forecaster</h1>
          </div>
          <p className="text-slate-300">Multi-model forecasting with confidence intervals, accuracy metrics & ensemble methods</p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 md:gap-8">
          <div className="lg:col-span-1 space-y-4 md:space-y-6">
            <div className="bg-slate-800 rounded-lg p-6 border border-slate-700">
              <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                <Upload className="w-5 h-5" /> Data Upload
              </h2>
              <label className="block cursor-pointer">
                <div className="border-2 border-dashed border-slate-600 rounded-lg p-6 text-center hover:border-blue-400 transition">
                  <input type="file" accept=".csv" onChange={handleFileUpload} className="hidden" />
                  <p className="text-slate-300 text-sm">Drag CSV or click</p>
                  {fileName && <p className="text-blue-400 text-sm mt-2">✓ {fileName}</p>}
                </div>
              </label>
            </div>

            {dataPreview.length > 0 && (
              <div className="bg-slate-800 rounded-lg p-6 border border-slate-700">
                <h3 className="text-sm font-semibold text-white mb-3">Data ({data.length} points)</h3>
                <div className="space-y-2 text-sm">
                  {dataPreview.map((row, i) => (
                    <div key={i} className="text-slate-300">
                      <span className="text-slate-500">{row.timestamp}:</span> {row.value.toFixed(4)}
                    </div>
                  ))}
                  {data.length > 5 && <div className="text-slate-500 text-xs">+{data.length - 5} more</div>}
                </div>
              </div>
            )}

            <div className="bg-slate-800 rounded-lg p-6 border border-slate-700">
              <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                <Settings className="w-5 h-5" /> Transformations
              </h2>
              <select 
                value={transformation}
                onChange={(e) => setTransformation(e.target.value)}
                className="w-full bg-slate-700 text-white rounded px-3 py-2 border border-slate-600 text-sm"
              >
                <option value="none">None</option>
                <option value="log">Log Transform</option>
                <option value="diff">Differencing</option>
                <option value="pct_change">Percent Change</option>
              </select>
            </div>

            <div className="bg-slate-800 rounded-lg p-6 border border-slate-700">
              <h2 className="text-lg font-semibold text-white mb-4">Ensemble</h2>
              <label className="flex items-center gap-3 mb-3">
                <input
                  type="checkbox"
                  checked={useEnsemble}
                  onChange={(e) => setUseEnsemble(e.target.checked)}
                  className="w-4 h-4 rounded"
                />
                <span className="text-white text-sm">Use Ensemble</span>
              </label>
              {useEnsemble && (
                <select 
                  value={ensembleMethod}
                  onChange={(e) => setEnsembleMethod(e.target.value)}
                  className="w-full bg-slate-700 text-white rounded px-3 py-2 border border-slate-600 text-sm"
                >
                  <option value="mean">Mean</option>
                  <option value="median">Median</option>
                  <option value="weighted">Weighted by Accuracy</option>
                </select>
              )}
            </div>

            <div className="bg-slate-800 rounded-lg p-6 border border-slate-700">
              <h2 className="text-lg font-semibold text-white mb-4">Models</h2>
              <div className="space-y-2">
                {models.map((model) => (
                  <label key={model.id} className="flex items-center gap-2 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={selectedModels.includes(model.id)}
                      onChange={() => handleModelToggle(model.id)}
                      className="w-4 h-4"
                    />
                    <span className="text-white text-sm">{model.name}</span>
                  </label>
                ))}
              </div>
            </div>

            <div className="bg-slate-800 rounded-lg p-6 border border-slate-700">
              <label className="text-slate-300 text-sm mb-2 block">
                Periods: <span className="text-blue-400 font-bold">{forecastHorizon}</span>
              </label>
              <input
                type="range"
                min="1"
                max="48"
                value={forecastHorizon}
                onChange={(e) => setForecastHorizon(parseInt(e.target.value))}
                className="w-full"
              />
            </div>

            <button
              onClick={generateForecasts}
              disabled={loading || data.length === 0 || selectedModels.length === 0}
              className="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-slate-700 text-white font-semibold py-3 rounded-lg"
            >
              {loading ? 'Generating...' : 'Generate'}
            </button>

            {results && (
              <button
                onClick={downloadResults}
                className="w-full bg-green-600 hover:bg-green-700 text-white font-semibold py-3 rounded-lg flex items-center justify-center gap-2"
              >
                <Download className="w-4 h-4" /> Download
              </button>
            )}
          </div>

          <div className="lg:col-span-2 space-y-4 md:space-y-6">
            {error && (
              <div className="bg-red-900/30 border border-red-700 rounded-lg p-4 flex gap-3">
                <AlertCircle className="w-5 h-5 text-red-400" />
                <p className="text-red-200 text-sm">{error}</p>
              </div>
            )}

            {results ? (
              <>
                {Object.keys(results.metrics).length > 0 && (
                  <div className="bg-slate-800 rounded-lg p-6 border border-slate-700">
                    <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                      <BarChart3 className="w-5 h-5" /> Accuracy Metrics
                    </h2>
                    <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                      {selectedModels.map(modelId => {
                        const metrics = results.metrics[modelId];
                        const modelName = models.find(m => m.id === modelId)?.name;
                        return (
                          <div key={modelId} className="bg-slate-700 rounded p-4">
                            <p className="text-slate-300 text-xs font-medium mb-2 truncate">{modelName}</p>
                            <div className="space-y-1 text-xs">
                              <div><span className="text-slate-400">MAE:</span> <span className="text-blue-300">{metrics.mae.toFixed(3)}</span></div>
                              <div><span className="text-slate-400">RMSE:</span> <span className="text-blue-300">{metrics.rmse.toFixed(3)}</span></div>
                              <div><span className="text-slate-400">MAPE:</span> <span className="text-blue-300">{metrics.mape.toFixed(1)}%</span></div>
                              <div><span className="text-slate-400">R²:</span> <span className="text-blue-300">{metrics.r2.toFixed(3)}</span></div>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                )}

                <div className="bg-slate-800 rounded-lg p-6 border border-slate-700">
                  <h2 className="text-lg font-semibold text-white mb-4">Visualization</h2>
                  <ResponsiveContainer width="100%" height={300}>
                    <ComposedChart data={results.chartData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#475569" />
                      <XAxis dataKey="timestamp" stroke="#94a3b8" tick={{ fontSize: 10 }} />
                      <YAxis stroke="#94a3b8" />
                      <Tooltip contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #475569' }} />
                      <Legend wrapperStyle={{ color: '#e2e8f0' }} />
                      <Line type="monotone" dataKey="value" stroke="#94a3b8" name="Historical" dot={false} strokeWidth={2} />
                      {selectedModels.map((modelId, idx) => (
                        <Line
                          key={modelId}
                          type="monotone"
                          dataKey={modelId}
                          stroke={colors[idx % colors.length]}
                          name={models.find(m => m.id === modelId)?.name}
                          strokeDasharray="5 5"
                          dot={false}
                          strokeWidth={2}
                        />
                      ))}
                      {results.ensemble && (
                        <Line type="monotone" dataKey="ensemble" stroke="#06b6d4" name="Ensemble" strokeWidth={3} dot={false} />
                      )}
                    </ComposedChart>
                  </ResponsiveContainer>
                </div>

                <div className="bg-slate-800 rounded-lg p-6 border border-slate-700">
                  <h2 className="text-lg font-semibold text-white mb-4">Forecasts (95% CI)</h2>
                  <div className="overflow-x-auto">
                    <table className="w-full text-xs">
                      <thead>
                        <tr className="border-b border-slate-700">
                          <th className="text-left py-2 px-2 text-slate-300">Period</th>
                          {selectedModels.map(modelId => (
                            <th key={modelId} className="text-center py-2 px-2 text-slate-300">{models.find(m => m.id === modelId)?.name}</th>
                          ))}
                          {results.ensemble && <th className="text-center py-2 px-2 text-slate-300">Ensemble</th>}
                        </tr>
                      </thead>
                      <tbody>
                        {results.forecasts.slice(0, 12).map((row, i) => (
                          <tr key={i} className="border-b border-slate-700 hover:bg-slate-700/50">
                            <td className="py-2 px-2 text-slate-300">{row.timestamp}</td>
                            {selectedModels.map(modelId => (
                              <td key={modelId} className="text-center py-2 px-2">
                                <div className="text-slate-200">{row[modelId].toFixed(3)}</div>
                                <div className="text-slate-400 text-xs">[{row[`${modelId}_lower`].toFixed(2)}, {row[`${modelId}_upper`].toFixed(2)}]</div>
                              </td>
                            ))}
                            {results.ensemble && (
                              <td className="text-center py-2 px-2">
                                <div className="text-cyan-300 font-semibold">{row.ensemble.toFixed(3)}</div>
                                <div className="text-slate-400 text-xs">[{row.ensemble_lower.toFixed(2)}, {row.ensemble_upper.toFixed(2)}]</div>
                              </td>
                            )}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                    {results.forecasts.length > 12 && (
                      <p className="text-slate-400 text-xs mt-3">Showing 12 of {results.forecasts.length} periods (download CSV for all)</p>
                    )}
                  </div>
                </div>
              </>
            ) : (
              <div className="bg-slate-800 rounded-lg p-12 border border-slate-700 text-center">
                <TrendingUp className="w-12 h-12 text-slate-600 mx-auto mb-4" />
                <p className="text-slate-400">Upload data to get started</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default EnhancedForecastingApp;