import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

interface PerformanceChartProps {
  metrics: {
    accuracy: number;
    dataDrift: number;
    lastUpdated: string;
  };
}

const data = [
  { time: '00:00', accuracy: 92, drift: 0.1 },
  { time: '06:00', accuracy: 94, drift: 0.08 },
  { time: '12:00', accuracy: 89, drift: 0.22 },
  { time: '18:00', accuracy: 94, drift: 0.11 },
];

export const PerformanceChart: React.FC<PerformanceChartProps> = ({ metrics }) => {
  return (
    <div className="chart-container">
      <h3>Model Performance & Data Drift</h3>
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="time" />
          <YAxis yAxisId="left" />
          <YAxis yAxisId="right" orientation="right" />
          <Tooltip />
          <Line 
            yAxisId="left"
            type="monotone" 
            dataKey="accuracy" 
            stroke="#8884d8" 
            strokeWidth={2}
            name="Accuracy %"
          />
          <Line 
            yAxisId="right"
            type="monotone" 
            dataKey="drift" 
            stroke="#82ca9d" 
            strokeWidth={2}
            name="Data Drift"
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};