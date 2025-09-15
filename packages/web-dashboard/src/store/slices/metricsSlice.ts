import { createSlice, PayloadAction } from '@reduxjs/toolkit';

interface MetricsState {
  realTimeData: any[];
  historicalData: any[];
  isLoading: boolean;
}

const initialState: MetricsState = {
  realTimeData: [],
  historicalData: [],
  isLoading: false,
};

const metricsSlice = createSlice({
  name: 'metrics',
  initialState,
  reducers: {
    setRealTimeData: (state, action: PayloadAction<any>) => {
      state.realTimeData = [...state.realTimeData, action.payload].slice(-1000);
    },
    setHistoricalData: (state, action: PayloadAction<any[]>) => {
      state.historicalData = action.payload;
    },
    setLoading: (state, action: PayloadAction<boolean>) => {
      state.isLoading = action.payload;
    },
  },
});

export const { setRealTimeData, setHistoricalData, setLoading } = metricsSlice.actions;
export default metricsSlice.reducer;