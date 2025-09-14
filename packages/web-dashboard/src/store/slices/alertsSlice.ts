import { createSlice, PayloadAction } from '@reduxjs/toolkit';

interface Alert {
  id: string;
  message: string;
  severity: 'error' | 'warning' | 'info' | 'success';
  timestamp: Date;
}

interface AlertsState {
  alerts: Alert[];
  unreadCount: number;
}

const initialState: AlertsState = {
  alerts: [],
  unreadCount: 0,
};

const alertsSlice = createSlice({
  name: 'alerts',
  initialState,
  reducers: {
    addAlert: (state, action: PayloadAction<Omit<Alert, 'id'>>) => {
      const newAlert: Alert = {
        ...action.payload,
        id: Date.now().toString(),
        timestamp: new Date(),
      };
      state.alerts.unshift(newAlert);
      state.unreadCount += 1;
    },
    markAsRead: (state) => {
      state.unreadCount = 0;
    },
    clearAlerts: (state) => {
      state.alerts = [];
      state.unreadCount = 0;
    },
  },
});

export const { addAlert, markAsRead, clearAlerts } = alertsSlice.actions;
export default alertsSlice.reducer;