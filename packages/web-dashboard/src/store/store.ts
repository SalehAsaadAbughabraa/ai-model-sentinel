import { configureStore } from '@reduxjs/toolkit';
import metricsReducer from './slices/metricsSlice';
import alertsReducer from './slices/alertsSlice';

export const store = configureStore({
  reducer: {
    metrics: metricsReducer,
    alerts: alertsReducer,
  },
  middleware: (getDefaultMiddleware) =>
    getDefaultMiddleware({
      serializableCheck: {
        ignoredActions: ['persist/PERSIST'],
      },
    }),
});

export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;