import React, { createContext, useContext, ReactNode } from 'react';

interface RealTimeContextType {
  // سيتم إضافة وظائف حية هنا لاحقاً
}

const RealTimeContext = createContext<RealTimeContextType | undefined>(undefined);

export const RealTimeProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  return (
    <RealTimeContext. value={{}}>
      {children}
    </RealTimeContext.Provider>
  );
};

export const useRealTime = () => {
  const context = useContext(RealTimeContext);
  if (context === undefined) {
    throw new Error('useRealTime must be used within a RealTimeProvider');
  }
  return context;
};