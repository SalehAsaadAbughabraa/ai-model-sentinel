import React from 'react';
import { RefreshCw } from 'lucide-react';

interface RefreshIndicatorProps {
  lastUpdated: string;
  isUpdating?: boolean;
}

export const RefreshIndicator: React.FC<RefreshIndicatorProps> = ({
  lastUpdated,
  isUpdating = false
}) => {
  return (
    <div className="refresh-indicator">
      <RefreshCw 
        size={16} 
        className={isUpdating ? 'spinning' : ''} 
      />
      <span>
        Last updated: {new Date(lastUpdated).toLocaleTimeString()}
      </span>
      {isUpdating && <span className="updating-text">Updating...</span>}
    </div>
  );
};