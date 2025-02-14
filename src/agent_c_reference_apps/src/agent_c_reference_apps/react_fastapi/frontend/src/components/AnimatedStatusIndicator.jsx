import React from 'react';

const AnimatedStatusIndicator = ({ isProcessing, className = '' }) => {
  return (
    <div className="relative">
      <div
        className={`
          w-3 h-3 rounded-full 
          transition-all duration-300 ease-in-out
          ${className}
          ${isProcessing ? 'scale-110' : 'scale-100'}
        `}
      >
        {/* Ripple effect when processing */}
        {isProcessing && (
          <>
            <div className={`
              absolute inset-0 rounded-full 
              ${className} opacity-75
              animate-ping
            `} />
            <div className={`
              absolute -inset-1 rounded-full 
              ${className} opacity-50
              animate-pulse
            `} />
          </>
        )}
      </div>
    </div>
  );
};

export default AnimatedStatusIndicator;