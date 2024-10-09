import React from 'react';
import PoseEstimation from './components/PoseEstimation';

function App() {
  return (
    <div className="min-h-screen bg-gray-100 flex items-center justify-center">
      <div className="w-full max-w-3xl">
        <h1 className="text-3xl font-bold text-center text-gray-800 mb-8">Pose Estimation Demo</h1>
        <PoseEstimation />
      </div>
    </div>
  );
}

export default App;
