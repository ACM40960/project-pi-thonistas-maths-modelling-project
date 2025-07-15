import React from "react";
import "./SketchResults.css";

function SketchResults({ prediction, onReset }) {
  return (
    <div className="results-overlay">
      <div className="results-card">
        <h2 className="results-title">✨ Great job!</h2>
        <p className="results-prompt">You drew:</p>
        <h1 className="results-prediction">{prediction}</h1>
        <button className="results-button" onClick={onReset}>
          🎨 Draw Again
        </button>
      </div>
    </div>
  );
}

export default SketchResults;
