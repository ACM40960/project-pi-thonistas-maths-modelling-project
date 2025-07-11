// src/components/Sketchcanvas/Sketchcanvas.jsx
import React, { useRef, useState } from "react";
import { ReactSketchCanvas } from "react-sketch-canvas";
import "./Sketchcanvas.css";

function SketchCanvas() {
  const canvasRef = useRef(null);
  const [prediction, setPrediction] = useState("");

  const handleClear = () => {
    if (canvasRef.current) canvasRef.current.clearCanvas();
    setPrediction("");
  };

  const handleSubmit = async () => {
    try {
      const paths = await canvasRef.current.exportPaths();
      if (paths.length === 0) {
        alert("Canvas is empty! Please draw something.");
        return;
      }

      const base64Image = await canvasRef.current.exportImage("png");
      const response = await fetch("http://localhost:5050/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: base64Image }),
      });

      const data = await response.json();
      if (data.label) {
        const confidence = data.confidence
          ? `(${(data.confidence * 100).toFixed(2)}%)`
          : "";
        setPrediction(`${data.label} ${confidence}`);
      } else if (data.error) {
        setPrediction(`Error: ${data.error}`);
      } else {
        setPrediction("No prediction received.");
      }
    } catch (error) {
      console.error("Error submitting drawing:", error);
      setPrediction("Error: Unable to connect to server.");
    }
  };

  return (
    <div className="maincontainer"> 
      <div className="sketch-container">
        <ReactSketchCanvas
          ref={canvasRef}
          width="500px"
          height="400px"
          strokeWidth={4}
          strokeColor="black"
          style={{
            border: "2px dashed #c8a2c8",
            borderRadius: "8px",
            marginBottom: "16px",
          }}
        />
        <div className="controls">
          <button onClick={handleClear}>Clear</button>
          <button onClick={handleSubmit}>Submit</button>
        </div>
        {prediction && (
          <div className="prediction">
            Prediction: <strong>{prediction}</strong>
          </div>
        )}
      </div>
    </div>
  );
}

export default SketchCanvas;
