import React, { useState, useRef, useEffect } from "react";
import SideBar from "../SideBar/SideBar";
import "./GenerateCanvas.css";

function GenerateCanvas() {
  
  const [selectedClass, setSelectedClass] = useState("");
  const [isGenerating, setIsGenerating] = useState(false);
  const [strokes, setStrokes] = useState([]);
  const canvasRef = useRef(null);

  const handleSelectCategory = async (label) => {
    setSelectedClass(label);
    setIsGenerating(true);
    setStrokes([]);

    try {
      const res = await fetch("http://localhost:5050/category", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ label }),
      });

      const data = await res.json();
      if (data.strokes) {
        setStrokes(data.strokes);
      }
    } catch (err) {
      console.error("Error notifying backend:", err);
    } finally {
      setIsGenerating(false);
    }
  };



  return (
    <div className="maincontainer ">
      <div className="generate-canvas">

        <SideBar
          selectedClass={selectedClass}
          onSelectCategory={handleSelectCategory}
          disabled={isGenerating}
        />

        <div className="drawing-area">
          <h2>
            {selectedClass ? (
              <>Model is drawing: <span className="highlight">{selectedClass}</span></>
            ) : (
              <>Select a category of your choice</>
            )}
          </h2>

          <div className="canvas-placeholder">
            <canvas width="400" height="300" id="generated-canvas"></canvas>
          </div>
        </div>
        
      </div>
    </div>
  );
}

export default GenerateCanvas;
