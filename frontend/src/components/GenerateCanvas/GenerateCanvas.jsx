import React, { useState, useRef } from "react";
import SideBar from "../SideBar/SideBar";
import GenerateSelectCategory from "../GenerateSelectCategory/GenerateSelectCategory";
import "./GenerateCanvas.css";

function GenerateCanvas() {
  const [selectedCategory, setSelectedCategory] = useState(""); // category user selected in modal
  const [selectedClass, setSelectedClass] = useState("");       // category sent to backend for drawing
  const [showCategoryModal, setShowCategoryModal] = useState(true);
  const [isGenerating, setIsGenerating] = useState(false);
  const [strokes, setStrokes] = useState([]);
  const canvasRef = useRef(null);

  const handleSelectCategory = async (label) => {
    setSelectedClass(label);
    setShowCategoryModal(false);
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

  const handleDownload = () => {
    const canvas = document.getElementById("generated-canvas");
    const link = document.createElement("a");
    link.download = `${selectedClass}.png`;
    link.href = canvas.toDataURL("image/png");
    link.click();
  };

  const handleRandomize = () => {
    const classes = [
      "clock", "door", "bat", "bicycle", "paintbrush",
      "cactus", "lightbulb", "smileyface", "bus", "guitar"
    ];
    const randomClass = classes[Math.floor(Math.random() * classes.length)];
    setSelectedCategory(randomClass);
    handleSelectCategory(randomClass);
  };

  const handleCreateAnother = () => {
    setSelectedClass("");
    setSelectedCategory("");
    setShowCategoryModal(true);
  };

  return (
    <div className="maincontainer">
      <div className="generate-canvas">
        {showCategoryModal && (
          <GenerateSelectCategory
            selectedClass={selectedCategory}
            onSelectCategory={setSelectedCategory}
            onCreate={() => handleSelectCategory(selectedCategory)}
            disabled={isGenerating}
          />
        )}

        <SideBar
          onDownload={handleDownload}
          onRandomize={handleRandomize}
          onCreateAnother={handleCreateAnother}
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
            <canvas id="generated-canvas" ref={canvasRef}></canvas>
          </div>
        </div>
      </div>
    </div>
  );
}

export default GenerateCanvas;
