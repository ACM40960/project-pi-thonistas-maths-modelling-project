import React, { useState, useRef } from "react";
import SideBar from "../SideBar/SideBar";
import GenerateSelectCategory from "../GenerateSelectCategory/GenerateSelectCategory";
import "./GenerateCanvas.css";

function GenerateCanvas() {
  const [selectedCategory, setSelectedCategory] = useState("");
  const [selectedClass, setSelectedClass] = useState("");
  const [showCategoryModal, setShowCategoryModal] = useState(true);
  const [isGenerating, setIsGenerating] = useState(false);
  const canvasRef = useRef(null);
  const cursorRef = useRef(null);

  const handleSelectCategory = async (label) => {
    setSelectedClass(label);
    setShowCategoryModal(false);
    setIsGenerating(true);

    try {
      const res = await fetch("http://localhost:5050/ndjson-strokes", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ label }),
      });

      const data = await res.json();

      if (data.drawing) {
        console.log("Stroke data received from backend:", data.drawing);
        drawStrokesAnimated(data.drawing);
      }
    } catch (err) {
      console.error("Error fetching strokes:", err);
    } finally {
      
    }
  };

  const drawStrokesAnimated = (strokes) => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.lineWidth = 2;
    ctx.strokeStyle = "black";

    // Normalize coordinates to canvas size
    const allPoints = strokes.flatMap(([xList, yList]) =>
      xList.map((x, i) => [x, yList[i]])
    );

    const xs = allPoints.map(([x]) => x);
    const ys = allPoints.map(([, y]) => y);

    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);

    const scaleX = canvas.width / (maxX - minX + 1e-6);
    const scaleY = canvas.height / (maxY - minY + 1e-6);
    const scale = Math.min(scaleX, scaleY);

    const normalize = (x, y) => [
      (x - minX) * scale,
      (y - minY) * scale
    ];

    let strokeIndex = 0;
    let pointIndex = 0;

    function drawNext() {
      if (strokeIndex >= strokes.length) {
        if (cursorRef.current) cursorRef.current.style.display = "none";
        setIsGenerating(false);
        return;
      }

      const stroke = strokes[strokeIndex];
      const xList = stroke[0];
      const yList = stroke[1];

      if (pointIndex >= xList.length - 1) {
        strokeIndex++;
        pointIndex = 0;
        requestAnimationFrame(drawNext);
        return;
      }

      const [x0n, y0n] = normalize(xList[pointIndex], yList[pointIndex]);
      const [x1n, y1n] = normalize(xList[pointIndex + 1], yList[pointIndex + 1]);

      // Draw stroke
      ctx.beginPath();
      ctx.moveTo(x0n, y0n);
      ctx.lineTo(x1n, y1n);
      ctx.stroke();

      // Move cursor image
      const cursor = cursorRef.current;
      if (cursor) {
        cursor.style.left = `${x1n}px`;
        cursor.style.top = `${y1n}px`;
        cursor.style.display = "block";
      }

      pointIndex++;
      setTimeout(drawNext, 50); 
    }

    drawNext();
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

          <div className="canvas-placeholder" style={{ position: "relative" }}>
            <canvas
              id="generated-canvas"
              ref={canvasRef}
              width={500}
              height={500}
            ></canvas>
            <img
              ref={cursorRef}
              src="/cursor2.cur"
              alt="cursor"
              style={{
                position: "absolute",
                width: "30px",
                height: "30px",
                display: "none",
                pointerEvents: "none",
                transform: "translate(-50%, -50%)"
              }}
            />
          </div>
        </div>
      </div>
    </div>
  );
}

export default GenerateCanvas;
