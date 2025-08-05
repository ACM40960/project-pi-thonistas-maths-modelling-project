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
      const res = await fetch("http://localhost:5050/generate-strokes", {
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

    // Convert relative to absolute
    const absStrokes = [];
    let x = 0, y = 0;

    for (const [dx, dy, pen] of strokes) {
      x += dx;
      y += dy;

      // Convert scalar pen to old one-hot format
      let p1 = 0, p2 = 0, p3 = 0;
      if (pen === 0) p1 = 1;   // pen down
      else if (pen === 1) p2 = 1; // pen lift
      else if (pen === 2) p3 = 1; // end

      absStrokes.push([x, y, p1, p2, p3]);
    }

    // Find bounds for normalization
    const xs = absStrokes.map(pt => pt[0]);
    const ys = absStrokes.map(pt => pt[1]);

    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);

    const scaleX = canvas.width / (maxX - minX + 1e-6);
    const scaleY = canvas.height / (maxY - minY + 1e-6);
    const scale = Math.min(scaleX, scaleY);

    const normalize = (x, y) => [
      (x - minX) * scale,
      (y - minY) * scale,
    ];

    let prev = null;
    let i = 0;

    function drawNext() {
      if (i >= absStrokes.length) {
        if (cursorRef.current) cursorRef.current.style.display = "none";
        setIsGenerating(false);
        return;
      }

      const [x, y, p1, p2, p3] = absStrokes[i];
      const [nx, ny] = normalize(x, y);

      if (prev && prev[2] === 1) {
        ctx.beginPath();
        ctx.moveTo(prev[0], prev[1]);
        ctx.lineTo(nx, ny);
        ctx.stroke();
      }

      const cursor = cursorRef.current;
      if (cursor) {
        cursor.style.left = `${nx}px`;
        cursor.style.top = `${ny}px`;
        cursor.style.display = "block";
      }

      prev = [nx, ny, p1];
      i++;
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
              <>
                Model is drawing:{" "}
                <span className="highlight">
                  {selectedClass.charAt(0).toUpperCase() + selectedClass.slice(1)}
                </span>
              </>
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
