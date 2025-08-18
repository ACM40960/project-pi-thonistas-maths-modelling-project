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

  const drawBase64ToCanvas = (dataUrl) => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");

    const img = new Image();
    img.onload = () => {
      // Clear canvas
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // Fit image into canvas (contain) and center it
      const cw = canvas.width;
      const ch = canvas.height;
      const iw = img.width;
      const ih = img.height;

      const scale = Math.min(cw / iw, ch / ih);
      const drawW = iw * scale;
      const drawH = ih * scale;
      const dx = (cw - drawW) / 2;
      const dy = (ch - drawH) / 2;

      ctx.drawImage(img, dx, dy, drawW, drawH);
      setIsGenerating(false);
    };
    img.onerror = () => {
      console.error("Failed to load generated image");
      setIsGenerating(false);
    };
    img.src = dataUrl;
  };

  const handleSelectCategory = async (label) => {
    if (!label) return;
    setSelectedClass(label);
    setShowCategoryModal(false);
    setIsGenerating(true);

    try {
      const res = await fetch("http://localhost:5050/category", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ label }),
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.error || `HTTP ${res.status}`);
      }

      const data = await res.json();
      let dataUrl = data.image;

      // Safety: ensure data URL prefix exists
      if (dataUrl && !dataUrl.startsWith("data:image")) {
        dataUrl = `data:image/png;base64,${dataUrl}`;
      }

      if (dataUrl) {
        drawBase64ToCanvas(dataUrl);
      } else {
        throw new Error("No image returned from /category");
      }
    } catch (err) {
      console.error("Error fetching generated image:", err);
      setIsGenerating(false);
    }
  };

  const handleDownload = () => {
    const canvas = document.getElementById("generated-canvas");
    const link = document.createElement("a");
    link.download = `${selectedClass || "sketch"}.png`;
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
              <>Model generated: <span className="highlight">{selectedClass}</span></>
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
            {/* No cursor / animation */}
          </div>
        </div>
      </div>
    </div>
  );
}

export default GenerateCanvas;
