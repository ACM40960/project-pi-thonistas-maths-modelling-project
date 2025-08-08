import React, { useRef, useState } from "react";
import "./Transformers.css";

const CATEGORIES = [
  "clock","door","bat","bicycle","paintbrush",
  "cactus","lightbulb","smileyface","bus","guitar"
];

export default function TransformersPage() {
  const [label, setLabel] = useState("");
  const [busy, setBusy] = useState(false);
  const canvasRef = useRef(null);
  const cursorRef = useRef(null);

  const drawStrokesAnimated = (strokes) => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.lineWidth = 2;
    ctx.strokeStyle = "black";

    // Flatten and normalize to canvas
    const points = strokes.flatMap(([xs, ys]) => xs.map((x, i) => [x, ys[i]]));
    const xs = points.map(p => p[0]);
    const ys = points.map(p => p[1]);
    const minX = Math.min(...xs), maxX = Math.max(...xs);
    const minY = Math.min(...ys), maxY = Math.max(...ys);
    const sx = canvas.width / (maxX - minX + 1e-6);
    const sy = canvas.height / (maxY - minY + 1e-6);
    const s = Math.min(sx, sy);
    const norm = (x, y) => [(x - minX) * s, (y - minY) * s];

    let si = 0, pi = 0;

    function step() {
      if (si >= strokes.length) {
        if (cursorRef.current) cursorRef.current.style.display = "none";
        setBusy(false);
        return;
      }

      const [xList, yList] = strokes[si];
      if (pi >= xList.length - 1) {
        si++;
        pi = 0;
        requestAnimationFrame(step);
        return;
      }

      const [x0, y0] = norm(xList[pi], yList[pi]);
      const [x1, y1] = norm(xList[pi + 1], yList[pi + 1]);

      ctx.beginPath();
      ctx.moveTo(x0, y0);
      ctx.lineTo(x1, y1);
      ctx.stroke();

      const c = cursorRef.current;
      if (c) {
        c.style.left = `${x1}px`;
        c.style.top = `${y1}px`;
        c.style.display = "block";
      }

      pi++;
      setTimeout(step, 50);
    }

    step();
  };

  const handleCreate = async () => {
    if (!label) return;
    setBusy(true);
    try {
      const res = await fetch("http://localhost:5050/transformer-strokes", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ label })
      });
      const data = await res.json();
      if (data.drawing) {
        drawStrokesAnimated(data.drawing);
      } else {
        setBusy(false);
        console.error("No drawing returned:", data);
      }
    } catch (e) {
      console.error(e);
      setBusy(false);
    }
  };

  const handleDownload = () => {
    const a = document.createElement("a");
    a.download = `${label || "transformer"}.png`;
    a.href = canvasRef.current.toDataURL("image/png");
    a.click();
  };

  return (
    <div className="transformers-home-container">
      <div className="transformers-wrap">
        <h2 className="transformers-title">
          Transformers Generator
        </h2>

        <div className="transformers-controls">
          <select
            className="transformers-select"
            value={label}
            onChange={(e) => setLabel(e.target.value)}
            disabled={busy}
          >
            <option value="">Select category…</option>
            {CATEGORIES.map(c => (
              <option key={c} value={c}>{c}</option>
            ))}
          </select>

          <button
            className="t-btn"
            onClick={handleCreate}
            disabled={!label || busy}
          >
            Create
          </button>

          <button
            className="t-btn white"
            onClick={handleDownload}
            disabled={busy}
          >
            Download
          </button>
        </div>

        <div className="transformers-canvas-area">
          <canvas id="transformers-canvas" ref={canvasRef} width={500} height={500} />
          <img
            ref={cursorRef}
            src="/cursor2.cur"
            alt=""
            className="cursor-img"
          />
        </div>
      </div>
    </div>
  );
}
