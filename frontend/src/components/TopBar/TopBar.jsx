import React from "react";
import { useNavigate } from "react-router-dom";
import "./TopBar.css";

function TopBar() {
  const navigate = useNavigate();

  return (
    <div className="topbar">
      <div className="title">
        <button className="title-button" onClick={() => navigate("/")}>
          SketchCoder
        </button>
      </div>
      <div className="nav-buttons">
        <button onClick={() => navigate("/sketchcanvas")}>Sketch</button>
        <button onClick={() => navigate("/generatecanvas")}>Generate</button>
      </div>
    </div>
  );
}

export default TopBar;