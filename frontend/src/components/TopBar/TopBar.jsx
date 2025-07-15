import React from "react";
import { useNavigate } from "react-router-dom";
import "./TopBar.css";
import { FaPaintBrush } from "react-icons/fa";
import { RiAiGenerate2 } from "react-icons/ri";

function TopBar() {
  const navigate = useNavigate();

  return (
    <div className="topbar">
      <div className="title">
        <button className="title-button boldonse" onClick={() => navigate("/")}>
          SKETCHBOT
        </button>
      </div>
      <div className="nav-buttons">
        <button onClick={() => navigate("/sketchcanvas")}> <FaPaintBrush style={{ marginRight: "6px" }} /> Sketch</button>
        <button onClick={() => navigate("/generatecanvas")}> <RiAiGenerate2 size={20} style={{ marginRight: "6px" }} /> Generate</button>
      </div>
    </div>
  );
}

export default TopBar;