import React from "react";
import { useNavigate } from "react-router-dom";
import "./TopBar.css";
import { FaPaintBrush } from "react-icons/fa";
import { HiOutlineSparkles } from "react-icons/hi";
import { FaRobot } from "react-icons/fa";


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
        <button onClick={() => navigate("/sketchcanvas")}>
          <FaPaintBrush style={{ marginRight: "6px" }} /> Sketch
        </button>
        <button onClick={() => navigate("/generatecanvas")}>
          <HiOutlineSparkles size={20} style={{ marginRight: "6px" }} /> Generate
        </button>
        <button onClick={() => navigate("/generateTransformers")}
        >
          <FaRobot size={20} style={{ marginRight: "6px" }} />Transformers
        </button>
      </div>
    </div>
  );
}

export default TopBar;
