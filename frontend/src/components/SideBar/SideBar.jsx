// src/components/SideBar/SideBar.jsx
import React from "react";
import "./SideBar.css";

function SideBar({ selectedClass, onSelectCategory, disabled }) {
  const classes = [
    "clock", "door", "bat", "bicycle", "paintbrush",
    "cactus", "lightbulb", "smileyface", "bus", "guitar"
  ];

  return (
    <div className="sidebar">
      {classes.map((item, index) => (
        <button
          key={index}
          className={`class-button ${
            disabled && selectedClass === item ? "disabled" : ""
          }`}
          onClick={() => onSelectCategory(item)}
          disabled={disabled && selectedClass === item}
        >
          {item}
        </button>
      ))}
    </div>
  );
}

export default SideBar;
