import React from "react";
import "./GenerateSelectCategory.css";

function GenerateSelectCategory({ selectedClass, onSelectCategory, onCreate, disabled }) {
  const categories = [
    "clock", "door", "bat", "bicycle", "paintbrush",
    "cactus", "lightbulb", "smileyface", "bus", "guitar"
  ];

  return (
    <div className="overlay">
      <div className="category-modal">
        <h3 className="category-heading">Choose the category you want in for doodle:</h3>
        <div className="category-grid">
          {categories.map((label, index) => (
            <button
              key={index}
              className={`category-button ${selectedClass === label ? "selected" : ""}`}
              onClick={() => onSelectCategory(label)}
              disabled={disabled}
            >
              {label}
            </button>
          ))}
        </div>
        <button
          className="create-button"
          onClick={onCreate}
          disabled={!selectedClass || disabled}
        >
          Create
        </button>
      </div>
    </div>
  );
}

export default GenerateSelectCategory;
