import React from "react";
import "./SideBar.css";
import { FaDownload } from "react-icons/fa";
import { FaRandom } from "react-icons/fa";
import { FaPalette } from "react-icons/fa";

function SideBar({ onDownload, onRandomize, onCreateAnother, disabled }) {
  return (
    <div className="sidebar">
      <button className="action-button" onClick={onDownload} disabled={disabled}>
        <FaDownload style={{ marginRight: "6px" }} /> Download
      </button>
      <button className="action-button" onClick={onRandomize} disabled={disabled}>
        <FaRandom style={{ marginRight: "6px" }}/>Randomize
      </button>
      <button className="action-button" onClick={onCreateAnother} disabled={disabled}>
        <FaPalette style={{ marginRight: "6px" }}/>  Reset
      </button>
    </div>
  );
}

export default SideBar;
