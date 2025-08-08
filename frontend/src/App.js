import React from "react";
import { Routes, Route } from "react-router-dom";
import TopBar from "./components/TopBar/TopBar";
import Home from "./components/Home/Home";
import SketchCanvas from "./components/Sketchcanvas/Sketchcanvas";
import GenerateCanvas from "./components/GenerateCanvas/GenerateCanvas";
import TransformersPage from "./components/Transformers/TransformersPage"; // NEW
import Footer from "./components/Footer/Footer";
import "./App.css";

function App() {
  return (
    <div className="app-container">
      <TopBar />
      <div className="main-content">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/sketchcanvas" element={<SketchCanvas />} />
          <Route path="/generatecanvas" element={<GenerateCanvas />} />
          <Route path="/generateTransformers" element={<TransformersPage />} /> {/* NEW */}
        </Routes>
      </div>
      <Footer />
    </div>
  );
}

export default App;
