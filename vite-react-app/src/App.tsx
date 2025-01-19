import React, { useState } from "react";
import axios from "axios";
import "./App.css"; // Import the CSS file for styling

function App() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [result, setResult] = useState<string>("");
  const [imagePreview, setImagePreview] = useState<string | null>(null); // New state for image preview

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (files && files[0]) {
      setSelectedFile(files[0]);

      // Generate a preview of the image
      const fileURL = URL.createObjectURL(files[0]);
      setImagePreview(fileURL);
    }
  };

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    if (!selectedFile) {
      alert("Please select an image file.");
      return;
    }

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const response = await axios.post(
        "http://localhost:8000/classify/",
        formData,
        {
          headers: { "Content-Type": "multipart/form-data" },
        }
      );
      console.log(response.data);
      setResult(response.data.best_prediction);
    } catch (error) {
      console.error("Error uploading file:", error);
      setResult("An error occurred. Please try again.");
    }
  };

  return (
    <div className="app-container">
      <h1 className="app-title">Font Classification</h1>
      <p className="app-description">
        Upload an image and get the font classification result!
      </p>

      <form className="upload-form" onSubmit={handleSubmit}>
        <input
          type="file"
          accept="image/*"
          onChange={handleFileChange}
          className="file-input"
        />
        <button type="submit" className="submit-button">
          Upload and Classify
        </button>
      </form>

      {/* Show image preview if file is selected */}
      {imagePreview && (
        <div className="image-preview-container">
          <h3>Preview:</h3>
          <img src={imagePreview} alt="Preview" className="image-preview" />
        </div>
      )}

      {result && (
        <div className="result">
          <h2>Result:</h2>
          <p className="result-text">{result}</p>
        </div>
      )}
    </div>
  );
}

export default App;
