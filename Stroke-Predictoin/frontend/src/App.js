import React, { useState } from "react";
import "./App.css"; // Import the CSS file

function App() {
  const [formData, setFormData] = useState({
    gender: "",
    age: "",
    hypertension: "",
    heart_disease: "",
    ever_married: "",
    work_type: "",
    Residence_type: "",
    avg_glucose_level: "",
    bmi: "",
    smoking_status: "",
  });

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    const response = await fetch("http://localhost:5000/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(formData),
    });

    const result = await response.json();
    console.log("Response:", result);

    alert(`Predicted Stroke: ${result.stroke}`);
  };

  return (
    <div className="app-container">
      <h1 className="title">
        <span>Stroke Prediction</span>
      </h1>

      <span className="run run-left" style={{ animationDelay: "0s" }}></span>
      <span className="run run-right" style={{ animationDelay: "0.2s" }}></span>
      <span className="run run-left" style={{ animationDelay: "0.4s" }}></span>
      <span className="run run-right" style={{ animationDelay: "0.6s" }}></span>
      <span className="run run-left" style={{ animationDelay: "0.8s" }}></span>
      <span className="run run-right" style={{ animationDelay: "1s" }}></span>
      <span className="run run-left" style={{ animationDelay: "1.2s" }}></span>
      <span className="run run-right" style={{ animationDelay: "1.4s" }}></span>
      <span className="run run-left" style={{ animationDelay: "1.6s" }}></span>
      <span className="run run-right" style={{ animationDelay: "1.8s" }}></span>
      <span className="run run-left" style={{ animationDelay: "2s" }}></span>
      <span className="run run-right" style={{ animationDelay: "2.2s" }}></span>
      <span className="run run-left" style={{ animationDelay: "2.4s" }}></span>
      <span className="run run-right" style={{ animationDelay: "2.6s" }}></span>
      <span className="run run-left" style={{ animationDelay: "2.8s" }}></span>
      <span className="run run-right" style={{ animationDelay: "3s" }}></span>
      <span className="run run-left" style={{ animationDelay: "3.2s" }}></span>
      <span className="run run-right" style={{ animationDelay: "3.4s" }}></span>
      <span className="run run-left" style={{ animationDelay: "3.6s" }}></span>
      <span className="run run-right" style={{ animationDelay: "3.8s" }}></span>
      <span className="run run-left" style={{ animationDelay: "4s" }}></span>
      <span className="run run-right" style={{ animationDelay: "4.2s" }}></span>

      <div className="form-container">
      <form onSubmit={handleSubmit}>
        {/* Gender */}
        <div className="form-group">
          <label>
            Gender:
            <select name="gender" onChange={handleChange} required>
              <option value="">Select Gender</option>
              <option value="Male">Male</option>
              <option value="Female">Female</option>
              <option value="Other">Other</option>
            </select>
          </label>
        </div>

        {/* Age */}
        <div className="form-group">
          <label>
            Age:
            <input
              type="number"
              name="age"
              onChange={handleChange}
              placeholder="Age"
              required
            />
          </label>
        </div>

        {/* Hypertension */}
        <div className="form-group">
          <label>
            Hypertension:
            <select name="hypertension" onChange={handleChange} required>
              <option value="">Select</option>
              <option value="0">No</option>
              <option value="1">Yes</option>
            </select>
          </label>
        </div>

        {/* Heart Disease */}
        <div className="form-group">
          <label>
            Heart Disease:
            <select name="heart_disease" onChange={handleChange} required>
              <option value="">Select</option>
              <option value="0">No</option>
              <option value="1">Yes</option>
            </select>
          </label>
        </div>

        {/* Ever Married */}
        <div className="form-group">
          <label>
            Ever Married:
            <select name="ever_married" onChange={handleChange} required>
              <option value="">Select</option>
              <option value="No">No</option>
              <option value="Yes">Yes</option>
            </select>
          </label>
        </div>

        {/* Work Type */}
        <div className="form-group">
          <label>
            Work Type:
            <select name="work_type" onChange={handleChange} required>
              <option value="">Select</option>
              <option value="Private">Private</option>
              <option value="Self-employed">Self-employed</option>
              <option value="Govt_job">Govt_job</option>
              <option value="Children">Children</option>
              <option value="Never_worked">Never worked</option>
            </select>
          </label>
        </div>

        {/* Residence Type */}
        <div className="form-group">
          <label>
            Residence Type:
            <select name="Residence_type" onChange={handleChange} required>
              <option value="">Select</option>
              <option value="Urban">Urban</option>
              <option value="Rural">Rural</option>
            </select>
          </label>
        </div>

        {/* Average Glucose Level */}
        <div className="form-group">
          <label>
            Avg Glucose Level:
            <input
              type="number"
              name="avg_glucose_level"
              onChange={handleChange}
              placeholder="Average Glucose Level"
              step="0.01"
              required
            />
          </label>
        </div>

        {/* BMI */}
        <div className="form-group">
          <label>
            BMI:
            <input
              type="number"
              name="bmi"
              onChange={handleChange}
              placeholder="BMI"
              step="0.1"
              required
            />
          </label>
        </div>

        {/* Smoking Status */}
        <div className="form-group">
          <label>
            Smoking Status:
            <select name="smoking_status" onChange={handleChange} required>
              <option value="">Select</option>
              <option value="formerly smoked">Formerly Smoked</option>
              <option value="never smoked">Never Smoked</option>
              <option value="smokes">Smokes</option>
            </select>
          </label>
        </div>

        {/* Submit Button */}
        <button type="submit">Predict Stroke</button>
      </form>
      </div>

    </div>
  );
}

export default App;
