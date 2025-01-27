import { useState } from "react";
import axios from "axios";

function Scan() {
  const [image, setImage] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState("");

  // Handle image upload
  const handleImageUpload = (e) => {
    const file = e.target.files[0];
    if (file && file.type.startsWith("image/")) {
      setImage(file);
      setError("");
    } else {
      setError("Please upload a valid image file.");
      setImage(null);
    }
  };

  // Handle form submission (send image to the backend)
  const handleSubmit = async () => {
    if (!image) {
      setError("Please upload an image before submitting.");
      return;
    }

    setLoading(true);
    setError("");
    setResult("");

    const formData = new FormData();
    formData.append("file", image); // Ensure the field name matches the FastAPI backend (file)

    try {
      const response = await axios.post("http://localhost:8000/predict/", formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      });
      setResult(response.data.prediction);  // Assuming the backend returns a field 'prediction'
    } catch (err) {
      setError("Failed to get a prediction. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen">
      <div className="bg-gray-800 shadow-lg rounded-lg p-8 w-full max-w-lg">
        <h1 className="text-3xl font-bold text-center mb-6 text-blue-400">
          Brain Tumor Detection
        </h1>
        <p className="text-sm text-center mb-4 text-gray-400">
          Upload an MRI scan to check if it indicates a brain tumor.
        </p>
        <div className="flex flex-col items-center">
          <input
            type="file"
            accept="image/*"
            onChange={handleImageUpload}
            className="mb-4 w-full px-3 py-2 text-gray-700 border border-gray-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
          {error && (
            <p className="text-sm text-red-500 text-center mb-4">{error}</p>
          )}

          <button
            onClick={handleSubmit}
            className={`w-full bg-blue-500 text-white py-3 px-6 rounded-lg text-lg font-semibold transition-transform ${
              loading ? "opacity-50 cursor-not-allowed" : "hover:scale-105"
            }`}
            disabled={loading}
          >
            {loading ? "Analyzing..." : "Submit"}
          </button>
        </div>

        {result && (
          <div className="mt-6 bg-green-100 text-green-800 p-4 rounded-lg shadow">
            <p className="font-semibold text-lg">Prediction Result:</p>
            <p className="text-xl font-bold">{result}</p>
          </div>
        )}
      </div>
    </div>
  );
}

export default Scan;
