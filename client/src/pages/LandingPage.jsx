import { Link } from 'react-router-dom';

function LandingPage() {
  return (
    <div className="min-h-screen bg-black flex flex-col justify-center items-center text-white px-6">
      <div className="text-center space-y-6">
        <h1 className="text-4xl font-bold text-blue-400">Brain Tumor Detection</h1>
        <p className="text-lg text-gray-300">
          Welcome to the Brain Tumor Detection application. 
          Use our tool to upload an image and check if it indicates the presence of a brain tumor.
        </p>
        <Link to="/Scan">
          <button className="mt-8 px-6 py-3 bg-blue-600 text-white font-bold rounded-full hover:bg-blue-500 transition duration-300">
            Get Started
          </button>
        </Link>
      </div>
    </div>
  );
}

export default LandingPage;
