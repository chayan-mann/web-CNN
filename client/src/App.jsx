import { BrowserRouter as Router, Route, Routes, Link } from "react-router-dom";
import Scan from "./pages/Scan";
import About from "./pages/About";
import LandingPage from "./pages/LandingPage";

function App() {
  return (
    <Router>
      <div className="min-h-screen bg-gradient-to-b from-gray-900 to-black text-white">
        <Navbar />
        <Routes>
          <Route path="/" element={<LandingPage />} /> 
          <Route path="/scan" element={<Scan />} />
          <Route path="/about" element={<About />} />
        </Routes>
      </div>
    </Router>
  );
}

function Navbar() {
  return (
    <nav className="fixed top-0 left-0 w-full bg-gray-800 shadow-lg z-50">
      <div className="flex items-center justify-between px-10 py-4">
      <Link to="/" className="text-2xl font-bold text-blue-400">
          Brain Tumor Detection
        </Link>
        <ul className="flex w-full justify-evenly">
          <li>
            <Link
              to="/"
              className="text-lg font-medium hover:text-blue-300 transition"
            >
              Home
            </Link>
          </li>
          <li>
            <Link
              to="/scan"
              className="text-lg font-medium hover:text-blue-300 transition"
            >
              Scan
            </Link>
          </li>
          <li>
            <Link
              to="/about"
              className="text-lg font-medium hover:text-blue-300 transition"
            >
              About
            </Link>
          </li>
          
          <li>
            <Link
              to="/contact"
              className="text-lg font-medium hover:text-blue-300 transition"
            >
              Contact
            </Link>
          </li>
        </ul>
      </div>
    </nav>
  );
}


export default App;
