import "./App.css";
import { BrowserRouter as Router, Route, Routes } from "react-router-dom";
import Home from "./pages/home";
import SecondPage from "./pages";
import Signin from "./pages/signin";
import SignUp from "./pages/signup";
import Dashboard from "./pages/dashboard";
import VideoScreen from "./pages/video";
import VideoScreen2 from "./pages/video2";

function App() {
  return (
    <>
      <Router>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/index" element={<SecondPage />} />
          <Route path="/signin" element={<Signin />} />
          <Route path="/signup" element={<SignUp />} />
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/video-detection" element={<VideoScreen />} />
          <Route path="/video-detection2" element={<VideoScreen2 />} />
        </Routes>
      </Router>
    </>
  );
}

export default App;
