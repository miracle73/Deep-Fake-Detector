import "./App.css";
import { BrowserRouter as Router, Route, Routes } from "react-router-dom";
import Home from "./pages/home";
import SecondPage from "./pages";
function App() {
  return (
    <>
      <Router>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/index" element={<SecondPage />} />
        </Routes>
      </Router>
    </>
  );
}

export default App;
