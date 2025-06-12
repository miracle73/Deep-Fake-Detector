import "./App.css";
import { BrowserRouter as Router, Route, Routes } from "react-router-dom";
import Home from "./pages/home";
import SecondPage from "./pages";
import Signin from "./pages/signin";
import SignUp from "./pages/signup";
function App() {
  return (
    <>
      <Router>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/index" element={<SecondPage />} />
          <Route path="/signin" element={<Signin />} />
          <Route path="/signup" element={<SignUp />} />
        </Routes>
      </Router>
    </>
  );
}

export default App;
