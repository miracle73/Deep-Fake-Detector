import "./App.css";
import { BrowserRouter as Router, Route, Routes } from "react-router-dom";
import Signin from "./pages/signin";
import SignUp from "./pages/signup";
import Dashboard from "./pages/dashboard";
import SubscriptionPlans from "./pages/subscriptionPlans";
import Notifications from "./pages/notifications";
import Settings from "./pages/settings";
import Billing from "./pages/billing";
import DeepfakeDetector from "./pages";
import VideoScreen from "./pages/video";
import ForgotPassword from "./pages/forgotPassword";
import ResetPassword from "./pages/resetPassword";
import VerifyEmail from "./pages/verifyEmail";
import CheckEmail from "./pages/checkEmail";
import TermsAndConditions from "./pages/termsAndCondition";

function App() {
  return (
    <>
      <Router>
        <Routes>
          <Route path="/" element={<DeepfakeDetector />} />
          <Route path="/signin" element={<Signin />} />
          <Route path="/signup" element={<SignUp />} />
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/video-detection" element={<VideoScreen />} />
          <Route path="/plans" element={<SubscriptionPlans />} />
          <Route path="/notifications" element={<Notifications />} />
          <Route path="/settings" element={<Settings />} />
          <Route path="/billing" element={<Billing />} />
          <Route path="/forgot-password" element={<ForgotPassword />} />
          <Route path="/reset-password" element={<ResetPassword />} />
          <Route path="/verify-email" element={<VerifyEmail />} />
          <Route path="/check-email" element={<CheckEmail />} />
          <Route
            path="/terms-and-conditions"
            element={<TermsAndConditions />}
          />
        </Routes>
      </Router>
    </>
  );
}

export default App;
