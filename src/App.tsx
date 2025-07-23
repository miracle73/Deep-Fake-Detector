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
import VerifyEmail from "./pages/verifyEmail";
import CheckEmail from "./pages/checkEmail";
// import TermsAndConditions from "./pages/termsAndCondition";
import ForgotPassword from "./pages/forgotPassword";
import ResetPassword from "./pages/resetPassword";
import ImageScreen from "./pages/image";
import AudioScreen from "./pages/audio";
import { useSelector } from "react-redux";
import type { RootState } from "./store/store";
import { Navigate } from "react-router-dom";
import TermsOfService from "./pages/termsAndCondition";
import PrivacyPolicy from "./pages/privacypolicy";

interface ProtectedRouteProps {
  children: React.ReactNode;
}

const ProtectedRoute = ({ children }: ProtectedRouteProps) => {
  const isAuthenticated = useSelector(
    (state: RootState) => state.auth.isAuthenticated
  );

  return isAuthenticated ? children : <Navigate to="/signin" replace />;
};

function App() {
  return (
    <>
      <Router>
        <Routes>
          {/* Public Routes */}
          <Route path="/signin" element={<Signin />} />
          <Route path="/signup" element={<SignUp />} />
          <Route path="/verify-email" element={<VerifyEmail />} />
          <Route path="/check-email" element={<CheckEmail />} />
          <Route path="/terms-and-conditions" element={<TermsOfService />} />
          <Route path="/privacy-policy" element={<PrivacyPolicy />} />
          <Route path="/forgot-password" element={<ForgotPassword />} />
          <Route path="/reset-password/:token" element={<ResetPassword />} />

          <Route path="/" element={<DeepfakeDetector />} />

          {/* Protected Routes */}
          <Route
            path="/dashboard"
            element={
              <ProtectedRoute>
                <Dashboard />
              </ProtectedRoute>
            }
          />
          <Route
            path="/video-detection"
            element={
              <ProtectedRoute>
                <VideoScreen />
              </ProtectedRoute>
            }
          />
          <Route
            path="/plans"
            element={
              <ProtectedRoute>
                <SubscriptionPlans />
              </ProtectedRoute>
            }
          />
          <Route
            path="/notifications"
            element={
              <ProtectedRoute>
                <Notifications />
              </ProtectedRoute>
            }
          />
          <Route
            path="/settings"
            element={
              <ProtectedRoute>
                <Settings />
              </ProtectedRoute>
            }
          />
          <Route
            path="/billing"
            element={
              <ProtectedRoute>
                <Billing />
              </ProtectedRoute>
            }
          />
          <Route
            path="/image-detection"
            element={
              <ProtectedRoute>
                <ImageScreen />
              </ProtectedRoute>
            }
          />
          <Route
            path="/audio-detection"
            element={
              <ProtectedRoute>
                <AudioScreen />
              </ProtectedRoute>
            }
          />
        </Routes>
      </Router>
    </>
  );
}

export default App;
