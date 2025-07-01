// import { useEffect, useState } from "react";
import {
  Mail,
  ArrowLeft,
  //  RefreshCw,
  CheckCircle,
} from "lucide-react";
import { useNavigate, useLocation } from "react-router-dom";
import BackgroundImage from "../assets/images/check-email.png";

function CheckEmail() {
  const navigate = useNavigate();
  const location = useLocation();
  const email = location.state?.email;
  //   const [resendCooldown, setResendCooldown] = useState(0);

  // Redirect if no email provided
  //   useEffect(() => {
  //     if (!email) {
  //       navigate("/signup");
  //     }
  //   }, [email, navigate]);

  // Countdown for resend button
  //   useEffect(() => {
  //     let interval: NodeJS.Timeout;
  //     if (resendCooldown > 0) {
  //       interval = setInterval(() => {
  //         setResendCooldown((prev) => prev - 1);
  //       }, 1000);
  //     }
  //     return () => clearInterval(interval);
  //   }, [resendCooldown]);

  //   const handleResendEmail = () => {
  //     // TODO: Implement resend logic when you have the API endpoint
  //     setResendCooldown(60); // 60 second cooldown
  //     console.log("Resend email to:", email);
  //   };

  return (
    <div className="min-h-screen flex flex-col lg:flex-row">
      {/* Mobile header - only visible on mobile */}
      <div className="lg:hidden p-6 bg-white">
        <div className="text-center max-md:text-left">
          <h1 className="text-xl font-bold text-gray-900">
            <span className="font-bold">Safeguard</span>{" "}
            <span className="font-normal">Media</span>
          </h1>
        </div>
      </div>

      {/* Left side - Check email content */}
      <div className="flex-1  p-4 lg:p-8 bg-white">
        <div className="w-full flex justify-start max-lg:hidden">
          <button
            type="button"
            className="flex items-center text-sm text-gray-600 hover:text-gray-900 mb-6 self-start"
            onClick={() => navigate("/signup")}
          >
            <ArrowLeft className="w-4 h-4 mr-2" />
            Back to Sign Up
          </button>
        </div>
        <div className="flex items-center justify-center">
          <div className="w-full max-w-sm ">
            {/* Back button */}

            <div className="text-center">
              <h2 className="text-2xl font-semibold text-gray-900">
                Check Your Email
              </h2>
              <p className="text-gray-600 mt-2 text-sm">
                We've sent a verification link to{" "}
                <span className="font-medium">{email}</span>
              </p>
            </div>

            <div className="border border-gray-200 shadow-sm rounded-2xl mt-5">
              <div className="p-8 space-y-6">
                {/* Email icon */}
                <div className="flex justify-center">
                  <div className="bg-blue-100 p-4 rounded-full">
                    <Mail className="w-8 h-8 text-blue-600" />
                  </div>
                </div>

                {/* Instructions */}
                <div className="space-y-4 text-center">
                  <h3 className="text-lg font-semibold text-gray-900">
                    Verify your email address
                  </h3>

                  <div className="space-y-3 text-sm text-gray-600">
                    <div className="flex items-start space-x-3">
                      <CheckCircle className="w-4 h-4 text-green-500 mt-0.5 flex-shrink-0" />
                      <p className="text-left">
                        Check your email inbox for a verification link
                      </p>
                    </div>

                    <div className="flex items-start space-x-3">
                      <CheckCircle className="w-4 h-4 text-green-500 mt-0.5 flex-shrink-0" />
                      <p className="text-left">
                        Click the verification link to activate your account
                      </p>
                    </div>

                    <div className="flex items-start space-x-3">
                      <CheckCircle className="w-4 h-4 text-green-500 mt-0.5 flex-shrink-0" />
                      <p className="text-left">
                        Check your spam folder if you don't see the email
                      </p>
                    </div>
                  </div>
                </div>

                {/* Action buttons */}
                <div className="space-y-4">
                  {/* Resend email button */}
                  {/* <button
                  type="button"
                  disabled={resendCooldown > 0}
                  onClick={handleResendEmail}
                  className="w-full h-12 bg-[#0F2FA3] hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed text-white font-medium rounded-[50px] flex items-center justify-center"
                >
                  {resendCooldown > 0 ? (
                    `Resend in ${resendCooldown}s`
                  ) : (
                    <>
                      <RefreshCw className="w-4 h-4 mr-2" />
                      Resend Email
                    </>
                  )}
                </button> */}

                  {/* Sign In Link */}
                  <div className="text-center text-sm text-gray-600">
                    {"Already verified? "}
                    <button
                      type="button"
                      className="text-blue-600 hover:text-blue-500 hover:underline font-medium"
                      onClick={() => navigate("/signin")}
                    >
                      Sign In
                    </button>
                  </div>
                </div>

                {/* Help text */}
                <div className="text-center">
                  <p className="text-xs text-gray-500">
                    If you continue to have problems, please{" "}
                    <button
                      type="button"
                      className="text-blue-600 hover:text-blue-500 hover:underline"
                      onClick={() => {
                        // TODO: Add contact support functionality
                        console.log("Contact support");
                      }}
                    >
                      contact support
                    </button>
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Right side - Desktop only branding */}
      {/* Right side - Background image (desktop only) */}
      <div className="hidden lg:block flex-1 relative">
        <img
          src={BackgroundImage}
          alt="Person working on laptop"
          className="absolute inset-0 w-full h-full object-cover"
        />
        <div className="absolute top-8 right-8 ">
          <div className="text-black font-semibold text-xl">
            <span className="font-bold">Safeguard</span>{" "}
            <span className="font-normal">Media</span>
          </div>
        </div>
      </div>
    </div>
  );
}

export default CheckEmail;
