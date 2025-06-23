import { useEffect } from "react";
import { CheckCircle, AlertCircle, Loader } from "lucide-react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { useVerifyEmailQuery } from "../services/apiService";

function VerifyEmail() {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const token = searchParams.get("token");

  // Execute verification query immediately if token exists
  const {
    data: verificationResult,
    error: verificationError,
    isLoading,
  } = useVerifyEmailQuery(token ?? "", {
    skip: !token,
  });

  // Redirect to signup if no token
  useEffect(() => {
    if (!token) {
      navigate("/signup");
    }
  }, [token, navigate]);

  // Show loading state
  if (isLoading) {
    return (
      <div className="min-h-screen flex flex-col lg:flex-row">
        {/* Mobile header */}
        <div className="lg:hidden p-6 bg-white">
          <div className="text-center max-md:text-left">
            <h1 className="text-xl font-bold text-gray-900">
              <span className="font-bold">Safeguard</span>{" "}
              <span className="font-normal">Media</span>
            </h1>
          </div>
        </div>

        {/* Loading message */}
        <div className="flex-1 flex items-center justify-center p-4 lg:p-8 bg-white">
          <div className="w-full max-w-md">
            <div className="border border-gray-200 shadow-sm rounded-2xl">
              <div className="p-8 space-y-6 text-center">
                <div className="flex justify-center mb-4">
                  <div className="bg-blue-100 p-4 rounded-full">
                    <Loader className="w-8 h-8 text-blue-600 animate-spin" />
                  </div>
                </div>

                <div className="space-y-3">
                  <h3 className="text-lg font-semibold text-gray-900">
                    Verifying Your Email
                  </h3>
                  <p className="text-gray-600">
                    Please wait while we verify your email address...
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Right side - Desktop branding */}
        <div className="hidden lg:block flex-1 relative bg-gradient-to-br from-blue-50 to-cyan-100">
          <div className="absolute top-8 right-8">
            <div className="text-black font-semibold text-xl">
              <span className="font-bold">Safeguard</span>{" "}
              <span className="font-normal">Media</span>
            </div>
          </div>
          <div className="flex items-center justify-center h-full">
            <div className="text-center p-8">
              <div className="bg-white p-6 rounded-full w-24 h-24 flex items-center justify-center mx-auto mb-6">
                <Loader className="w-12 h-12 text-blue-600 animate-spin" />
              </div>
              <h3 className="text-2xl font-bold text-gray-900 mb-4">
                Verifying...
              </h3>
              <p className="text-gray-600 max-w-sm">
                We're confirming your email verification. This should only take
                a moment.
              </p>
            </div>
          </div>
        </div>
      </div>
    );
  }

  // Show success state
  if (verificationResult?.success) {
    return (
      <div className="min-h-screen flex flex-col lg:flex-row">
        {/* Mobile header */}
        <div className="lg:hidden p-6 bg-white">
          <div className="text-center max-md:text-left">
            <h1 className="text-xl font-bold text-gray-900">
              <span className="font-bold">Safeguard</span>{" "}
              <span className="font-normal">Media</span>
            </h1>
          </div>
        </div>

        {/* Success message */}
        <div className="flex-1 flex items-center justify-center p-4 lg:p-8 bg-white">
          <div className="w-full max-w-md">
            <div className="text-center">
              <h2 className="text-2xl font-semibold text-gray-900">
                Email Verified!
              </h2>
            </div>
            <div className="border border-gray-200 shadow-sm rounded-2xl mt-5">
              <div className="p-8 space-y-6 text-center">
                <div className="flex justify-center mb-4">
                  <div className="bg-green-100 p-4 rounded-full">
                    <CheckCircle className="w-8 h-8 text-green-600" />
                  </div>
                </div>

                <div className="space-y-3">
                  <h3 className="text-lg font-semibold text-gray-900">
                    Welcome to Safeguard Media!
                  </h3>
                  <p className="text-gray-600">
                    {verificationResult.message ||
                      "Your email has been successfully verified. You can now access all features of your account."}
                  </p>
                </div>

                <button
                  type="button"
                  className="w-full h-12 bg-[#0F2FA3] hover:bg-blue-700 text-white font-medium rounded-[50px]"
                  onClick={() => navigate("/signin")}
                >
                  Continue to Sign In
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* Right side - Desktop branding */}
        <div className="hidden lg:block flex-1 relative bg-gradient-to-br from-green-50 to-emerald-100">
          <div className="absolute top-8 right-8">
            <div className="text-black font-semibold text-xl">
              <span className="font-bold">Safeguard</span>{" "}
              <span className="font-normal">Media</span>
            </div>
          </div>
          <div className="flex items-center justify-center h-full">
            <div className="text-center p-8">
              <div className="bg-white p-6 rounded-full w-24 h-24 flex items-center justify-center mx-auto mb-6">
                <CheckCircle className="w-12 h-12 text-green-600" />
              </div>
              <h3 className="text-2xl font-bold text-gray-900 mb-4">
                Account Activated!
              </h3>
              <p className="text-gray-600 max-w-sm">
                Your email has been verified and your account is now fully
                activated. Welcome to the Safeguard Media family!
              </p>
            </div>
          </div>
        </div>
      </div>
    );
  }

  // Show error state
  if (verificationError) {
    const errorMessage = (() => {
      if (
        verificationError &&
        typeof verificationError === "object" &&
        "data" in verificationError
      ) {
        const apiError = verificationError as { data?: { message?: string } };
        return (
          apiError.data?.message ||
          "Verification failed. The link may be invalid or expired."
        );
      }
      return "Verification failed. The link may be invalid or expired.";
    })();

    return (
      <div className="min-h-screen flex flex-col lg:flex-row">
        {/* Mobile header */}
        <div className="lg:hidden p-6 bg-white">
          <div className="text-center max-md:text-left">
            <h1 className="text-xl font-bold text-gray-900">
              <span className="font-bold">Safeguard</span>{" "}
              <span className="font-normal">Media</span>
            </h1>
          </div>
        </div>

        {/* Error message */}
        <div className="flex-1 flex items-center justify-center p-4 lg:p-8 bg-white">
          <div className="w-full max-w-md">
            <div className="text-center">
              <h2 className="text-2xl font-semibold text-gray-900">
                Verification Failed
              </h2>
            </div>
            <div className="border border-gray-200 shadow-sm rounded-2xl mt-5">
              <div className="p-8 space-y-6 text-center">
                <div className="flex justify-center mb-4">
                  <div className="bg-red-100 p-4 rounded-full">
                    <AlertCircle className="w-8 h-8 text-red-600" />
                  </div>
                </div>

                <div className="space-y-3">
                  <h3 className="text-lg font-semibold text-gray-900">
                    Something went wrong
                  </h3>
                  <p className="text-gray-600">{errorMessage}</p>
                </div>

                <div className="space-y-3">
                  <button
                    type="button"
                    className="w-full h-12 bg-[#0F2FA3] hover:bg-blue-700 text-white font-medium rounded-[50px]"
                    onClick={() => navigate("/signup")}
                  >
                    Back to Sign Up
                  </button>

                  <button
                    type="button"
                    className="w-full h-12 border border-gray-300 hover:bg-gray-50 text-gray-700 font-medium rounded-[50px]"
                    onClick={() => navigate("/signin")}
                  >
                    Try Sign In
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Right side - Desktop branding */}
        <div className="hidden lg:block flex-1 relative bg-gradient-to-br from-red-50 to-rose-100">
          <div className="absolute top-8 right-8">
            <div className="text-black font-semibold text-xl">
              <span className="font-bold">Safeguard</span>{" "}
              <span className="font-normal">Media</span>
            </div>
          </div>
          <div className="flex items-center justify-center h-full">
            <div className="text-center p-8">
              <div className="bg-white p-6 rounded-full w-24 h-24 flex items-center justify-center mx-auto mb-6">
                <AlertCircle className="w-12 h-12 text-red-600" />
              </div>
              <h3 className="text-2xl font-bold text-gray-900 mb-4">
                Verification Issues
              </h3>
              <p className="text-gray-600 max-w-sm">
                The verification link may have expired or is invalid. Please try
                signing up again or contact support if the issue persists.
              </p>
            </div>
          </div>
        </div>
      </div>
    );
  }

  // Fallback (shouldn't reach here)
  return null;
}

export default VerifyEmail;
