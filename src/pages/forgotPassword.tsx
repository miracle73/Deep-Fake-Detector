import type React from "react";
import { useState } from "react";
import { AlertCircle, Loader, ArrowLeft, Mail } from "lucide-react";
import { useNavigate } from "react-router-dom";
import { useForgotPasswordMutation } from "../services/apiService";

interface FormData {
  email: string;
}

interface FormErrors {
  email?: string;
  general?: string;
}

function ForgotPassword() {
  const navigate = useNavigate();
  const [formData, setFormData] = useState<FormData>({
    email: "",
  });
  const [errors, setErrors] = useState<FormErrors>({});
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isSuccess, setIsSuccess] = useState(false);

  const [forgotPassword] = useForgotPasswordMutation();

  // Validation function
  const validateEmail = (email: string): boolean => {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return emailRegex.test(email);
  };

  const validateForm = (): FormErrors => {
    const newErrors: FormErrors = {};

    // Email validation
    if (!formData.email.trim()) {
      newErrors.email = "Email address is required";
    } else if (!validateEmail(formData.email)) {
      newErrors.email = "Please enter a valid email address";
    }

    return newErrors;
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;

    setFormData({
      ...formData,
      [name]: value,
    });

    // Clear error for this field when user starts typing
    if (errors[name as keyof FormErrors]) {
      setErrors({
        ...errors,
        [name]: undefined,
      });
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    // Clear previous errors
    setErrors({});

    // Validate form
    const validationErrors = validateForm();
    if (Object.keys(validationErrors).length > 0) {
      setErrors(validationErrors);
      return;
    }

    setIsSubmitting(true);

    try {
      const result = await forgotPassword({
        email: formData.email.trim().toLowerCase(),
      }).unwrap();

      console.log("Forgot password request successful:", result);
      setIsSuccess(true);
      //   navigate("/reset-password");
    } catch (error: unknown) {
      console.error("Forgot password request failed:", error);

      if (error && typeof error === "object" && "data" in error) {
        const apiError = error as {
          data?: { message?: string; errors?: FormErrors };
        };
        if (apiError.data?.message) {
          setErrors({ general: apiError.data.message });
        } else if (apiError.data?.errors) {
          setErrors(apiError.data.errors);
        } else {
          setErrors({
            general: "Failed to send reset email. Please try again.",
          });
        }
      } else if (error && typeof error === "object" && "message" in error) {
        const messageError = error as { message: string };
        setErrors({ general: messageError.message });
      } else {
        setErrors({ general: "Failed to send reset email. Please try again." });
      }
    } finally {
      setIsSubmitting(false);
    }
  };

  if (isSuccess) {
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

        {/* Success message */}
        <div className="flex-1 flex items-center justify-center p-4 lg:p-8 bg-white">
          <div className="w-full max-w-md">
            <div className="text-center">
              <h2 className="text-2xl font-semibold text-gray-900">
                Check Your Email
              </h2>
            </div>
            <div className="border border-gray-200 shadow-sm rounded-2xl mt-5">
              <div className="p-8 space-y-6 text-center">
                <div className="flex justify-center mb-4">
                  <div className="bg-green-100 p-4 rounded-full">
                    <Mail className="w-8 h-8 text-green-600" />
                  </div>
                </div>

                <div className="space-y-3">
                  <h3 className="text-lg font-semibold text-gray-900">
                    Reset Link Sent!
                  </h3>
                  <p className="text-gray-600">
                    We've sent a password reset link to{" "}
                    <strong>{formData.email}</strong>
                  </p>
                  <p className="text-sm text-gray-500">
                    Check your email and click the link to reset your password.
                    The link will expire in 15 minutes.
                  </p>
                </div>

                <div className="space-y-4">
                  {/* <button
                    type="button"
                    className="w-full h-12 bg-[#0F2FA3] hover:bg-blue-700 text-white font-medium rounded-[50px]"
                    onClick={() => navigate("/signin")}
                  >
                    Back to Sign In
                  </button> */}

                  <button
                    type="button"
                    className="w-full h-12 bg-[#0F2FA3] hover:bg-blue-700 text-white font-medium rounded-[50px]"
                    onClick={handleSubmit}
                  >
                    Resend Reset Link
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Right side - Desktop only branding */}
        <div className="hidden lg:block flex-1 relative bg-gradient-to-br from-blue-50 to-indigo-100">
          <div className="absolute top-8 right-8">
            <div className="text-black font-semibold text-xl">
              <span className="font-bold">Safeguard</span>{" "}
              <span className="font-normal">Media</span>
            </div>
          </div>
          <div className="flex items-center justify-center h-full">
            <div className="text-center p-8">
              <div className="bg-white p-6 rounded-full w-24 h-24 flex items-center justify-center mx-auto mb-6">
                <Mail className="w-12 h-12 text-blue-600" />
              </div>
              <h3 className="text-2xl font-bold text-gray-900 mb-4">
                Password Recovery
              </h3>
              <p className="text-gray-600 max-w-sm">
                Secure password reset process to get you back into your account
                quickly and safely.
              </p>
            </div>
          </div>
        </div>
      </div>
    );
  }

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

      {/* Left side - Forgot password form */}
      <div className="flex-1 flex items-center justify-center p-4 lg:p-8 bg-white">
        <div className="w-full max-w-sm">
          {/* Back button */}
          <button
            type="button"
            className="flex items-center text-sm text-gray-600 hover:text-gray-900 mb-6"
            onClick={() => navigate("/signin")}
          >
            <ArrowLeft className="w-4 h-4 mr-2" />
            Back to Sign In
          </button>

          <div className="text-center">
            <h2 className="text-2xl font-semibold text-gray-900">
              Forgot Password?
            </h2>
            <p className="text-gray-600 mt-2 text-sm">
              No worries, we'll send you reset instructions.
            </p>
          </div>

          <div className="border border-gray-200 shadow-sm rounded-2xl mt-5">
            <div className="p-8 space-y-6">
              {/* General Error Message */}
              {errors.general && (
                <div className="flex items-center p-3 text-sm text-red-600 bg-red-50 border border-red-200 rounded-lg">
                  <AlertCircle className="w-4 h-4 mr-2 flex-shrink-0" />
                  <span>{errors.general}</span>
                </div>
              )}

              <form onSubmit={handleSubmit} className="space-y-4">
                {/* Email Input */}
                <div className="space-y-2">
                  <label
                    htmlFor="email"
                    className="block text-sm font-medium text-gray-700"
                  >
                    Email address
                  </label>
                  <input
                    id="email"
                    name="email"
                    type="email"
                    placeholder="Enter your email address"
                    value={formData.email}
                    onChange={handleInputChange}
                    className={`w-full h-12 px-3 py-2 border rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 ${
                      errors.email ? "border-red-300" : "border-gray-300"
                    }`}
                  />
                  {errors.email && (
                    <p className="text-sm text-red-600">{errors.email}</p>
                  )}
                </div>

                {/* Submit Button */}
                <button
                  type="submit"
                  disabled={isSubmitting}
                  className="w-full h-12 bg-[#0F2FA3] hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed text-white font-medium rounded-[50px] flex items-center justify-center"
                >
                  {isSubmitting ? (
                    <>
                      <Loader className="animate-spin -ml-1 mr-2 h-4 w-4" />
                      Sending...
                    </>
                  ) : (
                    "Reset Password"
                  )}
                </button>

                {/* Sign up link */}
                <div className="text-center text-sm text-gray-600">
                  {"Don't have an account? "}
                  <button
                    type="button"
                    className="text-blue-600 hover:text-blue-500 hover:underline font-medium"
                    onClick={() => navigate("/signup")}
                  >
                    Sign Up
                  </button>
                </div>
              </form>
            </div>
          </div>
        </div>
      </div>

      {/* Right side - Desktop only branding */}
      <div className="hidden lg:block flex-1 relative bg-gradient-to-br from-blue-50 to-indigo-100">
        <div className="absolute top-8 right-8">
          <div className="text-black font-semibold text-xl">
            <span className="font-bold">Safeguard</span>{" "}
            <span className="font-normal">Media</span>
          </div>
        </div>
        <div className="flex items-center justify-center h-full">
          <div className="text-center p-8">
            <div className="bg-white p-6 rounded-full w-24 h-24 flex items-center justify-center mx-auto mb-6">
              <Mail className="w-12 h-12 text-blue-600" />
            </div>
            <h3 className="text-2xl font-bold text-gray-900 mb-4">
              Reset Your Password
            </h3>
            <p className="text-gray-600 max-w-sm">
              Enter your email address and we'll send you a link to reset your
              password.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

export default ForgotPassword;
