import type React from "react";
import { useState, useEffect } from "react";
import { AlertCircle, Loader, ArrowLeft, Mail, X } from "lucide-react";
import { useNavigate } from "react-router-dom";
import { useForgotPasswordMutation } from "../services/apiService";

interface FormData {
  email: string;
}

interface FormErrors {
  email?: string;
  general?: string;
}

function ForgotPassword2() {
  const navigate = useNavigate();
  const [formData, setFormData] = useState<FormData>({
    email: "",
  });
  const [errors, setErrors] = useState<FormErrors>({});
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isSuccess, setIsSuccess] = useState(false);

  const [forgotPassword] = useForgotPasswordMutation();
  useEffect(() => {
    if (errors.general) {
      const timer = setTimeout(() => {
        setErrors((prev) => ({ ...prev, general: undefined }));
      }, 3000);

      return () => clearTimeout(timer);
    }
  }, [errors.general]);
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

  const handleResendResetLink = async () => {
    setIsSubmitting(true);

    try {
      const result = await forgotPassword({
        email: formData.email.trim().toLowerCase(),
      }).unwrap();

      console.log("Resend password request successful:", result);
      // You could show a success message here if needed
    } catch (error: unknown) {
      console.error("Resend password request failed:", error);
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
            general: "Failed to resend reset email. Please try again.",
          });
        }
      } else if (error && typeof error === "object" && "message" in error) {
        const messageError = error as { message: string };
        setErrors({ general: messageError.message });
      } else {
        setErrors({
          general: "Failed to resend reset email. Please try again.",
        });
      }
    } finally {
      setIsSubmitting(false);
    }
  };
  if (isSuccess) {
    return (
      <div className="min-h-screen bg-gray-100">
        {/* Blue Header */}
        <div className="bg-[#0F2FA3] text-white p-4">
          <button
            type="button"
            className="flex items-center text-sm hover:text-gray-200"
            onClick={() => navigate("/")}
          >
            <ArrowLeft className="w-4 h-4 mr-2" />
            Go back to home
          </button>
        </div>

        {/* Success Content */}
        <div className="flex items-center justify-center min-h-[calc(100vh-64px)] p-4">
          <div className="w-full max-w-md">
            <h1 className="text-2xl sm:text-3xl font-semibold text-gray-900 text-center mb-8">
              Check Your Email
            </h1>

            <div className="bg-white rounded-3xl p-6 sm:p-8 shadow-sm border border-gray-200">
              <div className="text-center space-y-6">
                <div className="flex justify-center mb-4">
                  <div className="bg-blue-100 p-4 rounded-full">
                    <Mail className="w-8 h-8 text-blue-600" />
                  </div>
                </div>

                <div className="space-y-3">
                  <h3 className="text-lg font-semibold text-gray-900">
                    Reset Link Sent!
                  </h3>
                  <p className="text-gray-600 text-sm">
                    We've sent a password reset link to{" "}
                    <strong>{formData.email}</strong>
                  </p>
                  <p className="text-xs text-gray-500">
                    Check your email and click the link to reset your password.
                    The link will expire in 15 minutes.
                  </p>
                </div>

                <button
                  type="button"
                  disabled={isSubmitting}
                  className="w-full h-12 bg-[#0F2FA3] hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed text-white font-medium rounded-full transition-colors flex items-center justify-center"
                  onClick={handleResendResetLink}
                >
                  {isSubmitting ? (
                    <>
                      <Loader className="animate-spin -ml-1 mr-2 h-4 w-4" />
                      Resending...
                    </>
                  ) : (
                    "Resend Reset Link"
                  )}
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-100">
      {/* Blue Header */}
      <div className="bg-[#0F2FA3] text-white p-4">
        <button
          type="button"
          className="flex items-center text-sm hover:text-gray-200 transition-colors"
          onClick={() => navigate("/signin")}
        >
          <ArrowLeft className="w-4 h-4 mr-2" />
          Go back to home
        </button>
      </div>

      {/* Main Content */}
      <div className="flex items-center justify-center min-h-[calc(100vh-64px)] p-4">
        <div className="w-full max-w-md">
          <h1 className="text-2xl sm:text-3xl font-semibold text-gray-900 text-center mb-8">
            Forgot Password
          </h1>

          <div className="bg-white rounded-3xl p-6 sm:p-8 shadow-sm border border-gray-200">
            <div className="text-center mb-6">
              <p className="text-gray-600 text-sm">
                Enter your email below and we'll send you a link to reset your
                password.
              </p>
            </div>

            {/* General Error Message */}
            {errors.general && (
              <div className="flex items-center p-3 text-sm text-red-600 bg-red-50 border border-red-200 rounded-lg">
                <AlertCircle className="w-4 h-4 mr-2 flex-shrink-0" />
                <span>{errors.general}</span>
                <button
                  type="button"
                  onClick={() => setErrors({ ...errors, general: undefined })}
                  className="ml-auto text-red-400 hover:text-red-600"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>
            )}

            <form onSubmit={handleSubmit} className="space-y-6">
              {/* Email Input */}
              <div className="space-y-2">
                <input
                  id="email"
                  name="email"
                  type="email"
                  placeholder="Enter your work or personal email"
                  value={formData.email}
                  onChange={handleInputChange}
                  className={`w-full h-12 px-4 py-3 border rounded-full focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-colors ${
                    errors.email ? "border-red-300" : "border-gray-300"
                  }`}
                />
                {errors.email && (
                  <p className="text-sm text-red-600 px-4">{errors.email}</p>
                )}
              </div>

              {/* Submit Button */}
              <button
                type="submit"
                disabled={isSubmitting}
                className="w-full h-12 bg-[#0F2FA3] hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed text-white font-medium rounded-full flex items-center justify-center transition-colors"
              >
                {isSubmitting ? (
                  <>
                    <Loader className="animate-spin -ml-1 mr-2 h-4 w-4" />
                    Sending...
                  </>
                ) : (
                  "Submit"
                )}
              </button>
            </form>
          </div>
        </div>
      </div>
    </div>
  );
}

export default ForgotPassword2;
