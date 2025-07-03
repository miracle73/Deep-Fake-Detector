import type React from "react";
import { useState, useEffect } from "react";
import {
  Eye,
  EyeOff,
  AlertCircle,
  Loader,
  CheckCircle,
  ArrowLeft,
  X,
} from "lucide-react";
import { useNavigate, useParams } from "react-router-dom";
import { useResetPasswordMutation } from "../services/apiService";

interface FormData {
  password: string;
  confirmPassword: string;
}

interface FormErrors {
  password?: string;
  confirmPassword?: string;
  general?: string;
}

function ResetPassword() {
  const navigate = useNavigate();
  // const [searchParams] = useSearchParams();
  // const token = searchParams.get("token");
  const { token } = useParams<{ token: string }>();

  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const [formData, setFormData] = useState<FormData>({
    password: "",
    confirmPassword: "",
  });
  const [errors, setErrors] = useState<FormErrors>({});
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isSuccess, setIsSuccess] = useState(false);

  const [resetPassword] = useResetPasswordMutation();
  useEffect(() => {
    if (errors.general) {
      const timer = setTimeout(() => {
        setErrors((prev) => ({ ...prev, general: undefined }));
      }, 3000);

      return () => clearTimeout(timer);
    }
  }, [errors.general]);

  // Redirect to forgot-password if no token
  useEffect(() => {
    if (!token) {
      navigate("/forgot-password");
    }
  }, [token, navigate]);

  // Validation functions
  const validatePassword = (password: string): boolean => {
    // At least 8 characters, 1 uppercase, 1 lowercase, 1 number
    const passwordRegex =
      /^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)[a-zA-Z\d@$!%*?&]{8,}$/;
    return passwordRegex.test(password);
  };

  const validateForm = (): FormErrors => {
    const newErrors: FormErrors = {};

    // Password validation
    if (!formData.password) {
      newErrors.password = "Password is required";
    } else if (!validatePassword(formData.password)) {
      newErrors.password =
        "Password must be at least 8 characters with uppercase, lowercase, and number";
    }

    // Confirm password validation
    if (!formData.confirmPassword) {
      newErrors.confirmPassword = "Please confirm your password";
    } else if (formData.password !== formData.confirmPassword) {
      newErrors.confirmPassword = "Passwords do not match";
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
      const result = await resetPassword({
        token: token!,
        password: formData.password,
      }).unwrap();

      console.log("Password reset successful:", result);
      setIsSuccess(true);
    } catch (error: unknown) {
      console.error("Password reset failed:", error);
      if (error && typeof error === "object" && "data" in error) {
        const apiError = error as {
          data?: { message?: string; errors?: FormErrors };
        };
        if (apiError.data?.message) {
          setErrors({ general: apiError.data.message });
        } else if (apiError.data?.errors) {
          setErrors(apiError.data.errors);
        } else {
          setErrors({ general: "Failed to reset password. Please try again." });
        }
      } else if (error && typeof error === "object" && "message" in error) {
        const messageError = error as { message: string };
        setErrors({ general: messageError.message });
      } else {
        setErrors({ general: "Failed to reset password. Please try again." });
      }
    } finally {
      setIsSubmitting(false);
      setFormData({ password: "", confirmPassword: "" });
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
              Password Reset Successful
            </h1>

            <div className="bg-white rounded-3xl p-6 sm:p-8 shadow-sm border border-gray-200">
              <div className="text-center space-y-6">
                <div className="flex justify-center mb-4">
                  <div className="bg-green-100 p-4 rounded-full">
                    <CheckCircle className="w-8 h-8 text-green-600" />
                  </div>
                </div>

                <div className="space-y-3">
                  <h3 className="text-lg font-semibold text-gray-900">
                    All Set!
                  </h3>
                  <p className="text-gray-600 text-sm">
                    Your password has been successfully reset. You can now sign
                    in with your new password.
                  </p>
                </div>

                <button
                  type="button"
                  className="w-full h-12 bg-[#0F2FA3] hover:bg-blue-700 text-white font-medium rounded-full transition-colors"
                  onClick={() => navigate("/signin")}
                >
                  Continue to Sign In
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
          onClick={() => navigate("/")}
        >
          <ArrowLeft className="w-4 h-4 mr-2" />
          Go back to home
        </button>
      </div>

      {/* Main Content */}
      <div className="flex items-center justify-center min-h-[calc(100vh-64px)] p-4">
        <div className="w-full max-w-md">
          <h1 className="text-2xl sm:text-3xl font-semibold text-gray-900 text-center mb-8">
            Reset Password
          </h1>

          <div className="bg-white rounded-3xl p-6 sm:p-8 shadow-sm border border-gray-200">
            <div className="mb-6">
              <h3 className="text-base font-medium text-gray-900 mb-4">
                Create new password
              </h3>
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

            <form onSubmit={handleSubmit} className="space-y-4">
              {/* Password Input */}
              <div className="space-y-2">
                <div className="relative">
                  <input
                    id="password"
                    name="password"
                    type={showPassword ? "text" : "password"}
                    placeholder="placeholder"
                    value={formData.password}
                    onChange={handleInputChange}
                    className={`w-full h-12 px-4 py-3 border rounded-full focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 pr-12 transition-colors ${
                      errors.password ? "border-red-300" : "border-gray-300"
                    }`}
                  />
                  <button
                    type="button"
                    className="absolute inset-y-0 right-0 pr-4 flex items-center"
                    onClick={() => setShowPassword(!showPassword)}
                  >
                    {showPassword ? (
                      <EyeOff className="h-4 w-4 text-gray-400" />
                    ) : (
                      <Eye className="h-4 w-4 text-gray-400" />
                    )}
                  </button>
                </div>
                {errors.password && (
                  <p className="text-sm text-red-600 px-4">{errors.password}</p>
                )}
              </div>

              {/* Confirm Password Input */}
              <div className="space-y-2">
                <label className="block text-sm font-medium text-gray-700 px-1">
                  Confirm new password
                </label>
                <div className="relative">
                  <input
                    id="confirmPassword"
                    name="confirmPassword"
                    type={showConfirmPassword ? "text" : "password"}
                    placeholder="placeholder"
                    value={formData.confirmPassword}
                    onChange={handleInputChange}
                    className={`w-full h-12 px-4 py-3 border rounded-full focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 pr-12 transition-colors ${
                      errors.confirmPassword
                        ? "border-red-300"
                        : "border-gray-300"
                    }`}
                  />
                  <button
                    type="button"
                    className="absolute inset-y-0 right-0 pr-4 flex items-center"
                    onClick={() => setShowConfirmPassword(!showConfirmPassword)}
                  >
                    {showConfirmPassword ? (
                      <EyeOff className="h-4 w-4 text-gray-400" />
                    ) : (
                      <Eye className="h-4 w-4 text-gray-400" />
                    )}
                  </button>
                </div>
                {errors.confirmPassword && (
                  <p className="text-sm text-red-600 px-4">
                    {errors.confirmPassword}
                  </p>
                )}
              </div>

              {/* Submit Button */}
              <div className="pt-4">
                <button
                  type="submit"
                  disabled={isSubmitting}
                  className="w-full h-12 bg-[#0F2FA3] hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed text-white font-medium rounded-full flex items-center justify-center transition-colors"
                >
                  {isSubmitting ? (
                    <>
                      <Loader className="animate-spin -ml-1 mr-2 h-4 w-4" />
                      Creating...
                    </>
                  ) : (
                    "Create Password"
                  )}
                </button>
              </div>
            </form>
          </div>
        </div>
      </div>
    </div>
  );
}

export default ResetPassword;
