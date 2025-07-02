import type React from "react";
import { useState, useEffect } from "react";
import {
  Eye,
  EyeOff,
  AlertCircle,
  Loader,
  CheckCircle,
  Lock,
  X,
} from "lucide-react";
import { useNavigate, useSearchParams } from "react-router-dom";
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
  const [searchParams] = useSearchParams();
  const token = searchParams.get("token");

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
  // Redirect if no token
  //   useEffect(() => {
  //     if (!token) {
  //       navigate("/forgot-password");
  //     }
  //   }, [token, navigate]);

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
                Password Reset Successful
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
                    All Set!
                  </h3>
                  <p className="text-gray-600">
                    Your password has been successfully reset. You can now sign
                    in with your new password.
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

        {/* Right side - Desktop only branding */}
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
                Welcome Back!
              </h3>
              <p className="text-gray-600 max-w-sm">
                Your account is now secure with your new password. Ready to get
                started?
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

      {/* Left side - Reset password form */}
      <div className="flex-1 flex items-center justify-center p-4 lg:p-8 bg-white">
        <div className="w-full max-w-sm">
          <div className="text-center">
            <h2 className="text-2xl font-semibold text-gray-900">
              Set New Password
            </h2>
            <p className="text-gray-600 mt-2 text-sm">
              Enter your new password below.
            </p>
          </div>

          <div className="border border-gray-200 shadow-sm rounded-2xl mt-5">
            <div className="p-8 space-y-6">
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
                  <label
                    htmlFor="password"
                    className="block text-sm font-medium text-gray-700"
                  >
                    New Password
                  </label>
                  <div className="relative">
                    <input
                      id="password"
                      name="password"
                      type={showPassword ? "text" : "password"}
                      placeholder="Enter your new password"
                      value={formData.password}
                      onChange={handleInputChange}
                      className={`w-full h-12 px-3 py-2 border rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 pr-10 ${
                        errors.password ? "border-red-300" : "border-gray-300"
                      }`}
                    />
                    <button
                      type="button"
                      className="absolute inset-y-0 right-0 pr-3 flex items-center"
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
                    <p className="text-sm text-red-600">{errors.password}</p>
                  )}
                </div>

                {/* Confirm Password Input */}
                <div className="space-y-2">
                  <label
                    htmlFor="confirmPassword"
                    className="block text-sm font-medium text-gray-700"
                  >
                    Confirm New Password
                  </label>
                  <div className="relative">
                    <input
                      id="confirmPassword"
                      name="confirmPassword"
                      type={showConfirmPassword ? "text" : "password"}
                      placeholder="Confirm your new password"
                      value={formData.confirmPassword}
                      onChange={handleInputChange}
                      className={`w-full h-12 px-3 py-2 border rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 pr-10 ${
                        errors.confirmPassword
                          ? "border-red-300"
                          : "border-gray-300"
                      }`}
                    />
                    <button
                      type="button"
                      className="absolute inset-y-0 right-0 pr-3 flex items-center"
                      onClick={() =>
                        setShowConfirmPassword(!showConfirmPassword)
                      }
                    >
                      {showConfirmPassword ? (
                        <EyeOff className="h-4 w-4 text-gray-400" />
                      ) : (
                        <Eye className="h-4 w-4 text-gray-400" />
                      )}
                    </button>
                  </div>
                  {errors.confirmPassword && (
                    <p className="text-sm text-red-600">
                      {errors.confirmPassword}
                    </p>
                  )}
                </div>

                {/* Password Requirements */}
                <div className="bg-gray-50 p-3 rounded-lg">
                  <p className="text-xs text-gray-600 mb-2">
                    Password must contain:
                  </p>
                  <ul className="text-xs text-gray-600 space-y-1">
                    <li className="flex items-center">
                      <div
                        className={`w-1.5 h-1.5 rounded-full mr-2 ${
                          formData.password.length >= 8
                            ? "bg-green-500"
                            : "bg-gray-300"
                        }`}
                      />
                      At least 8 characters
                    </li>
                    <li className="flex items-center">
                      <div
                        className={`w-1.5 h-1.5 rounded-full mr-2 ${
                          /[A-Z]/.test(formData.password)
                            ? "bg-green-500"
                            : "bg-gray-300"
                        }`}
                      />
                      One uppercase letter
                    </li>
                    <li className="flex items-center">
                      <div
                        className={`w-1.5 h-1.5 rounded-full mr-2 ${
                          /[a-z]/.test(formData.password)
                            ? "bg-green-500"
                            : "bg-gray-300"
                        }`}
                      />
                      One lowercase letter
                    </li>
                    <li className="flex items-center">
                      <div
                        className={`w-1.5 h-1.5 rounded-full mr-2 ${
                          /\d/.test(formData.password)
                            ? "bg-green-500"
                            : "bg-gray-300"
                        }`}
                      />
                      One number
                    </li>
                  </ul>
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
                      Updating...
                    </>
                  ) : (
                    "Update Password"
                  )}
                </button>

                {/* Back to sign in */}
                {/* <div className="text-center text-sm text-gray-600">
                  <button
                    type="button"
                    className="text-blue-600 hover:text-blue-500 hover:underline font-medium"
                    onClick={() => navigate("/signin")}
                  >
                    Back to Sign In
                  </button>
                </div> */}
              </form>
            </div>
          </div>
        </div>
      </div>

      {/* Right side - Desktop only branding */}
      <div className="hidden lg:block flex-1 relative bg-gradient-to-br from-purple-50 to-indigo-100">
        <div className="absolute top-8 right-8">
          <div className="text-black font-semibold text-xl">
            <span className="font-bold">Safeguard</span>{" "}
            <span className="font-normal">Media</span>
          </div>
        </div>
        <div className="flex items-center justify-center h-full">
          <div className="text-center p-8">
            <div className="bg-white p-6 rounded-full w-24 h-24 flex items-center justify-center mx-auto mb-6">
              <Lock className="w-12 h-12 text-purple-600" />
            </div>
            <h3 className="text-2xl font-bold text-gray-900 mb-4">
              Secure Your Account
            </h3>
            <p className="text-gray-600 max-w-sm">
              Choose a strong password to keep your account safe and secure.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

export default ResetPassword;
