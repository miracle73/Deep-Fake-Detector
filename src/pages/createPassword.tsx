import type React from "react";
import { useState, useEffect } from "react";
import {
  Eye,
  EyeOff,
  AlertCircle,
  Loader,
  X,
  // Check
} from "lucide-react";
import "../App.css";
import { useNavigate, useSearchParams } from "react-router-dom";
import { useCreatePasswordMutation } from "../services/apiService";
import SafeguardMediaLogo from "../assets/images/SafeguardMedia8.svg";

interface FormData {
  password: string;
  confirmPassword: string;
}

interface FormErrors {
  password?: string;
  confirmPassword?: string;
  general?: string;
}

// interface PasswordValidation {
//   minLength: boolean;
//   hasUppercase: boolean;
//   hasLowercase: boolean;
//   hasNumber: boolean;
// }

function CreatePassword() {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const token = searchParams.get("token");
  const [createPassword] = useCreatePasswordMutation();
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const [formData, setFormData] = useState<FormData>({
    password: "",
    confirmPassword: "",
  });
  const [errors, setErrors] = useState<FormErrors>({});
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [successMessage, setSuccessMessage] = useState<string>("");
  //   const [passwordValidation, setPasswordValidation] =
  //     useState<PasswordValidation>({
  //       minLength: false,
  //       hasUppercase: false,
  //       hasLowercase: false,
  //       hasNumber: false,
  //     });
  //   const [showValidation, setShowValidation] = useState(false);

  //   const [resetPassword] = useResetPasswordMutation();

  // Redirect if no token is present
  useEffect(() => {
    if (!token) {
      navigate("/signin");
    }
  }, [token, navigate]);

  useEffect(() => {
    if (errors.general) {
      const timer = setTimeout(() => {
        setErrors((prev) => ({ ...prev, general: undefined }));
      }, 5000);

      return () => clearTimeout(timer);
    }
  }, [errors.general]);

  useEffect(() => {
    if (successMessage) {
      const timer = setTimeout(() => {
        setSuccessMessage("");
      }, 3000);

      return () => clearTimeout(timer);
    }
  }, [successMessage]);

  // Real-time password validation
  //   useEffect(() => {
  //     const password = formData.password;
  //     setPasswordValidation({
  //       minLength: password.length >= 8,
  //       hasUppercase: /[A-Z]/.test(password),
  //       hasLowercase: /[a-z]/.test(password),
  //       hasNumber: /\d/.test(password),
  //     });
  //   }, [formData.password]);

  // Password validation function - matches SignUp component
  const validatePassword = (password: string): boolean => {
    // At least 8 characters, 1 uppercase, 1 lowercase, 1 number
    const passwordRegex =
      /^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)[a-zA-Z\d@$!%*?&]{8,}$/;
    return passwordRegex.test(password);
  };

  const validateForm = (): FormErrors => {
    const newErrors: FormErrors = {};

    // Password validation - matches SignUp component validation
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

    // Show validation when user starts typing password
    // if (name === "password") {
    //   setShowValidation(value.length > 0);
    // }

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

    if (!token) {
      setErrors({
        general: "Invalid or expired token. Please try again.",
      });
      return;
    }

    setIsSubmitting(true);

    try {
      await createPassword({
        token: token,
        password: formData.password,
      }).unwrap();

      setSuccessMessage(
        "Password created successfully! Redirecting to sign in..."
      );

      // Redirect to signin after successful password creation
      setTimeout(() => {
        navigate("/signin");
      }, 2000);
    } catch (error: unknown) {
      console.error("Password creation failed:", error);

      if (error && typeof error === "object" && "data" in error) {
        const apiError = error as {
          data?: { message?: string; errors?: FormErrors };
        };
        if (apiError.data?.message) {
          setErrors({ general: apiError.data.message });
        } else if (apiError.data?.errors) {
          setErrors(apiError.data.errors);
        } else {
          setErrors({ general: "Password creation failed. Please try again." });
        }
      } else if (error && typeof error === "object" && "message" in error) {
        const messageError = error as { message: string };
        setErrors({ general: messageError.message });
      } else {
        setErrors({ general: "Password creation failed. Please try again." });
      }
    } finally {
      setIsSubmitting(false);
    }
  };

  //   const ValidationItem = ({
  //     isValid,
  //     text,
  //   }: {
  //     isValid: boolean;
  //     text: string;
  //   }) => (
  //     <li
  //       className={`flex items-center text-sm ${
  //         isValid ? "text-green-600" : "text-red-600"
  //       }`}
  //     >
  //       {isValid ? (
  //         <Check className="w-4 h-4 mr-2 flex-shrink-0" />
  //       ) : (
  //         <X className="w-4 h-4 mr-2 flex-shrink-0" />
  //       )}
  //       <span className={isValid ? "line-through" : ""}>{text}</span>
  //     </li>
  //   );

  return (
    <div className="min-h-screen flex flex-col lg:flex-row">
      {/* Mobile header - only visible on mobile */}
      <div className="lg:hidden p-6 bg-white">
        <div className="text-center max-md:text-left">
          <h1 className="text-xl font-bold flex items-center gap-2 text-gray-900">
            <img
              src={SafeguardMediaLogo}
              alt="Safeguardmedia Logo"
              className="h-12 w-auto"
            />
            <span className="font-bold">Safeguardmedia</span>
          </h1>
        </div>
      </div>

      {/* Left side - Create password form */}
      <div className="flex-1 flex items-center justify-center p-4 lg:p-8 bg-white">
        <div className="w-full max-w-md">
          <div className="text-center">
            <h2 className="text-2xl font-semibold text-gray-900">
              Create New Password
            </h2>
            <p className="text-gray-600 mt-2">
              Please enter your new password below
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

              {/* Success Message */}
              {successMessage && (
                <div className="flex items-center p-3 text-sm text-green-600 bg-green-50 border border-green-200 rounded-lg">
                  <div className="w-4 h-4 mr-2 flex-shrink-0">
                    <svg
                      viewBox="0 0 20 20"
                      fill="currentColor"
                      className="w-4 h-4"
                    >
                      <path
                        fillRule="evenodd"
                        d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
                        clipRule="evenodd"
                      />
                    </svg>
                  </div>
                  <span>{successMessage}</span>
                  <button
                    type="button"
                    onClick={() => setSuccessMessage("")}
                    className="ml-auto text-green-400 hover:text-green-600"
                  >
                    <X className="w-4 h-4" />
                  </button>
                </div>
              )}

              <form onSubmit={handleSubmit} className="space-y-6">
                {/* Password Field */}
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
                      {!showPassword ? (
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

                {/* {showValidation && (
                  <div className="bg-gray-50 border border-gray-200 rounded-lg p-4">
                    <h4 className="text-sm font-medium text-gray-800 mb-3">
                      Password Requirements:
                    </h4>
                    <ul className="space-y-2">
                      <ValidationItem
                        isValid={passwordValidation.minLength}
                        text="At least 8 characters long"
                      />
                      <ValidationItem
                        isValid={passwordValidation.hasUppercase}
                        text="Contains at least one uppercase letter"
                      />
                      <ValidationItem
                        isValid={passwordValidation.hasLowercase}
                        text="Contains at least one lowercase letter"
                      />
                      <ValidationItem
                        isValid={passwordValidation.hasNumber}
                        text="Contains at least one number"
                      />
                    </ul>
                  </div>
                )} */}

                {/* Confirm Password Field */}
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
                      {!showConfirmPassword ? (
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

                {/* Submit Button */}
                <button
                  type="submit"
                  disabled={isSubmitting}
                  className="w-full h-12 bg-[#0F2FA3] hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed text-white font-medium rounded-[50px] flex items-center justify-center"
                >
                  {isSubmitting ? (
                    <>
                      <Loader className="animate-spin -ml-1 mr-2 h-4 w-4" />
                      Creating Password...
                    </>
                  ) : (
                    "Create Password"
                  )}
                </button>

                {/* Back to Sign In Link */}
                <div className="text-center text-sm text-gray-600">
                  <button
                    type="button"
                    className="text-blue-600 hover:text-blue-500 hover:underline font-medium"
                    onClick={() => navigate("/signin")}
                  >
                    Back to Sign In
                  </button>
                </div>
              </form>
            </div>
          </div>
        </div>
      </div>

      {/* Right side - Background image (desktop only) */}
      <div className="hidden lg:flex flex-1 bg-gradient-to-br from-blue-50 to-indigo-100 flex-col justify-between items-center p-8 min-h-screen overflow-y-auto">
        {/* Top section - Logo and brand */}
        <div className="text-center pt-16">
          <div className="flex items-center justify-center mb-4">
            <img
              src={SafeguardMediaLogo}
              alt="Safeguardmedia Logo"
              className="h-16 w-auto mr-3"
            />
          </div>
          <h1 className="text-3xl font-bold text-gray-900 mb-2">
            Safeguardmedia
          </h1>
          <p className="text-lg text-gray-600 max-w-md">
            Secure your account with a strong new password
          </p>
        </div>

        {/* Bottom section - Security message */}
        <div className="text-center pb-8 max-w-sm">
          <p className="text-sm text-gray-600 mb-4">
            Your security is our priority. Create a strong password to protect
            your account.
          </p>
          <div className="flex justify-center space-x-6 text-sm">
            <a
              onClick={() => navigate("/terms-and-conditions")}
              className="text-blue-600 hover:text-blue-800 hover:underline font-medium"
            >
              Terms of Service
            </a>
            <a
              onClick={() => navigate("/privacy-policy")}
              className="text-blue-600 hover:text-blue-800 hover:underline font-medium"
            >
              Privacy Policy
            </a>
          </div>
        </div>
      </div>
    </div>
  );
}

export default CreatePassword;
