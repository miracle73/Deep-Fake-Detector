import type React from "react";
import { useState, useEffect } from "react";
import { Eye, EyeOff, AlertCircle, Loader, Mail, X } from "lucide-react";
import "../App.css";
import BackgroundImage from "../assets/images/signin-page-2.png";
import { useNavigate } from "react-router-dom";
import { useRegisterMutation } from "../services/apiService";
import PhoneInput from "react-phone-number-input";
import "react-phone-number-input/style.css";
import { useGoogleLogin } from "@react-oauth/google";
import { useGoogleLoginMutation } from "../services/apiService";
import { useDispatch } from "react-redux";
import { loginUser } from "../store/slices/authSlices";
import { setUserInfo } from "../store/slices/userSlices";

interface FormData {
  firstName: string;
  lastName: string;
  email: string;
  phone: string;
  password: string;
  agreeTerms: boolean;
}

interface FormErrors {
  firstName?: string;
  lastName?: string;
  email?: string;
  phone?: string;
  password?: string;
  agreeTerms?: string;
  general?: string;
}

function SignUp() {
  const dispatch = useDispatch();
  const navigate = useNavigate();
  const [showPassword, setShowPassword] = useState(false);
  const [accountType, setAccountType] = useState("individual");
  const [formData, setFormData] = useState<FormData>({
    firstName: "",
    lastName: "",
    email: "",
    phone: "",
    password: "",
    agreeTerms: false,
  });
  const [errors, setErrors] = useState<FormErrors>({});
  const [isSubmitting, setIsSubmitting] = useState(false);

  const [register] = useRegisterMutation();
  useEffect(() => {
    if (errors.general) {
      const timer = setTimeout(() => {
        setErrors((prev) => ({ ...prev, general: undefined }));
      }, 3000);

      return () => clearTimeout(timer);
    }
  }, [errors.general]);

  // Validation functions
  const validateEmail = (email: string): boolean => {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return emailRegex.test(email);
  };

  const validatePhone = (phone: string): boolean => {
    // Basic phone validation
    const phoneRegex = /^[+]?[1-9][\d]{0,15}$/;

    return phoneRegex.test(phone.replace(/[\s\-()]/g, ""));
  };

  const validatePassword = (password: string): boolean => {
    // At least 8 characters, 1 uppercase, 1 lowercase, 1 number
    const passwordRegex =
      /^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)[a-zA-Z\d@$!%*?&]{8,}$/;
    return passwordRegex.test(password);
  };

  const [googleLogin] = useGoogleLoginMutation();

  const handleGoogleLogin = useGoogleLogin({
    onSuccess: async (token) => {
      console.log("Google token received:", token);
      try {
        const response = await googleLogin({
          idToken: token,
          agreedToTerms: true,
          userType: "individual",
        }).unwrap();

        // Handle successful Google login

        console.log("Google login successful:", response);

        if (response.success) {
          // Dispatch user data to Redux store
          dispatch(
            setUserInfo({
              _id: response.user.id,
              email: response.user.email,
              userType: response.user.userType,
              plan: response.user.plan,
              isGoogleUser: true,
              firstName: response.user.firstName,
              lastName: response.user.lastName,
            })
          );

          // Dispatch token to auth store
          dispatch(loginUser(response.token));

          // Store token in localStorage for persistence
          localStorage.setItem("authToken", response.token);

          // Navigate to dashboard
          navigate("/dashboard");
        } else {
          setErrors({ general: "Google sign-in failed. Please try again." });
        }
      } catch (error: unknown) {
        console.error("Google login failed:", error);

        if (error && typeof error === "object" && "data" in error) {
          const apiError = error as {
            data?: { message?: string; errors?: FormErrors };
          };
          if (apiError.data?.message) {
            setErrors({ general: apiError.data.message });
          } else if (apiError.data?.errors) {
            setErrors(apiError.data.errors);
          } else {
            setErrors({ general: "Google sign-in failed. Please try again." });
          }
        } else if (error && typeof error === "object" && "message" in error) {
          const messageError = error as { message: string };
          setErrors({ general: messageError.message });
        } else {
          setErrors({ general: "Google sign-in failed. Please try again." });
        }
      }
    },
    onError: (error) => {
      console.error("Google OAuth error:", error);
      setErrors({ general: "Google authentication failed. Please try again." });
    },
  });
  const validateForm = (): FormErrors => {
    const newErrors: FormErrors = {};

    // First name validation
    if (!formData.firstName.trim()) {
      newErrors.firstName = "First name is required";
    } else if (formData.firstName.trim().length < 2) {
      newErrors.firstName = "First name must be at least 2 characters";
    }

    // Last name validation
    if (!formData.lastName.trim()) {
      newErrors.lastName = "Last name is required";
    } else if (formData.lastName.trim().length < 2) {
      newErrors.lastName = "Last name must be at least 2 characters";
    }

    // Email validation
    if (!formData.email.trim()) {
      newErrors.email = "Email address is required";
    } else if (!validateEmail(formData.email)) {
      newErrors.email = "Please enter a valid email address";
    }

    // Phone validation
    if (!formData.phone.trim()) {
      newErrors.phone = "Phone number is required";
    } else if (!validatePhone(formData.phone)) {
      newErrors.phone = "Please enter a valid phone number";
    }

    // Password validation
    if (!formData.password) {
      newErrors.password = "Password is required";
    } else if (!validatePassword(formData.password)) {
      newErrors.password =
        "Password must be at least 8 characters with uppercase, lowercase, and number";
    }

    // Terms validation
    if (!formData.agreeTerms) {
      newErrors.agreeTerms = "You must agree to the Terms of Use";
    }

    return newErrors;
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value, type, checked } = e.target;
    const newValue = type === "checkbox" ? checked : value;

    setFormData({
      ...formData,
      [name]: newValue,
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

    // Only proceed if individual account type
    if (accountType !== "individual") {
      return;
    }
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
      // Prepare data for API according to RegisterRequest interface
      const registrationData = {
        firstName: formData.firstName.trim(),
        lastName: formData.lastName.trim(),
        email: formData.email.trim().toLowerCase(),
        password: formData.password,
        phone: formData.phone,
        agreedToTerms: formData.agreeTerms,
        userType: accountType,
      };

      const result = await register(registrationData).unwrap();

      // Handle successful registration
      console.log("Registration successful:", result);

      // Dispatch user data to Redux store
      dispatch(
        setUserInfo({
          _id: result.user.id,
          email: result.user.email,
          userType: result.user.userType,
          plan: result.user.plan,
          isGoogleUser: false,
          firstName: result.user.firstName,
          lastName: result.user.lastName,
        })
      );

      // Dispatch token to auth store
      dispatch(loginUser(result.token));

      navigate("/check-email", { state: { email: formData.email } });
    } catch (error: unknown) {
      console.error("Registration failed:", error);

      if (error && typeof error === "object" && "data" in error) {
        const apiError = error as {
          data?: { message?: string; errors?: FormErrors };
        };
        if (apiError.data?.message) {
          setErrors({ general: apiError.data.message });
        } else if (apiError.data?.errors) {
          // Handle field-specific errors from backend
          setErrors(apiError.data.errors);
        } else {
          setErrors({ general: "Registration failed. Please try again." });
        }
      } else if (error && typeof error === "object" && "message" in error) {
        const messageError = error as { message: string };
        setErrors({ general: messageError.message });
      } else {
        setErrors({ general: "Registration failed. Please try again." });
      }
    } finally {
      setIsSubmitting(false);
      setFormData({
        firstName: "",
        lastName: "",
        email: "",
        phone: "",
        password: "",
        agreeTerms: false,
      });
    }
  };

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

      {/* Left side - Sign up form */}
      <div className="flex-1 flex items-center justify-center p-4 lg:p-8 bg-white">
        <div className="w-full max-w-md">
          <div className="text-center">
            <h2 className="text-2xl font-semibold text-gray-900">Sign Up</h2>
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
              <form onSubmit={handleSubmit} className="space-y-6">
                {/* Account Type Toggle */}
                <div className="flex p-1 bg-gray-100 rounded-full">
                  <button
                    type="button"
                    className={`flex-1 py-2 px-4 text-sm font-medium rounded-full transition-colors ${
                      accountType === "individual"
                        ? "bg-black text-white"
                        : "text-gray-700 hover:bg-gray-200"
                    }`}
                    onClick={() => setAccountType("individual")}
                  >
                    Individual
                  </button>
                  <button
                    type="button"
                    className={`flex-1 py-2 px-4 text-sm font-medium rounded-full transition-colors ${
                      accountType === "teams"
                        ? "bg-black text-white"
                        : "text-gray-700 hover:bg-gray-200"
                    }`}
                    onClick={() => setAccountType("teams")}
                  >
                    Teams & Enterprises
                  </button>
                </div>

                {accountType === "teams" && (
                  <div className="bg-gray-100 border border-blue-200 rounded-xl p-6 text-center">
                    <div className="flex justify-center mb-4">
                      <div className="bg-gray-200 p-3 rounded-full">
                        <Mail className="w-6 h-6 text-black" />
                      </div>
                    </div>
                    <h3 className="text-lg font-semibold text-gray-900 mb-2">
                      Ready for Enterprise Solutions?
                    </h3>
                    <p className="text-gray-600 mb-4">
                      Get in touch with our team to discuss custom plans, volume
                      discounts, and enterprise features.
                    </p>
                    <div className="text-sm text-gray-700">
                      <p className="mb-2">Contact us at:</p>
                      <a
                        href="mailto:info@safeguardmedia.io"
                        className="text-black hover:text-gray-700 font-medium hover:underline"
                      >
                        info@safeguardmedia.io
                      </a>
                    </div>
                  </div>
                )}
                {accountType === "individual" && (
                  <>
                    {/* Google Sign In Button */}
                    <button
                      className="w-full h-12 flex items-center justify-center bg-white border border-gray-300 hover:bg-gray-50 rounded-xl font-normal text-sm"
                      type="button"
                      onClick={() => handleGoogleLogin()}
                    >
                      <svg className="w-5 h-5 mr-3" viewBox="0 0 24 24">
                        <path
                          fill="#4285F4"
                          d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"
                        />
                        <path
                          fill="#34A853"
                          d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"
                        />
                        <path
                          fill="#FBBC05"
                          d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"
                        />
                        <path
                          fill="#EA4335"
                          d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"
                        />
                      </svg>
                      Continue with Google
                    </button>

                    {/* Divider */}
                    <div className="relative">
                      <div className="absolute inset-0 flex items-center">
                        <div className="w-full border-t border-gray-200" />
                      </div>
                      <div className="relative flex justify-center text-sm">
                        <span className="px-4 bg-white text-gray-500">
                          Or Continue with Email
                        </span>
                      </div>
                    </div>
                    {/* Name Fields */}
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <div className="space-y-2">
                        <label
                          htmlFor="firstName"
                          className="block text-sm font-medium text-gray-700"
                        >
                          First name
                        </label>
                        <input
                          id="firstName"
                          name="firstName"
                          type="text"
                          placeholder="Enter your first name"
                          value={formData.firstName}
                          onChange={handleInputChange}
                          className={`w-full h-12 px-3 py-2 border rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 ${
                            errors.firstName
                              ? "border-red-300"
                              : "border-gray-300"
                          }`}
                        />
                        {errors.firstName && (
                          <p className="text-sm text-red-600">
                            {errors.firstName}
                          </p>
                        )}
                      </div>
                      <div className="space-y-2">
                        <label
                          htmlFor="lastName"
                          className="block text-sm font-medium text-gray-700"
                        >
                          Last name
                        </label>
                        <input
                          id="lastName"
                          name="lastName"
                          type="text"
                          placeholder="Enter your last name"
                          value={formData.lastName}
                          onChange={handleInputChange}
                          className={`w-full h-12 px-3 py-2 border rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 ${
                            errors.lastName
                              ? "border-red-300"
                              : "border-gray-300"
                          }`}
                        />
                        {errors.lastName && (
                          <p className="text-sm text-red-600">
                            {errors.lastName}
                          </p>
                        )}
                      </div>
                    </div>

                    {/* Email Field */}
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

                    {/* Phone Field */}
                    <div className="space-y-2">
                      <label
                        htmlFor="phone"
                        className="block text-sm font-medium text-gray-700"
                      >
                        Phone number
                      </label>
                      <PhoneInput
                        className={`w-full h-12 px-3 py-2 border rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 pr-10 ${
                          errors.phone ? "phone-input-error" : ""
                        }`}
                        placeholder="Enter phone number"
                        value={formData.phone}
                        onChange={(value) => {
                          setFormData({
                            ...formData,
                            phone: value || "",
                          });

                          // Clear error when user starts typing
                          if (errors.phone) {
                            setErrors({
                              ...errors,
                              phone: undefined,
                            });
                          }
                        }}
                        defaultCountry="NG" // Default to Nigeria
                        // className={`w-full ${
                        //   errors.phone ? "phone-input-error" : ""
                        // }`}
                        style={{
                          "--PhoneInputCountryFlag-aspectRatio": "1.33",
                          "--PhoneInputCountryFlag-height": "1em",
                          "--PhoneInputCountrySelectArrow-color": "#6b7280",
                          "--PhoneInput-color--focus": "#3b82f6",
                        }}
                        inputProps={{
                          className: `w-full h-12 px-3 py-2 border rounded-xl focus:outline-none ${
                            errors.phone ? "border-red-300" : "border-gray-300"
                          }`,
                        }}
                        countrySelectProps={{
                          className:
                            "h-12 border-r border-gray-300 rounded-l-xl focus:outline-none focus:ring-2 focus:ring-blue-500",
                        }}
                      />
                      {errors.phone && (
                        <p className="text-sm text-red-600">{errors.phone}</p>
                      )}
                    </div>

                    {/* Password Field */}
                    <div className="space-y-2">
                      <label
                        htmlFor="password"
                        className="block text-sm font-medium text-gray-700"
                      >
                        Password
                      </label>
                      <div className="relative">
                        <input
                          id="password"
                          name="password"
                          type={showPassword ? "text" : "password"}
                          placeholder="Enter your password"
                          value={formData.password}
                          onChange={handleInputChange}
                          className={`w-full h-12 px-3 py-2 border rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 pr-10 ${
                            errors.password
                              ? "border-red-300"
                              : "border-gray-300"
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
                        <p className="text-sm text-red-600">
                          {errors.password}
                        </p>
                      )}
                    </div>

                    {/* Terms Checkbox */}
                    <div className="space-y-2">
                      <div className="flex items-start">
                        <div className="flex items-center h-5">
                          <input
                            id="agreeTerms"
                            name="agreeTerms"
                            type="checkbox"
                            checked={formData.agreeTerms}
                            onChange={handleInputChange}
                            className={`h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded ${
                              errors.agreeTerms ? "border-red-300" : ""
                            }`}
                          />
                        </div>
                        <div className="ml-3 text-sm">
                          <label
                            htmlFor="agreeTerms"
                            className="font-medium text-gray-700"
                          >
                            Yes, I agree with the{" "}
                            <a
                              href="#"
                              className="text-blue-600 hover:underline"
                            >
                              Terms of Use
                            </a>
                          </label>
                        </div>
                      </div>
                      {errors.agreeTerms && (
                        <p className="text-sm text-red-600">
                          {errors.agreeTerms}
                        </p>
                      )}
                    </div>

                    {/* Register Button */}
                    <button
                      type="submit"
                      disabled={isSubmitting}
                      className="w-full h-12 bg-[#0F2FA3] hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed text-white font-medium rounded-[50px] flex items-center justify-center"
                    >
                      {isSubmitting ? (
                        <>
                          <Loader className="animate-spin -ml-1 mr-2 h-4 w-4" />
                          Registering...
                        </>
                      ) : (
                        "Register"
                      )}
                    </button>

                    {/* Sign In Link */}
                    <div className="text-center text-sm text-gray-600 cursor-pointer">
                      {"Already have an account? "}
                      <button
                        type="button"
                        className="text-blue-600 hover:text-blue-500 hover:underline font-medium"
                        onClick={() => navigate("/signin")}
                      >
                        Sign In
                      </button>
                    </div>
                  </>
                )}
              </form>
            </div>
          </div>
        </div>
      </div>

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

export default SignUp;
