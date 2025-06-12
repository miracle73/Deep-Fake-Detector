import type React from "react";
import { useState } from "react";
import { Eye, EyeOff } from "lucide-react";
import "../App.css";
import BackgroundImage from "../assets/images/signin-page-2.png";

function SignUp() {
  const [showPassword, setShowPassword] = useState(false);
  const [accountType, setAccountType] = useState("individual");
  const [formData, setFormData] = useState({
    firstName: "",
    lastName: "",
    email: "",
    phone: "",
    password: "",
    agreeTerms: false,
  });

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value, type, checked } = e.target;
    setFormData({
      ...formData,
      [name]: type === "checkbox" ? checked : value,
    });
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    console.log("Form submitted:", formData);
    // Add your registration logic here
  };

  return (
    <div className="min-h-screen flex flex-col lg:flex-row">
      {/* Mobile header - only visible on mobile */}
      <div className="lg:hidden p-6 bg-white">
        <div className="text-center max-md:text-left">
          <h1 className="text-xl font-bold text-gray-900">
            <span className="font-bold">Df</span>{" "}
            <span className="font-normal">Detector</span>
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
              <form onSubmit={handleSubmit} className="space-y-6">
                {/* Account Type Toggle */}
                <div className="flex p-1 bg-gray-100 rounded-full">
                  <button
                    type="button"
                    className={`flex-1 py-2 px-4 text-sm font-medium rounded-full transition-colors ${
                      accountType === "individual"
                        ? "bg-[#FBFBEF] text-[#020717]"
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
                      placeholder="placeholder"
                      value={formData.firstName}
                      onChange={handleInputChange}
                      className="w-full h-12 px-3 py-2 border border-gray-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                    />
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
                      placeholder="placeholder"
                      value={formData.lastName}
                      onChange={handleInputChange}
                      className="w-full h-12 px-3 py-2 border border-gray-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                    />
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
                    placeholder="placeholder"
                    value={formData.email}
                    onChange={handleInputChange}
                    className="w-full h-12 px-3 py-2 border border-gray-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  />
                </div>

                {/* Phone Field */}
                <div className="space-y-2">
                  <label
                    htmlFor="phone"
                    className="block text-sm font-medium text-gray-700"
                  >
                    Phone number
                  </label>
                  <input
                    id="phone"
                    name="phone"
                    type="tel"
                    placeholder="placeholder"
                    value={formData.phone}
                    onChange={handleInputChange}
                    className="w-full h-12 px-3 py-2 border border-gray-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  />
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
                      placeholder="XXXX XXXX"
                      value={formData.password}
                      onChange={handleInputChange}
                      className="w-full h-12 px-3 py-2 border border-gray-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 pr-10"
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
                </div>

                {/* Terms Checkbox */}
                <div className="flex items-start">
                  <div className="flex items-center h-5">
                    <input
                      id="agreeTerms"
                      name="agreeTerms"
                      type="checkbox"
                      checked={formData.agreeTerms}
                      onChange={handleInputChange}
                      className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                    />
                  </div>
                  <div className="ml-3 text-sm">
                    <label
                      htmlFor="agreeTerms"
                      className="font-medium text-gray-700"
                    >
                      Yes, I agree with the{" "}
                      <a href="#" className="text-blue-600 hover:underline">
                        Terms of Use
                      </a>
                    </label>
                  </div>
                </div>

                {/* Register Button */}
                <button
                  type="submit"
                  className="w-full h-12 bg-[#0F2FA3] hover:bg-blue-700 text-white font-medium rounded-[50px]"
                >
                  Register
                </button>

                {/* Sign In Link */}
                <div className="text-center text-sm text-gray-600">
                  {"Already have an account? "}
                  <a
                    href="#"
                    className="text-blue-600 hover:text-blue-500 hover:underline font-medium"
                  >
                    Sign In
                  </a>
                </div>
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
            <span className="font-bold">Df</span>{" "}
            <span className="font-normal">Detector</span>
          </div>
        </div>
      </div>
    </div>
  );
}

export default SignUp;
