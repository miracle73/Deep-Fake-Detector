import { useState } from "react";
import { Check, Plus, X } from "lucide-react";

const SubscriptionPlans = () => {
  const [selectedTab, setSelectedTab] = useState("Individual");

  return (
    <div className="min-h-screen bg-white">
      {/* Header */}
      <header className="bg-white">
        <div className="max-w-7xl mx-auto px-5 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center">
              <h1 className="text-xl font-bold text-gray-900">
                <span className="font-bold">Safeguard</span>{" "}
                <span className="font-normal">Media</span>
              </h1>
            </div>
            <nav className="hidden md:flex items-center space-x-8">
              <a
                href="#"
                className="text-gray-600 hover:text-gray-900 text-sm font-medium"
              >
                Features
              </a>
              <a
                href="#"
                className="text-gray-600 hover:text-gray-900 text-sm font-medium"
              >
                Pricing
              </a>
              <a
                href="#"
                className="text-gray-600 hover:text-gray-900 text-sm font-medium"
              >
                FAQs
              </a>
              <button className="bg-[#0F2FA3] hover:bg-blue-700 text-white px-4 py-2 rounded-lg text-sm font-medium">
                Get Started
              </button>
            </nav>
            {/* Mobile menu button */}
            <button className="md:hidden bg-[#0F2FA3] hover:bg-blue-700 text-white px-3 py-1.5 rounded-lg text-sm font-medium">
              Get Started
            </button>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-5 sm:px-6 lg:px-8 py-12">
        {/* Pricing Section */}
        <div className="text-center mb-12">
          <h2 className="text-3xl sm:text-4xl font-bold text-gray-900 mb-8">
            Explore Plans
          </h2>

          {/* Tab Toggle */}
          <div className="inline-flex bg-[#E6E6E6] rounded-full p-1 mb-12">
            <button
              onClick={() => setSelectedTab("Individual")}
              className={`px-6 py-2 rounded-full text-sm font-medium transition-colors ${
                selectedTab === "Individual"
                  ? "bg-[#FBFBEF] text-gray-900 shadow-sm"
                  : "text-gray-600 hover:text-gray-900"
              }`}
            >
              Individual
            </button>
            <button
              onClick={() => setSelectedTab("Teams & Enterprises")}
              className={`px-6 py-2 rounded-full text-sm font-medium transition-colors ${
                selectedTab === "Teams & Enterprises"
                  ? "bg-white text-gray-900 shadow-sm"
                  : "text-gray-600 hover:text-gray-900"
              }`}
            >
              Teams & Enterprises
            </button>
          </div>

          {/* Pricing Cards */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 max-w-5xl mx-auto">
            {/* Free Plan */}
            <div className="bg-white rounded-xl border border-gray-200 p-6 sm:p-8">
              <div className="text-left h-full flex flex-col">
                <h3 className="text-lg font-semibold text-gray-900 mb-2">
                  Free
                </h3>
                <div className="mb-4">
                  <span className="text-4xl font-bold text-gray-900">$0</span>
                </div>
                <p className="text-sm text-gray-600 mb-6">
                  Try Safeguard Media for free
                </p>

                <div className="space-y-3 mb-8 flex-1">
                  <div className="flex items-start space-x-3">
                    <Check className="w-5 h-5 text-green-500 mt-0.5 flex-shrink-0" />
                    <span className="text-sm text-gray-700">
                      Analyze media for up to 4,000 seconds each month.
                    </span>
                  </div>
                </div>

                <div className="w-full flex justify-center items-center">
                  <button className="bg-gray-900 hover:bg-gray-800 text-white py-3 px-6 rounded-[40px] font-medium transition-colors">
                    Get Started
                  </button>
                </div>
              </div>
            </div>

            {/* Pro Plan */}
            <div className="bg-white rounded-xl border border-gray-200 p-6 sm:p-8">
              <div className="text-left h-full flex flex-col">
                <h3 className="text-lg font-semibold text-gray-900 mb-2">
                  Pro
                </h3>
                <div className="mb-4">
                  <span className="text-4xl font-bold text-gray-900">$19</span>
                </div>
                <p className="text-sm text-gray-600 mb-6">
                  per month, billed annually
                </p>

                <div className="space-y-3 mb-8 flex-1">
                  <div className="flex items-start space-x-3">
                    <Check className="w-5 h-5 text-green-500 mt-0.5 flex-shrink-0" />
                    <span className="text-sm text-gray-700">
                      All features in Free mode.
                    </span>
                  </div>
                  <div className="flex items-start space-x-3">
                    <Check className="w-5 h-5 text-green-500 mt-0.5 flex-shrink-0" />
                    <span className="text-sm text-gray-700">
                      Analyze media for 15,000 seconds each month.
                    </span>
                  </div>
                </div>

                <div className="w-full flex justify-center items-center">
                  <button className="bg-gray-900 hover:bg-gray-800 text-white py-3 px-6 rounded-[40px] font-medium transition-colors">
                    Get Started
                  </button>
                </div>
              </div>
            </div>

            {/* Max Plan */}
            <div className="bg-white rounded-xl border border-gray-200 p-6 sm:p-8">
              <div className="text-left h-full flex flex-col">
                <h3 className="text-lg font-semibold text-gray-900 mb-2">
                  Max
                </h3>
                <div className="mb-4">
                  <span className="text-4xl font-bold text-gray-900">$49</span>
                </div>
                <p className="text-sm text-gray-600 mb-6">
                  per month, billed annually
                </p>

                <div className="space-y-3 mb-8 flex-1">
                  <div className="flex items-start space-x-3">
                    <Check className="w-5 h-5 text-green-500 mt-0.5 flex-shrink-0" />
                    <span className="text-sm text-gray-700">
                      All features in Pro mode.
                    </span>
                  </div>
                  <div className="flex items-start space-x-3">
                    <Check className="w-5 h-5 text-green-500 mt-0.5 flex-shrink-0" />
                    <span className="text-sm text-gray-700">
                      Analyze media for 45,000 seconds each month.
                    </span>
                  </div>
                </div>

                <div className="w-full flex justify-center items-center">
                  <button className="bg-gray-900 hover:bg-gray-800 text-white py-3 px-6 rounded-[40px] font-medium transition-colors">
                    Get Started
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* FAQ Section */}
        <div className="max-w-3xl mx-auto mb-20 mt-40 max-md:mt-20">
          <h2 className="text-3xl md:text-4xl font-bold text-gray-900 mb-12 text-center">
            Frequently Asked Questions
          </h2>

          <div className="space-y-6">
            {/* FAQ Item 1 - Expanded */}
            <div className="border-b border-gray-200 pb-6">
              <div className="flex justify-between items-center mb-4">
                <h3 className="text-lg font-semibold text-gray-900">
                  What is the "Robust Open-Source Deepfake Detector" project
                </h3>
                <button className="text-gray-400 hover:text-gray-600">
                  <X className="w-5 h-5" />
                </button>
              </div>
              <div className="text-gray-600">
                <p>
                  This project aims to develop an AI-based system designed to
                  detect deepfakes in video, audio, and image media. Its primary
                  goal is to help combat manipulated media and misinformation,
                  with a focus on providing a scalable, robust, and easy-to-use
                  solution.
                </p>
              </div>
            </div>

            {/* FAQ Item 2 - Collapsed */}
            <div className="border-b border-gray-200 pb-6">
              <div className="flex justify-between items-center">
                <h3 className="text-lg font-semibold text-gray-900">
                  Is this project open source, and can I contribute to its
                  development?
                </h3>
                <button className="text-gray-400 hover:text-gray-600">
                  <Plus className="w-5 h-5" />
                </button>
              </div>
            </div>

            {/* FAQ Item 3 - Collapsed */}
            <div className="border-b border-gray-200 pb-6">
              <div className="flex justify-between items-center">
                <h3 className="text-lg font-semibold text-gray-900">
                  How is the security of my uploaded media handled?
                </h3>
                <button className="text-gray-400 hover:text-gray-600">
                  <Plus className="w-5 h-5" />
                </button>
              </div>
            </div>

            {/* FAQ Item 4 - Collapsed */}
            <div className="border-b border-gray-200 pb-6">
              <div className="flex justify-between items-center">
                <h3 className="text-lg font-semibold text-gray-900">
                  How does Safeguard Media ensure it remains effective against
                  new deepfake techniques?
                </h3>
                <button className="text-gray-400 hover:text-gray-600">
                  <Plus className="w-5 h-5" />
                </button>
              </div>
            </div>
          </div>
        </div>
      </main>
      {/* Footer */}
      <footer className="bg-gray-900 text-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
          <div className="grid md:grid-cols-3 gap-8">
            {/* Company Column */}
            <div>
              <h3 className="text-sm font-medium text-gray-400 mb-4">
                Company
              </h3>
              <ul className="space-y-3">
                <li>
                  <a
                    href="#"
                    className="text-gray-300 hover:text-white text-sm"
                  >
                    About Us
                  </a>
                </li>
                <li>
                  <a
                    href="#"
                    className="text-gray-300 hover:text-white text-sm"
                  >
                    Careers
                  </a>
                </li>
                <li>
                  <a
                    href="#"
                    className="text-gray-300 hover:text-white text-sm"
                  >
                    Media
                  </a>
                </li>
              </ul>
            </div>

            {/* Cookies Column */}
            <div>
              <h3 className="text-sm font-medium text-gray-400 mb-4">
                Cookies
              </h3>
              <ul className="space-y-3">
                <li>
                  <a
                    href="#"
                    className="text-gray-300 hover:text-white text-sm"
                  >
                    Terms of Use
                  </a>
                </li>
                <li>
                  <a
                    href="#"
                    className="text-gray-300 hover:text-white text-sm"
                  >
                    Privacy Policies
                  </a>
                </li>
                <li>
                  <a
                    href="#"
                    className="text-gray-300 hover:text-white text-sm"
                  >
                    Responsible Disclosure Policy
                  </a>
                </li>
                <li>
                  <a
                    href="#"
                    className="text-gray-300 hover:text-white text-sm"
                  >
                    Compliance
                  </a>
                </li>
              </ul>
            </div>

            {/* Contacts Column */}
            <div>
              <h3 className="text-sm font-medium text-gray-400 mb-4">
                Contacts
              </h3>
              <ul className="space-y-3">
                <li>
                  <a
                    href="#"
                    className="text-gray-300 hover:text-white text-sm"
                  >
                    X
                  </a>
                </li>
                <li>
                  <a
                    href="#"
                    className="text-gray-300 hover:text-white text-sm"
                  >
                    Facebook
                  </a>
                </li>
                <li>
                  <a
                    href="mailto:iinfo@safeguardmedia.io"
                    className="text-gray-300 hover:text-white text-sm"
                  >
                    Mail: info@safeguardmedia.io
                  </a>
                </li>
              </ul>
            </div>
          </div>

          {/* Bottom Section */}
          <div className="border-t border-gray-700 mt-12 pt-8">
            <p className="text-center text-sm text-gray-400">
              © 2025, All Rights Reserved
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default SubscriptionPlans;
