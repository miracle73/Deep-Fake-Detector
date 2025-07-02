import { Check, Plus, X, Mail } from "lucide-react";
import { useState } from "react";
// import { useNavigate } from "react-router-dom";

const SubscriptionPlans = () => {
  const [selectedTab, setSelectedTab] = useState("Individual");
  // const navigate = useNavigate();

  const [expandedFAQ, setExpandedFAQ] = useState<number | null>(null);

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

          {selectedTab === "Teams & Enterprises" && (
            <div className="w-full flex justify-center">
              <div className="bg-gray-100 border w-1/2 border-blue-200 rounded-xl p-6 text-center">
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
            </div>
          )}

          {/* Pricing Cards */}
          {selectedTab === "Individual" && (
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
                    <span className="text-4xl font-bold text-gray-900">
                      $19
                    </span>
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
                    <span className="text-4xl font-bold text-gray-900">
                      $49
                    </span>
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
          )}
        </div>

        {/* FAQ Section */}
        <div
          id="faq-section"
          className="max-w-3xl mx-auto mb-20 mt-40 max-md:mt-20"
        >
          <h2 className="text-3xl md:text-4xl font-bold text-gray-900 mb-12 text-center">
            Frequently Asked Questions
          </h2>

          <div className="space-y-6">
            {/* FAQ Item 2 - File formats supported */}
            <div className="border-b border-gray-200 pb-6">
              <div className="flex justify-between items-center mb-4">
                <h3 className="text-lg font-semibold text-gray-900">
                  What file formats are supported?
                </h3>
                <button
                  className="text-gray-400 hover:text-gray-600"
                  onClick={() => setExpandedFAQ(expandedFAQ === 1 ? null : 1)}
                >
                  {expandedFAQ === 1 ? (
                    <X className="w-5 h-5" />
                  ) : (
                    <Plus className="w-5 h-5" />
                  )}
                </button>
              </div>
              {expandedFAQ === 1 && (
                <div className="text-gray-600">
                  <p className="mb-3">
                    Currently, SafeguardMedia supports the following formats:
                  </p>
                  <ul className="space-y-2 ml-4">
                    <li>
                      <strong>Videos:</strong> MP4, AVI, MOV
                    </li>
                    <li>
                      <strong>Images:</strong> JPEG, PNG, WEBP
                    </li>
                    <li>
                      <strong>Audio:</strong> MP3, WAV, AAC
                    </li>
                  </ul>
                  <p className="mt-3">
                    We are continuously expanding compatibility based on user
                    feedback and technological needs.
                  </p>
                </div>
              )}
            </div>

            {/* FAQ Item 3 - Content types detection */}
            <div className="border-b border-gray-200 pb-6">
              <div className="flex justify-between items-center mb-4">
                <h3 className="text-lg font-semibold text-gray-900">
                  Does SafeguardMedia detect deepfakes in videos, audio, and
                  images?
                </h3>
                <button
                  className="text-gray-400 hover:text-gray-600"
                  onClick={() => setExpandedFAQ(expandedFAQ === 2 ? null : 2)}
                >
                  {expandedFAQ === 2 ? (
                    <X className="w-5 h-5" />
                  ) : (
                    <Plus className="w-5 h-5" />
                  )}
                </button>
              </div>
              {expandedFAQ === 2 && (
                <div className="text-gray-600">
                  <p className="mb-3">
                    Yes. SafeguardMedia is built to detect manipulation across{" "}
                    three primary content types:
                  </p>
                  <ul className="space-y-2 ml-4">
                    <li>
                      <strong>Videos</strong> (frame-by-frame and audio sync
                      analysis)
                    </li>
                    <li>
                      <strong>Audio</strong> (voice cloning and manipulation
                      detection)
                    </li>
                    <li>
                      <strong>Static images</strong> (facial and pixel-level
                      inconsistencies)
                    </li>
                  </ul>
                </div>
              )}
            </div>

            {/* FAQ Item 4 - Performance across content types */}
            <div className="border-b border-gray-200 pb-6">
              <div className="flex justify-between items-center mb-4">
                <h3 className="text-lg font-semibold text-gray-900">
                  How does performance vary across different content types?
                </h3>
                <button
                  className="text-gray-400 hover:text-gray-600"
                  onClick={() => setExpandedFAQ(expandedFAQ === 3 ? null : 3)}
                >
                  {expandedFAQ === 3 ? (
                    <X className="w-5 h-5" />
                  ) : (
                    <Plus className="w-5 h-5" />
                  )}
                </button>
              </div>
              {expandedFAQ === 3 && (
                <div className="text-gray-600">
                  <p className="mb-3">
                    Performance may vary depending on content quality, length,
                    and type:
                  </p>
                  <ul className="space-y-2 ml-4">
                    <li>
                      <strong>Images:</strong> Near real-time results with high
                      accuracy for manipulated facial features.
                    </li>
                    <li>
                      <strong>Videos:</strong> Slightly longer processing due to
                      frame-by-frame analysis, with strong detection accuracy.
                    </li>
                    <li>
                      <strong>Audio:</strong> Accuracy improves with longer
                      samples (over 5 seconds) and clear speech patterns.
                    </li>
                  </ul>
                  <p className="mt-3">
                    Regardless of format, SafeguardMedia provides a confidence
                    score and visual explanation to help users understand the
                    results.
                  </p>
                </div>
              )}
            </div>

            {/* FAQ Item 5 - Security of uploaded media */}
            <div className="border-b border-gray-200 pb-6">
              <div className="flex justify-between items-center mb-4">
                <h3 className="text-lg font-semibold text-gray-900">
                  How is the security of my uploaded media handled?
                </h3>
                <button
                  className="text-gray-400 hover:text-gray-600"
                  onClick={() => setExpandedFAQ(expandedFAQ === 4 ? null : 4)}
                >
                  {expandedFAQ === 4 ? (
                    <X className="w-5 h-5" />
                  ) : (
                    <Plus className="w-5 h-5" />
                  )}
                </button>
              </div>
              {expandedFAQ === 4 && (
                <div className="text-gray-600">
                  <p>
                    SafeguardMedia prioritizes your privacy and data security.
                    All uploaded content is processed through encrypted
                    channels, and no media is stored without user consent. Our
                    infrastructure complies with industry standards for
                    cybersecurity and data protection to ensure your files
                    remain confidential and secure.
                  </p>
                </div>
              )}
            </div>

            {/* FAQ Item 6 - Staying effective against evolving techniques */}
            <div className="border-b border-gray-200 pb-6">
              <div className="flex justify-between items-center mb-4">
                <h3 className="text-lg font-semibold text-gray-900">
                  How does SafeguardMedia stay effective against evolving
                  deepfake techniques?
                </h3>
                <button
                  className="text-gray-400 hover:text-gray-600"
                  onClick={() => setExpandedFAQ(expandedFAQ === 5 ? null : 5)}
                >
                  {expandedFAQ === 5 ? (
                    <X className="w-5 h-5" />
                  ) : (
                    <Plus className="w-5 h-5" />
                  )}
                </button>
              </div>
              {expandedFAQ === 5 && (
                <div className="text-gray-600">
                  <p>
                    We continuously update our detection algorithms using
                    real-world datasets and the latest advancements in AI
                    research. Our system is built on adaptive learning
                    principles, allowing it to detect even the most recent and
                    sophisticated forms of synthetic media, including
                    adversarially crafted deepfakes.
                  </p>
                </div>
              )}
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
