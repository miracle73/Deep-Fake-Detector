import { useState, useEffect, useRef } from "react";
import { Card } from "../components/ui/card";
import { Button } from "../components/ui/button";
import { Check, Plus, X } from "lucide-react";
// import { Headphones, Play } from "lucide-react";
// import { DownwardArrow } from "../assets/svg";
import { FaArrowDownLong } from "react-icons/fa6";
import { AudioIcon, ImageIcon, VideoIcon } from "../assets/svg";
import { useNavigate } from "react-router-dom";
import TravelImage from "../assets/images/front-4.png";
import ScanImage from "../assets/images/scan-image.png";
import MediaHouseImage from "../assets/images/mediahouse-image.png";
import SafeguardMediaLogo from "../assets/images/SafeguardMedia8.svg";

export default function DeepfakeDetector() {
  const navigate = useNavigate();
  const [activeTab, setActiveTab] = useState("individual");
  const [selectedPlan, setSelectedPlan] = useState("pro");
  const imageRef = useRef<HTMLDivElement | null>(null);
  const textRef = useRef<HTMLHeadingElement | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [lineCoords, setLineCoords] = useState({ x1: 0, y1: 0, x2: 0, y2: 0 });
  const [expandedFAQ, setExpandedFAQ] = useState<number | null>(0);

  const updateLinePosition = () => {
    if (imageRef.current && textRef.current && containerRef.current) {
      const containerRect = containerRef.current.getBoundingClientRect();
      const imageRect = imageRef.current.getBoundingClientRect();
      const textRect = textRef.current.getBoundingClientRect();

      // Check if we're on large screens (lg breakpoint is 1024px)
      const isLargeScreen = window.innerWidth >= 1024;

      let imageX, imageY, textX, textY;

      if (isLargeScreen) {
        // Desktop: Point to top-center of image with padding
        imageX = imageRect.left + imageRect.width / 2 - containerRect.left;
        imageY = imageRect.top + 16 - containerRect.top; // 16px padding from top

        // Point to left edge of text
        textX = textRect.left - containerRect.left;
        textY = textRect.top + textRect.height / 2 - containerRect.top;
      } else {
        // Mobile/Tablet: Point to center of image
        imageX = imageRect.left + imageRect.width / 2 - containerRect.left;
        imageY = imageRect.top + imageRect.height / 2 - containerRect.top;

        // Point to center of text
        textX = textRect.left + textRect.width / 2 - containerRect.left;
        textY = textRect.top + textRect.height / 2 - containerRect.top;
      }

      setLineCoords({
        x1: imageX,
        y1: imageY,
        x2: textX,
        y2: textY,
      });
    }
  };

  useEffect(() => {
    updateLinePosition();

    const handleResize = () => {
      updateLinePosition();
    };

    window.addEventListener("resize", handleResize);

    // Update after images load
    const timer = setTimeout(updateLinePosition, 100);

    return () => {
      window.removeEventListener("resize", handleResize);
      clearTimeout(timer);
    };
  }, []);

  return (
    <div className="min-h-screen bg-white ">
      {/* Header */}
      <header className="">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center">
              <img
                src={SafeguardMediaLogo}
                alt="Safeguardmedia Logo"
                className="h-12 w-auto"
              />
              <span className="text-xl font-bold text-gray-900">
                Safeguardmedia
              </span>
            </div>
            <nav className="hidden md:flex items-center space-x-8">
              <a
                href="#"
                className="text-gray-600 hover:text-gray-900"
                onClick={() => navigate("/plans")}
              >
                Pricing
              </a>
              <a
                href="#faq-section"
                className="text-gray-600 hover:text-gray-900"
                onClick={(e) => {
                  e.preventDefault();
                  const faqSection = document.getElementById("faq-section");
                  if (faqSection) {
                    faqSection.scrollIntoView({ behavior: "smooth" });
                  }
                }}
              >
                FAQ
              </a>
              <a
                href="mailto:info@safeguardmedia.org"
                className="text-gray-600 hover:text-gray-900"
              >
                Support
              </a>

              <Button
                className="bg-[#0F2FA3] hover:bg-blue-700 text-white px-4 py-2 rounded-[50px]"
                onClick={() => navigate("/signin")}
              >
                Get Started
              </Button>
            </nav>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        {/* Hero Section */}
        <div className="text-center mb-12">
          <h1 className="text-4xl md:text-5xl font-bold text-gray-900 mb-4 leading-tight">
            Advanced Model for Reliable
            <br />
            Deepfake Detection.
          </h1>
          <p className="text-lg text-gray-600 mb-8 max-w-2xl mx-auto">
            Instantly analyze media for AI manipulation with our robust and
            easy-to-use platform. From individual checks to enterprise-scale
            integration, get the clarity you need.
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Button
              className="bg-[#0F2FA3] hover:bg-blue-700 text-white px-8 py-7 rounded-[50px] text-lg"
              onClick={() => navigate("/signup")}
            >
              Continue with Email
            </Button>
            <Button
              variant="outline"
              className="!px-10 !py-7 rounded-[50px] text-lg border-gray-300"
              onClick={() => {
                const faqSection = document.getElementById("faq-section");
                if (faqSection) {
                  faqSection.scrollIntoView({ behavior: "smooth" });
                }
              }}
            >
              Learn More
              <FaArrowDownLong />
            </Button>
          </div>
        </div>
        <div
          ref={containerRef}
          className="border border-[#8C8C8C] rounded-[50px] flex flex-col lg:flex-row items-start justify-between p-4 sm:p-6 mb-10 max-w-6xl w-full relative bg-white"
        >
          {/* SVG for connecting line */}
          <svg
            className="absolute inset-0 w-full h-full pointer-events-none z-10"
            style={{ overflow: "visible" }}
          >
            <defs>
              <marker
                id="arrowhead"
                markerWidth="10"
                markerHeight="7"
                refX="9"
                refY="3.5"
                orient="auto"
              >
                <polygon points="0 0, 10 3.5, 0 7" fill="#ef4444" />
              </marker>
            </defs>
            <line
              x1={lineCoords.x1}
              y1={lineCoords.y1}
              x2={lineCoords.x2}
              y2={lineCoords.y2}
              stroke="#ef4444"
              strokeWidth="2"
              markerEnd="url(#arrowhead)"
              className="drop-shadow-sm"
            />
            <circle
              cx={lineCoords.x1}
              cy={lineCoords.y1}
              r="4"
              fill="#ef4444"
              className="drop-shadow-sm"
            />
          </svg>

          {/* Image Section */}
          <div className="relative w-full lg:w-1/2 mb-6 lg:mb-0">
            <div
              ref={imageRef}
              className="rounded-2xl overflow-hidden shadow-md mx-auto lg:mx-0 h-[450px] lg:h-[400px]"
            >
              <img
                src={TravelImage}
                alt="Person with yellow suitcase at ancient ruins"
                className="w-full h-full object-cover"
              />
            </div>
          </div>

          {/* DF Results Card - Right Side */}
          <div className="w-full lg:w-1/2 lg:pl-8">
            {/* Header */}
            <div className="relative mb-6 ">
              <div className="flex flex-col items-center">
                <h1
                  ref={textRef}
                  className="text-lg sm:text-xl font-semibold text-center lg:text-left text-gray-900 mb-1"
                >
                  Deepfake Detected
                </h1>
                <p className="text-sm text-gray-600 text-center lg:text-left">
                  Similar match: <span className="font-medium">98%</span>
                </p>
              </div>
            </div>

            {/* DF Results Card */}
            <div className="bg-white rounded-xl border border-gray-200 overflow-hidden shadow-sm">
              {/* Header with DF Results and Deepfake badge */}
              <div className="bg-[#0F2FA3] text-white px-4 sm:px-6 py-3 sm:py-4 flex items-center justify-between">
                <span className="text-sm sm:text-base font-medium">
                  Safeguardmedia Results
                </span>
                <span className="bg-white text-red-600 px-2 sm:px-3 py-1 rounded-full text-xs sm:text-sm font-medium">
                  Deepfake
                </span>
              </div>

              {/* Results Content */}
              <div className="flex flex-col">
                {/* Confidence Score */}
                <div className="p-4 sm:p-6">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm sm:text-base text-gray-700 font-medium">
                      Confidence Score
                    </span>
                    <span className="text-2xl sm:text-3xl font-bold text-gray-900">
                      98%
                    </span>
                  </div>
                </div>

                {/* Divider */}
                <div className="border-t border-gray-200"></div>

                {/* Result Summary */}
                <div className="p-4 sm:p-6">
                  <h4 className="text-sm sm:text-base font-semibold text-[#020717] mb-3">
                    Result Summary:
                  </h4>
                  <p className="text-xs sm:text-sm text-[#020717] font-[300] leading-relaxed">
                    Our model analysis found significant indicators on this
                    media file strongly suggesting this media has been
                    manipulated using deepfake techniques.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* AI Content Information Section */}
        <div className="max-w-6xl mx-auto mt-40 max-md:mt-20 mb-12">
          <div className="text-center mb-12">
            <h2 className="text-3xl md:text-4xl font-bold text-gray-900 mb-4">
              Navigating the Age of AI-Generated Content
            </h2>
            <p className="text-lg text-gray-600 max-w-4xl mx-auto">
              In today's digital landscape, the core challenge is that deepfakes
              can severely damage your brand. However, our advanced AI analyzes
              media files for subtle artifacts and inconsistencies indicative of
              deepfakes.
            </p>
          </div>

          {/* Content Types Grid */}
          <div className="grid md:grid-cols-3 gap-8 mb-16">
            {/* Audio */}
            <div className="text-center">
              <div className="w-16 h-16 mx-auto mb-4  rounded-full flex items-center justify-center">
                <AudioIcon />
              </div>
              <h3 className="text-xl font-semibold text-gray-900 mb-3">
                Audio
              </h3>
              <p className="text-gray-600">
                Detect advanced audio scams, voice impersonation, and social
                engineering.
              </p>
            </div>

            {/* Video */}
            <div className="text-center">
              <div className="w-16 h-16 mx-auto mb-4 rounded-full flex items-center justify-center">
                <VideoIcon />
              </div>
              <h3 className="text-xl font-semibold text-gray-900 mb-3">
                Video
              </h3>
              <p className="text-gray-600">
                Uncover and block deepfake videos, synthetic impersonations, and
                visual social engineering.
              </p>
            </div>

            {/* Image */}
            <div className="text-center">
              <div className="w-16 h-16 mx-auto mb-4 rounded-full flex items-center justify-center">
                {/* <A className="w-8 h-8 text-purple-600" /> */}
                <ImageIcon />
              </div>
              <h3 className="text-xl font-semibold text-gray-900 mb-3">
                Image
              </h3>
              <p className="text-gray-600">
                Detect image manipulation, AI-generated forgeries, and visual
                identity fraud-instantly.
              </p>
            </div>
          </div>

          {/* For Individuals Section */}
          <div className="grid md:grid-cols-3 gap-10 items-start mb-20 mt-40 max-md:mt-20">
            <div className="md:col-span-2">
              <h3 className="text-2xl font-bold text-gray-900 mb-4">
                For Individuals
              </h3>
              <p className="text-gray-600 mb-8 max-w-2xl">
                This platform is for non-technical individuals who need an
                intuitive, easy-to-use platform for detecting deepfakes in
                various media formats (audio, video, images).
              </p>

              <div className="space-y-6">
                <div className="flex items-start space-x-3 pb-6 border-b border-gray-200">
                  <div className="flex-shrink-0 mt-1">
                    <svg
                      className="w-5 h-5 text-gray-700"
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M17 8l4 4m0 0l-4 4m4-4H3"
                      />
                    </svg>
                  </div>
                  <div>
                    <h4 className="font-semibold text-gray-900 mb-2">
                      Instant Analysis
                    </h4>
                    <p className="text-gray-600">
                      Upload your file and get results in moments. No technical
                      expertise needed.
                    </p>
                  </div>
                </div>

                <div className="flex items-start space-x-3 pb-6 border-b border-gray-200">
                  <div className="flex-shrink-0 mt-1">
                    <svg
                      className="w-5 h-5 text-gray-700"
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M17 8l4 4m0 0l-4 4m4-4H3"
                      />
                    </svg>
                  </div>
                  <div>
                    <h4 className="font-semibold text-gray-900 mb-2">
                      Supports All Media
                    </h4>
                    <p className="text-gray-600">
                      Check videos, audio clips, and images for manipulation.
                    </p>
                  </div>
                </div>

                <div className="flex items-start space-x-3">
                  <div className="flex-shrink-0 mt-1">
                    <svg
                      className="w-5 h-5 text-gray-700"
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M17 8l4 4m0 0l-4 4m4-4H3"
                      />
                    </svg>
                  </div>
                  <div>
                    <h4 className="font-semibold text-gray-900 mb-2">
                      Clear & Simple Results
                    </h4>
                    <p className="text-gray-600">
                      Understand detection outcomes with easy-to-read reports
                      and confidence scores.
                    </p>
                  </div>
                </div>
              </div>
            </div>

            <div className="md:col-span-1">
              <div className=" rounded-2xl h-[400px] w-full">
                <img
                  src={ScanImage}
                  alt="Scan Image"
                  className="w-full h-auto object-cover mx-auto lg:mx-0"
                />
              </div>
            </div>
          </div>

          {/* For Media Houses & Enterprise Section */}
          <div className="grid md:grid-cols-3 gap-10 items-start mb-20 mt-40 max-md:mt-20">
            <div className="md:col-span-1 order-1 md:order-2">
              <div className=" rounded-2xl h-[400px] w-full">
                <img
                  src={MediaHouseImage}
                  alt="Media House"
                  className="w-full h-auto object-cover mx-auto lg:mx-0"
                />
              </div>
            </div>

            <div className="md:col-span-2">
              <h3 className="text-2xl font-bold text-gray-900 mb-4">
                For Media Houses & Enterprise
              </h3>
              <p className="text-gray-600 mb-8 max-w-2xl">
                Large-scale media organizations requiring automated and robust
                deepfake detection capabilities.
              </p>

              <div className="space-y-6">
                <div className="flex items-start space-x-3 pb-6 border-b border-gray-200">
                  <div className="flex-shrink-0 mt-1">
                    <svg
                      className="w-5 h-5 text-gray-700"
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M17 8l4 4m0 0l-4 4m4-4H3"
                      />
                    </svg>
                  </div>
                  <div>
                    <h4 className="font-semibold text-gray-900 mb-2">
                      Robust API Integration
                    </h4>
                    <p className="text-gray-600">
                      Seamlessly connect our detection capabilities into your
                      existing CMS and workflows.
                    </p>
                  </div>
                </div>

                <div className="flex items-start space-x-3 pb-6 border-b border-gray-200">
                  <div className="flex-shrink-0 mt-1">
                    <svg
                      className="w-5 h-5 text-gray-700"
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M17 8l4 4m0 0l-4 4m4-4H3"
                      />
                    </svg>
                  </div>
                  <div>
                    <h4 className="font-semibold text-gray-900 mb-2">
                      Detailed Reporting & Analytics
                    </h4>
                    <p className="text-gray-600">
                      Track usage, model performance, and gain insights into
                      detected manipulations.
                    </p>
                  </div>
                </div>

                <div className="flex items-start space-x-3">
                  <div className="flex-shrink-0 mt-1">
                    <svg
                      className="w-5 h-5 text-gray-700"
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M17 8l4 4m0 0l-4 4m4-4H3"
                      />
                    </svg>
                  </div>
                  <div>
                    <h4 className="font-semibold text-gray-900 mb-2">
                      Bulk Media Analysis
                    </h4>
                    <p className="text-gray-600">
                      Process large volumes of content efficiently for real-time
                      deepfake detection.
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Pricing Section */}
          <div className="text-center mb-16 mt-40 max-md:mt-20">
            <h2 className="text-3xl md:text-4xl font-bold text-gray-900 mb-8">
              Explore Plans
            </h2>

            {/* Toggle Buttons */}
            <div className="flex justify-center mb-12">
              <div className="bg-gray-100 rounded-full p-1 flex">
                <button
                  className={`px-6 py-2 rounded-full text-sm font-medium transition-colors ${
                    activeTab === "individual"
                      ? "bg-[#0F2FA3] text-white"
                      : "text-gray-600 hover:text-gray-900"
                  }`}
                  onClick={() => setActiveTab("individual")}
                >
                  Individual
                </button>
                <button
                  className={`px-6 py-2 rounded-full text-sm font-medium transition-colors ${
                    activeTab === "teams"
                      ? "bg-[#0F2FA3] text-white"
                      : "text-gray-600 hover:text-gray-900"
                  }`}
                  onClick={() => setActiveTab("teams")}
                >
                  Teams & Enterprises
                </button>
              </div>
            </div>
            {activeTab === "individual" && (
              // Pricing Cards
              <div className="grid md:grid-cols-3 gap-6 max-w-5xl mx-auto">
                {/* Free Plan */}
                <Card
                  className={`p-6 rounded-2xl flex flex-col h-full cursor-pointer transition-all ${
                    selectedPlan === "free"
                      ? "border-2 border-yellow-400 shadow-lg"
                      : "border border-gray-200 hover:border-gray-300"
                  }`}
                  onClick={() => setSelectedPlan("free")}
                >
                  <div className="text-left flex-grow">
                    <h3 className="text-lg font-semibold text-gray-900 mb-2">
                      Free
                    </h3>
                    <div className="mb-4">
                      <span className="text-3xl font-bold text-gray-900">
                        $0
                      </span>
                    </div>
                    <p className="text-sm text-gray-600 mb-6">
                      Try Safeguardmedia for free
                    </p>

                    <div className="space-y-3 mb-8 flex-grow">
                      <div className="flex items-start space-x-3">
                        <Check className="w-5 h-5 text-green-500 flex-shrink-0 mt-0.5" />
                        <span className="text-sm text-gray-600">
                          Analyze media for up to 4,000 seconds each month.
                        </span>
                      </div>
                    </div>
                  </div>

                  <Button
                    className="w-full bg-[#0F2FA3] hover:bg-blue-700 text-white py-3 rounded-full mt-auto"
                    onClick={() => navigate("/plans")}
                  >
                    Subscribe
                  </Button>
                </Card>

                {/* Pro Plan */}
                <Card
                  className={`p-6 rounded-2xl relative flex flex-col h-full cursor-pointer transition-all ${
                    selectedPlan === "pro"
                      ? "border-2 border-yellow-400 shadow-lg"
                      : "border border-gray-200 hover:border-gray-300"
                  }`}
                  onClick={() => setSelectedPlan("pro")}
                >
                  <div className="text-left flex-grow">
                    <h3 className="text-lg font-semibold text-gray-900 mb-2">
                      Pro
                    </h3>
                    <div className="mb-4">
                      <span className="text-3xl font-bold text-gray-900">
                        $19
                      </span>
                    </div>
                    <p className="text-sm text-gray-600 mb-6">
                      per month, billed annually
                    </p>

                    <div className="space-y-3 mb-8 flex-grow">
                      <div className="flex items-start space-x-3">
                        <Check className="w-5 h-5 text-green-500 flex-shrink-0 mt-0.5" />
                        <span className="text-sm text-gray-600">
                          All features in Free mode.
                        </span>
                      </div>
                      <div className="flex items-start space-x-3">
                        <Check className="w-5 h-5 text-green-500 flex-shrink-0 mt-0.5" />
                        <span className="text-sm text-gray-600">
                          Analyze media for 15,000 seconds each month.
                        </span>
                      </div>
                    </div>
                  </div>

                  <Button
                    className="w-full bg-[#0F2FA3] hover:bg-blue-700 text-white py-3 rounded-full mt-auto"
                    onClick={() => navigate("/plans")}
                  >
                    Subscribe
                  </Button>
                </Card>

                {/* Max Plan */}
                <Card
                  className={`p-6 rounded-2xl flex flex-col h-full cursor-pointer transition-all ${
                    selectedPlan === "max"
                      ? "border-2 border-yellow-400 shadow-lg"
                      : "border border-gray-200 hover:border-gray-300"
                  }`}
                  onClick={() => setSelectedPlan("max")}
                >
                  <div className="text-left flex-grow">
                    <h3 className="text-lg font-semibold text-gray-900 mb-2">
                      Max
                    </h3>
                    <div className="mb-4">
                      <span className="text-3xl font-bold text-gray-900">
                        $49
                      </span>
                    </div>
                    <p className="text-sm text-gray-600 mb-6">
                      per month, billed annually
                    </p>

                    <div className="space-y-3 mb-8 flex-grow">
                      <div className="flex items-start space-x-3">
                        <Check className="w-5 h-5 text-green-500 flex-shrink-0 mt-0.5" />
                        <span className="text-sm text-gray-600">
                          All features in Pro mode.
                        </span>
                      </div>
                      <div className="flex items-start space-x-3">
                        <Check className="w-5 h-5 text-green-500 flex-shrink-0 mt-0.5" />
                        <span className="text-sm text-gray-600">
                          Analyze media for 45,000 seconds each month.
                        </span>
                      </div>
                    </div>
                  </div>

                  <Button
                    className="w-full bg-[#0F2FA3] hover:bg-blue-700 text-white py-3 rounded-full mt-auto"
                    onClick={() => navigate("/plans")}
                  >
                    Subscribe
                  </Button>
                </Card>
              </div>
            )}
            {/* Teams & Enterprises Coming Soon */}
            {activeTab === "teams" && (
              <div className="max-w-2xl mx-auto text-center py-6">
                <div className=" rounded-2xl p-12">
                  <h3 className="text-2xl font-bold text-gray-900 mb-4">
                    Coming Soon
                  </h3>
                  <p className="text-gray-600 text-lg">
                    We're working on enterprise solutions tailored for teams and
                    large organizations. Stay tuned for advanced features, bulk
                    processing, and custom integrations.
                  </p>
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
                        <strong>Images:</strong> Near real-time results with
                        high accuracy for manipulated facial features.
                      </li>
                      <li>
                        <strong>Videos:</strong> Slightly longer processing due
                        to frame-by-frame analysis, with strong detection
                        accuracy.
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
                  <a href="" className="text-gray-300 hover:text-white text-sm">
                    About Us
                  </a>
                </li>
                <li>
                  <a href="" className="text-gray-300 hover:text-white text-sm">
                    Our Technology
                  </a>
                </li>
                <li>
                  <a
                    href="#"
                    className="text-gray-300 hover:text-white text-sm"
                  >
                    Industries
                  </a>
                </li>
                <li>
                  <a
                    href="#"
                    className="text-gray-300 hover:text-white text-sm"
                  >
                    Help Center
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
                    href="/terms-and-conditions"
                    className="text-gray-300 hover:text-white text-sm"
                  >
                    Terms of Use
                  </a>
                </li>
                <li>
                  <a
                    href="/terms-and-conditions"
                    className="text-gray-300 hover:text-white text-sm"
                  >
                    Privacy Policies
                  </a>
                </li>
                <li>
                  <a
                    href="/terms-and-conditions"
                    className="text-gray-300 hover:text-white text-sm"
                  >
                    Responsible Disclosure Policy
                  </a>
                </li>
                <li>
                  <a
                    href="/terms-and-conditions"
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
                    href="https://www.instagram.com/safe_guard_media/"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-gray-300 hover:text-white text-sm"
                  >
                    Instagram
                  </a>
                </li>
                <li>
                  <a
                    href="https://x.com/safeguardmedia1"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-gray-300 hover:text-white text-sm"
                  >
                    X
                  </a>
                </li>
                <li>
                  <a
                    href="https://www.linkedin.com/company/safeguardmedia/about/?viewAsMember=true"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-gray-300 hover:text-white text-sm"
                  >
                    LinkedIn
                  </a>
                </li>
                <li>
                  <a
                    href="mailto:info@safeguardmedia.io"
                    className="text-gray-300 hover:text-white text-sm"
                  >
                    Contact Us: info@safeguardmedia.io
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
}
