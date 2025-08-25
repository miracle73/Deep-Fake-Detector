import { useState, useEffect, useRef } from "react";
import {
  Bell,
  LayoutGrid,
  Video,
  ImageIcon,
  AudioLines,
  Menu,
  X,
  Shield,
} from "lucide-react";
import { BackIcon } from "../assets/svg";
import { useNavigate, useParams, useLocation } from "react-router-dom";
import { useSelector } from "react-redux";
import type { RootState } from "../store/store";
import SafeguardMediaLogo from "../assets/images/SafeguardMedia8.svg";
import { CiSettings } from "react-icons/ci";

interface AnalysisResult {
  confidence: number;
  deepfake_probability: number;
  filename: string;
  is_deepfake: boolean;
  predicted_class: string;
  real_probability: number;
  segments_processed: number;
  total_duration: number;
}

const AudioScreen = () => {
  const navigate = useNavigate();
  const { token } = useParams();
  const location = useLocation();
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(
    null
  );
  const [fileName, setFileName] = useState("");
  const [fileSize, setFileSize] = useState("");
  const [fileUrl, setFileUrl] = useState("");
  const [uploadDate, setUploadDate] = useState("");

  const audioRef = useRef<HTMLAudioElement>(null);
  const storedUser = useSelector((state: RootState) => state.user.user);

  // Load analysis data on component mount
  useEffect(() => {
    if (token) {
      // First try to get data from localStorage
      const storedData = localStorage.getItem(`analysis_${token}`);
      if (storedData) {
        const parsedData = JSON.parse(storedData);
        console.log(parsedData.fileUrl, 5400);
        setAnalysisResult(parsedData.data);
        setFileName(parsedData.fileName || "Unknown Audio File");
        setFileSize(parsedData.fileSize || "Unknown size");
        setFileUrl(parsedData.fileUrl || "");

        setUploadDate(new Date().toLocaleDateString());
      }
      // If no stored data, try to get from location state
      if (location.state) {
        const { originalFile } = location.state;
        console.log(location.state, 4500);

        if (typeof originalFile === "string") {
          setFileUrl(originalFile);
        } else {
          const audioFile = URL.createObjectURL(originalFile);
          setFileUrl(audioFile);
        }

        setUploadDate(new Date().toLocaleDateString());
      }
    }
  }, [token, location.state]);

  const handleBack = () => {
    navigate("/dashboard");
  };

  // New functions for enhanced results section (similar to VideoScreen)
  const getRiskAssessment = () => {
    if (!analysisResult)
      return {
        riskLevel: "Unknown",
        interpretation: "Analysis not available",
        action: "Please retry analysis",
        riskColor: "gray",
        gaugeColor: "#9CA3AF",
      };

    const realProb = analysisResult?.real_probability || 0;
    const deepfakeProb = analysisResult?.deepfake_probability || 0;

    if (realProb >= 90 && deepfakeProb <= 10) {
      return {
        riskLevel: "Low",
        interpretation: "Very Likely Real",
        action: "Accept as authentic. Manual review optional.",
        riskColor: "green",
        gaugeColor: "#10B981",
      };
    } else if (realProb >= 70 && deepfakeProb <= 29) {
      return {
        riskLevel: "Medium",
        interpretation: "Likely Real, Some Risk",
        action: "Review manually if content is sensitive or high-stakes.",
        riskColor: "yellow",
        gaugeColor: "#F59E0B",
      };
    } else if (realProb >= 50 && deepfakeProb <= 49) {
      return {
        riskLevel: "Medium-High",
        interpretation: "Ambiguous / Uncertain",
        action:
          "Manual verification strongly recommended. Consider secondary tools.",
        riskColor: "orange",
        gaugeColor: "#F97316",
      };
    } else if (realProb >= 30 && deepfakeProb <= 69) {
      return {
        riskLevel: "High",
        interpretation: "Likely Deepfake, But Not Conclusive",
        action: "Treat cautiously. Manual review required; possibly reject.",
        riskColor: "red",
        gaugeColor: "#EF4444",
      };
    } else {
      return {
        riskLevel: "Very High",
        interpretation: "Very Likely Deepfake",
        action: "Reject or flag. Notify relevant stakeholders.",
        riskColor: "red",
        gaugeColor: "#DC2626",
      };
    }
  };

  const getResultStatus = () => {
    if (!analysisResult)
      return { text: "Unknown", color: "gray", bgColor: "bg-gray-100" };

    if (analysisResult?.is_deepfake) {
      return {
        text: "Deepfake",
        color: "red",
        bgColor: "bg-red-600",
        textColor: "text-red-600",
      };
    } else {
      return {
        text: "Authentic",
        color: "green",
        bgColor: "bg-green-600",
        textColor: "text-green-600",
      };
    }
  };

  const getConfidenceScore = () => {
    if (!analysisResult) return 0;
    return Math.round(analysisResult?.confidence || 0);
  };

  return (
    <div className={`min-h-screen bg-gray-50`}>
      {/* Hidden Audio Element */}
      {fileUrl && <audio ref={audioRef} src={fileUrl} preload="metadata" />}

      {/* Header - Same as original */}
      <header className="bg-white border-b border-gray-200 px-4 sm:px-6 py-4 w-full">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <button
              className="lg:hidden p-2 text-gray-400 hover:text-gray-600"
              onClick={() => setSidebarOpen(!sidebarOpen)}
            >
              <Menu className="w-5 h-5" />
            </button>
            <div className="flex items-center">
              <img
                src={SafeguardMediaLogo}
                alt="Safeguardmedia Logo"
                className="h-12 w-auto"
              />
              <span className="text-xl max-lg:text-sm font-bold text-gray-900">
                Safeguardmedia
              </span>
            </div>
          </div>
          <div className="flex items-center space-x-2 sm:space-x-4">
            <button
              className="p-2 text-gray-400 hover:text-gray-600 bg-[#F6F7FE] rounded-[30px] border-[0.88px] border-[#8C8C8C] max-lg:hidden"
              onClick={() => {
                navigate("/notifications");
              }}
            >
              <Bell className="w-4 h-4 sm:w-5 sm:h-5" />
            </button>

            {storedUser.email === "info@safeguardmedia.org" && (
              <button
                className="p-2 text-gray-400 hover:text-gray-600 bg-[#F6F7FE] rounded-[30px] border-[0.88px] border-[#8C8C8C]"
                onClick={() => navigate("/admin/feedback")}
                title="Admin Panel"
              >
                <Shield className="w-4 h-4 sm:w-5 sm:h-5" />
              </button>
            )}

            <div
              className="flex items-center space-x-2 cursor-pointer rounded-[30px]"
              onClick={() => {
                navigate("/settings");
              }}
            >
              <div className="w-6 h-6 sm:w-8 sm:h-8 bg-gray-300 rounded-full flex items-center justify-center">
                <span className="text-xs sm:text-sm font-medium text-gray-600">
                  {storedUser.firstName
                    ? storedUser.firstName.charAt(0).toUpperCase()
                    : "U"}
                </span>
              </div>
              <span className="hidden sm:inline text-sm text-gray-700">
                {storedUser.firstName || "Username"}
              </span>
            </div>
          </div>
        </div>
      </header>

      {/* Mobile Sidebar - Same as original */}
      {sidebarOpen && (
        <div className="lg:hidden fixed inset-0 z-50 flex">
          <div
            className="fixed inset-0 bg-black bg-opacity-50"
            onClick={() => setSidebarOpen(false)}
          />
          <div className="relative flex flex-col w-64 bg-white border-r border-gray-200">
            <div className="flex items-center justify-between p-4 border-b border-gray-200">
              <h2 className="text-lg font-semibold text-gray-900">Menu</h2>
              <button
                onClick={() => setSidebarOpen(false)}
                className="p-2 text-gray-400 hover:text-gray-600"
              >
                <X className="w-5 h-5" />
              </button>
            </div>
            <div className="flex-1 py-6 space-y-6 px-4">
              <div
                className="flex items-center space-x-3 text-gray-600 hover:text-blue-600 cursor-pointer"
                onClick={() => {
                  navigate("/dashboard");
                  setSidebarOpen(false);
                }}
              >
                <LayoutGrid className="w-6 h-6" />
                <span className="text-sm">Dashboard</span>
              </div>
              <div className="flex items-center space-x-3 text-blue-600">
                <AudioLines className="w-6 h-6" />
                <span className="text-sm">Audio</span>
              </div>
              <div className="flex items-center space-x-3 text-gray-400 cursor-not-allowed">
                <Video className="w-6 h-6" />
                <span className="text-sm">Video</span>
              </div>
              <div className="flex items-center space-x-3 text-gray-400 cursor-not-allowed">
                <ImageIcon className="w-6 h-6" />
                <span className="text-sm">Image</span>
              </div>
              <div
                className="flex items-center space-x-3 text-gray-600 hover:text-blue-600 cursor-pointer"
                onClick={() => {
                  navigate("/settings");
                  setSidebarOpen(false);
                }}
              >
                <CiSettings className="w-6 h-6" />
                <span className="text-xs">Settings</span>
              </div>
              <div
                className="flex items-center space-x-3 text-gray-600 hover:text-blue-600 cursor-pointer"
                onClick={() => {
                  navigate("/notifications");
                  setSidebarOpen(false);
                }}
              >
                <Bell className="w-6 h-6" />
                <span className="text-xs">Notifications</span>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Content Area with Sidebar */}
      <div className="flex">
        {/* Desktop Sidebar */}
        <div className="hidden lg:flex w-24 bg-white border-r border-gray-200 flex-col items-center py-6 space-y-8 min-h-[calc(100vh-73px)]">
          <div
            className="flex flex-col items-center space-y-2 text-gray-600 hover:text-blue-600 cursor-pointer"
            onClick={() => {
              navigate("/dashboard");
            }}
          >
            <LayoutGrid className="w-6 h-6" />
            <span className="text-xs">Dashboard</span>
          </div>
          <div className="flex flex-col items-center space-y-2 text-blue-600">
            <AudioLines className="w-6 h-6" />
            <span className="text-xs">Audio</span>
          </div>
          <div className="flex flex-col items-center space-y-2 text-gray-400 cursor-not-allowed">
            <Video className="w-6 h-6" />
            <span className="text-xs">Video</span>
          </div>
          <div className="flex flex-col items-center space-y-2 text-gray-400 cursor-not-allowed">
            <ImageIcon className="w-6 h-6" />
            <span className="text-xs">Image</span>
          </div>
          <div
            className="flex flex-col items-center space-y-2 text-gray-600 hover:text-blue-600 cursor-pointer"
            onClick={() => {
              navigate("/settings");
            }}
          >
            <CiSettings className="w-6 h-6" />
            <span className="text-xs">Settings</span>
          </div>
          <div
            className="flex flex-col items-center space-y-2 text-gray-600 hover:text-blue-600 cursor-pointer"
            onClick={() => {
              navigate("/notifications");
            }}
          >
            <Bell className="w-6 h-6" />
            <span className="text-xs">Notifications</span>
          </div>
        </div>

        {/* Main Content Container */}
        <div className="flex-1 flex flex-col overflow-hidden">
          {/* File Header Section - Full Width */}
          <div className="px-4 sm:px-6 pt-4 sm:pt-6">
            <div className=" p-2 sm:p-4 mb-2 sm:mb-6">
              {/* Header with Back button, filename and action buttons */}
              <div className="flex flex-col sm:flex-row sm:items-end sm:justify-between gap-4 mb-4">
                <div className="flex flex-col gap-2">
                  <div className="flex items-center gap-2">
                    <button
                      onClick={handleBack}
                      className=" hover:bg-gray-100 rounded-lg transition-colors"
                    >
                      <BackIcon />
                    </button>
                    <div>
                      <h2 className="text-lg sm:text-xl font-semibold text-[#020717]">
                        Back
                      </h2>
                    </div>
                  </div>
                  <div>
                    <h2 className="text-lg sm:text-xl font-semibold text-gray-900">
                      {fileName}
                    </h2>
                  </div>
                  {/* File details */}
                  <div className="text-sm text-gray-600">
                    <span>File size: {fileSize}</span>
                    <span className="mx-2">•</span>
                    <span>Date: {uploadDate}</span>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Audio Visualization and Results Section - Side by Side */}
          <div className="flex flex-col lg:flex-row px-4 sm:px-6 gap-4 sm:gap-6">
            {/* Audio Visualization - Left Side */}
            {/* Audio Visualization - Left Side */}
            <div className="w-full lg:w-2/3">
              <div className="bg-white rounded-xl border border-gray-200 p-6">
                {/* Audio player */}
                <div className="h-48 sm:h-64 bg-gray-50 rounded-lg flex items-center justify-center px-4">
                  {fileUrl ? (
                    <audio
                      controls
                      className="w-full max-w-md"
                      src={fileUrl}
                      preload="metadata"
                    >
                      Your browser does not support the audio element.
                    </audio>
                  ) : (
                    <div className="text-center">
                      <AudioLines className="w-16 h-16 text-gray-400 mx-auto mb-4" />
                      <p className="text-gray-500 text-sm">
                        Audio Waveform Visualization
                      </p>
                      <p className="text-gray-400 text-xs mt-1">
                        No audio file available
                      </p>
                    </div>
                  )}
                </div>
              </div>
            </div>
            {/* Enhanced Results Card - Right Side (Similar to VideoScreen) */}
            {/* Enhanced Results Card - Right Side (matching ImageScreen) */}
            <div className="w-full lg:w-1/3">
              <div className="bg-white rounded-xl border border-gray-200 overflow-hidden min-h-[50vh] flex flex-col">
                {/* Header with Results and status badge */}
                <div className="bg-[#0F2FA3] text-white px-4 sm:px-6 py-3 sm:py-4 flex items-center justify-between">
                  <span className="text-sm sm:text-base font-medium">
                    Safeguard Media Results
                  </span>
                  <span
                    className={`bg-white ${
                      getResultStatus().textColor
                    } px-3 py-1 rounded-full text-xs sm:text-sm font-medium`}
                  >
                    {getResultStatus().text}
                  </span>
                </div>

                {/* Results Content */}
                <div className="flex-1 flex flex-col">
                  {/* Confidence Gauge */}
                  <div className="p-4 sm:p-6 flex flex-col items-center justify-center">
                    <div className="relative w-32 h-32 mb-4">
                      {/* Gauge Background */}
                      <svg className="w-full h-full" viewBox="0 0 120 120">
                        {/* Background Circle */}
                        <circle
                          cx="60"
                          cy="60"
                          r="50"
                          fill="none"
                          stroke="#E5E7EB"
                          strokeWidth="10"
                        />
                        {/* Progress Circle */}
                        <circle
                          cx="60"
                          cy="60"
                          r="50"
                          fill="none"
                          stroke={getRiskAssessment().gaugeColor}
                          strokeWidth="10"
                          strokeLinecap="round"
                          strokeDasharray={`${
                            (getConfidenceScore() / 100) * 314.16
                          } 314.16`}
                          transform="rotate(-90 60 60)"
                          className="transition-all duration-1000 ease-out"
                          style={{
                            filter:
                              getConfidenceScore() === 0
                                ? "opacity(0)"
                                : "opacity(1)",
                          }}
                        />
                      </svg>
                      {/* Center Score */}
                      <div className="absolute inset-0 flex items-center justify-center">
                        <div className="text-center">
                          <div className="text-2xl sm:text-3xl font-bold text-gray-900">
                            {getConfidenceScore()}%
                          </div>
                          <div className="text-xs text-gray-500">
                            Confidence
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Risk Assessment */}
                  <div className="px-4 sm:px-6 pb-4">
                    <div className="space-y-4">
                      {/* Risk Level */}
                      <div className="flex items-center justify-between">
                        <span className="text-sm text-gray-600">
                          Risk Level:
                        </span>
                        <span
                          className={`px-3 py-1 rounded-full text-xs font-medium ${
                            getRiskAssessment().riskLevel === "Low"
                              ? "bg-green-100 text-green-800"
                              : getRiskAssessment().riskLevel === "Medium"
                              ? "bg-yellow-100 text-yellow-800"
                              : getRiskAssessment().riskLevel === "Medium-High"
                              ? "bg-orange-100 text-orange-800"
                              : getRiskAssessment().riskLevel === "High"
                              ? "bg-red-100 text-red-800"
                              : "bg-red-100 text-red-800"
                          }`}
                        >
                          {getRiskAssessment().riskLevel}
                        </span>
                      </div>

                      {/* Interpretation */}
                      <div>
                        <span className="text-sm text-gray-600 block mb-2">
                          Interpretation:
                        </span>
                        <p className="text-sm font-medium text-gray-900 break-words">
                          {getRiskAssessment().interpretation}
                        </p>
                      </div>
                    </div>
                  </div>

                  {/* Divider */}
                  <div className="border-t border-gray-200"></div>

                  {/* Analysis Details - Modified to match ImageScreen style */}
                  {analysisResult && (
                    <div className="border-t border-gray-200 p-4 sm:p-6">
                      <h4 className="text-sm font-semibold text-[#020717] mb-3">
                        Confidence Breakdown:
                      </h4>
                      <div className="space-y-2">
                        <div className="flex justify-between items-center text-xs">
                          <span className="text-gray-600">
                            Real Probability:
                          </span>
                          <span className="font-medium text-green-600">
                            {analysisResult?.real_probability?.toFixed(1) ||
                              "0.0"}
                            %
                          </span>
                        </div>
                        <div className="flex justify-between items-center text-xs">
                          <span className="text-gray-600">
                            Deepfake Probability:
                          </span>
                          <span className="font-medium text-red-600">
                            {analysisResult?.deepfake_probability?.toFixed(1) ||
                              "0.0"}
                            %
                          </span>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>

          {/* Audio Analysis Interface - Updated to use real audio */}
          <div className="px-2 sm:px-4 md:px-6 py-4 sm:py-6">
            <div className="flex flex-col lg:flex-row gap-4 sm:gap-6">
              {/* Results Explanation Panel - Right Side */}
            </div>
          </div>

          {/* Disclaimer Section */}
          <div className="px-4 sm:px-6 pb-4 sm:pb-6">
            <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
              <p className="text-sm text-blue-800">
                <span className="font-medium">Disclaimer:</span> Results are
                provided for informational purposes only and users assume full
                responsibility for any decisions based on these analyses.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AudioScreen;
