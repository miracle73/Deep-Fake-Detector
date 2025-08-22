import { useState, useEffect } from "react";
import {
  Bell,
  LayoutGrid,
  Video,
  ImageIcon,
  AudioLines,
  Menu,
  X,
  MessageSquarePlus,
  // Download,
  // Trash2,
} from "lucide-react";
// import FourthImage from "../assets/images/fourthImage.png";
import { BackIcon } from "../assets/svg";
import SafeguardMediaLogo from "../assets/images/SafeguardMedia8.svg";
import { CiSettings } from "react-icons/ci";
import { useSelector } from "react-redux";
import type { RootState } from "../store/store";
import { useNavigate, useParams } from "react-router-dom";
import { useLocation } from "react-router-dom";
import type { DetectAnalyzeResponse } from "../services/apiService";

const ImageScreen = () => {
  const navigate = useNavigate();
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const storedUser = useSelector((state: RootState) => state.user.user);
  const { token } = useParams<{ token: string }>();
  const location = useLocation();
  const [analysisResult, setAnalysisResult] =
    useState<DetectAnalyzeResponse | null>(null);
  const [fileName, setFileName] = useState<string>("");
  const [fileSize, setFileSize] = useState<string>("");
  const [fileUrl, setFileUrl] = useState<string>("");
  const [analysisDate, setAnalysisDate] = useState<string>("");

  const getRiskAssessment = () => {
    if (!analysisResult)
      return {
        riskLevel: "Unknown",
        interpretation: "Analysis not available",
        action: "Please retry analysis",
        riskColor: "gray",
        gaugeColor: "#9CA3AF",
      };

    const realProb = analysisResult.data.real_probability;
    const deepfakeProb = analysisResult.data.deepfake_probability;

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

    if (analysisResult.data.is_deepfake) {
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
    return Math.round(analysisResult.data.confidence);
  };

  const handleBack = () => {
    // Clean up localStorage
    if (token) {
      localStorage.removeItem(`analysis_${token}`);
    }
    navigate("/dashboard");
  };
  // const handleDownloadReport = () => {
  //   // Handle download report
  //   console.log("Downloading report...");
  // };

  // const handleDeleteReport = () => {
  //   // Handle delete report
  //   console.log("Deleting report...");
  // };

  // useEffect(() => {
  //   if (!token) {
  //     navigate("/dashboard");
  //   }
  // }, [token, navigate]);

  useEffect(() => {
    if (!token) {
      navigate("/dashboard");
      return;
    }

    // Try to get data from location state first (from navigation)
    if (location.state?.analysisResult) {
      setAnalysisResult(location.state.analysisResult);
      setFileName(location.state.fileName || "Unknown File");
      setFileSize(location.state.fileSize || "Unknown Size");
      setFileUrl(location.state.fileUrl || ""); // Add this line
      setAnalysisDate(new Date().toLocaleString());
    } else {
      // Fallback: try to get from localStorage
      const storedResult = localStorage.getItem(`analysis_${token}`);
      if (storedResult) {
        try {
          const result = JSON.parse(storedResult);
          setAnalysisResult(result);
          setFileName("Analyzed Image");
          setFileSize("Unknown Size");
          setFileUrl(""); // No URL available from localStorage
          setAnalysisDate(new Date().toLocaleString());
        } catch (error) {
          console.error("Failed to parse stored analysis result:", error);
          navigate("/dashboard");
        }
      } else {
        // No data found, redirect to dashboard
        navigate("/dashboard");
      }
    }
  }, [token, navigate, location.state]);
  return (
    <div className={`min-h-screen bg-gray-50 overflow-x-hidden`}>
      {/* Full Width Header */}
      <header className="bg-white border-b border-gray-200 px-4 sm:px-6 py-4 w-full overflow-x-hidden">
        <div className="flex items-center justify-between min-w-0">
          <div className="flex items-center space-x-3 min-w-0 flex-1">
            {/* Mobile menu button */}
            <button
              className="lg:hidden p-2 text-gray-400 hover:text-gray-600 flex-shrink-0"
              onClick={() => setSidebarOpen(!sidebarOpen)}
            >
              <Menu className="w-5 h-5" />
            </button>
            <div className="flex items-center min-w-0">
              <img
                src={SafeguardMediaLogo}
                alt="Safeguardmedia Logo"
                className="h-12 w-auto flex-shrink-0"
              />
              <span className="text-xl max-lg:text-sm font-bold text-gray-900 truncate ml-2">
                Safeguardmedia
              </span>
            </div>
          </div>
          <div className="flex items-center space-x-2 sm:space-x-4 flex-shrink-0">
            {/* <div
              className="hidden sm:flex  bg-[#FBFBEF] gap-2 justify-between items-center"
              onClick={() => {
                navigate("/plans");
              }}
            >
              <button className="bg-[#0F2FA3] hover:bg-blue-700 text-white px-4 py-2 rounded-[30px] text-sm font-medium">
                Upgrade
              </button>
            </div> */}

            {/* Mobile upgrade button */}
            {/* <button
              className="sm:hidden bg-[#0F2FA3] hover:bg-blue-700 text-white px-3 py-1.5 rounded-[20px] text-xs font-medium"
              onClick={() => {
                navigate("/plans");
              }}
            >
              Upgrade
            </button> */}

            <button
              className="p-2 text-gray-400 hover:text-gray-600 bg-[#F6F7FE] rounded-[30px] border-[0.88px] border-[#8C8C8C] max-lg:hidden"
              onClick={() => {
                navigate("/notifications");
              }}
            >
              <Bell className="w-4 h-4 sm:w-5 sm:h-5" />
            </button>
            <button
              className="p-2 text-gray-400 hover:text-gray-600 bg-[#F6F7FE] rounded-[30px] border-[0.88px] border-[#8C8C8C] relative group"
              onClick={() => {
                navigate("/feedback");
              }}
              title="Send Feedback"
            >
              <MessageSquarePlus className="w-4 h-4 sm:w-5 sm:h-5" />
              {/* Optional: Add a tooltip */}
              <span className="absolute -bottom-8 left-1/2 transform -translate-x-1/2 bg-gray-800 text-white text-xs px-2 py-1 rounded opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap">
                Send Feedback
              </span>
            </button>

            {/* <div className="flex items-center space-x-2 cursor-pointer rounded-[30px]">
              <div className="w-6 h-6 sm:w-8 sm:h-8 bg-gray-300 rounded-full flex items-center justify-center">
                <span className="text-xs sm:text-sm font-medium text-gray-600">
                  U
                </span>
              </div>
              <span className="hidden sm:inline text-sm text-gray-700">
                Username
              </span>
       
            </div> */}
            <div
              className="flex items-center space-x-2 cursor-pointer rounded-[30px]"
              onClick={() => {
                navigate("/settings");
              }}
            >
              <div className="w-6 h-6 sm:w-8 sm:h-8 bg-gray-300 rounded-full flex items-center justify-center flex-shrink-0">
                <span className="text-xs sm:text-sm font-medium text-gray-600">
                  {storedUser.firstName
                    ? storedUser.firstName.charAt(0).toUpperCase()
                    : "U"}
                </span>
              </div>
              <span className="hidden sm:inline text-sm text-gray-700 truncate">
                {storedUser.firstName || "Username"}
              </span>
            </div>
          </div>
        </div>
      </header>

      {/* Mobile Sidebar Overlay */}
      {sidebarOpen && (
        <div className="lg:hidden fixed inset-0 z-50 flex overflow-hidden">
          <div
            className="fixed inset-0 bg-black bg-opacity-50"
            onClick={() => setSidebarOpen(false)}
          />
          <div className="relative flex flex-col w-64 max-w-[80vw] bg-white border-r border-gray-200 overflow-y-auto">
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
              <div
                className="flex items-center space-x-3 text-gray-400 cursor-not-allowed"
                // onClick={() => {
                //   navigate("/audio-detection");
                //   setSidebarOpen(false);
                // }}
              >
                <AudioLines className="w-6 h-6" />
                <span className="text-sm">Audio</span>
              </div>
              <div
                className="flex items-center space-x-3 text-gray-400  cursor-not-allowed"
                // onClick={() => {
                //   navigate("/video-detection");
                //   setSidebarOpen(false);
                // }}
              >
                <Video className="w-6 h-6" />
                <span className="text-sm">Video</span>
              </div>
              <div
                className="flex items-center space-x-3 text-gray-600 hover:text-blue-600 cursor-pointer"
                // onClick={() => {
                //   navigate("/image-detection");
                //   setSidebarOpen(false);
                // }}
              >
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
      <div className="flex overflow-x-hidden">
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
          <div
            className="flex flex-col items-center space-y-2 text-gray-400  cursor-not-allowed"
            // onClick={() => {
            //   navigate("/audio-detection");
            // }}
          >
            <AudioLines className="w-6 h-6" />
            <span className="text-xs">Audio</span>
          </div>
          <div
            className="flex flex-col items-center space-y-2 text-gray-400  cursor-not-allowed"
            // onClick={() => {
            //   navigate("/video-detection");
            // }}
          >
            <Video className="w-6 h-6" />
            <span className="text-xs">Video</span>
          </div>
          <div
            className="flex flex-col items-center space-y-2 text-gray-600 hover:text-blue-600 cursor-pointer"
            // onClick={() => {
            //   navigate("/image-detection");
            // }}
          >
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
        <div className="flex-1 flex flex-col overflow-hidden overflow-x-hidden min-w-0">
          {/* Upper Section: File Header + Right Sidebar */}
          {/* File Header Section - Full Width */}
          <div className="px-4 sm:px-6 pt-4 sm:pt-6 overflow-x-hidden">
            <div className=" p-2 sm:p-4 mb-2 sm:mb-6">
              {/* Header with Back button, filename and action buttons */}
              <div className="flex flex-col sm:flex-row sm:items-end sm:justify-between gap-4 mb-4 overflow-x-hidden">
                <div className="flex flex-col gap-2 min-w-0 flex-1">
                  <div className="flex items-center gap-2 overflow-x-hidden">
                    <button
                      onClick={handleBack}
                      className=" hover:bg-gray-100 rounded-lg transition-colors flex-shrink-0"
                    >
                      <BackIcon />
                    </button>
                    <div className="min-w-0 flex-1">
                      <h2 className="text-lg sm:text-xl font-semibold text-[#020717] truncate">
                        Back
                      </h2>
                    </div>
                  </div>
                  <div className="min-w-0">
                    <h2 className="text-lg sm:text-xl font-semibold text-gray-900 break-words">
                      {fileName}
                    </h2>
                  </div>
                  {/* File details */}
                  <div className="text-sm text-gray-600 overflow-x-auto">
                    <span>File size: {fileSize}</span>
                    <span className="mx-2">•</span>
                    <span>Date: {analysisDate}</span>
                  </div>
                </div>
                {/* Right side - Action buttons */}
                <div className="flex items-center space-x-2 sm:space-x-3 flex-shrink-0">
                  {/* <button
                    onClick={handleDownloadReport}
                    className="flex items-center space-x-2 px-3 sm:px-4 py-2 border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors"
                  >
                    <Download className="w-4 h-4 text-gray-600" />
                    <span className="text-sm font-medium text-gray-700">
                      Download Report
                    </span>
                  </button> */}
                  {/* <button
                    onClick={handleDeleteReport}
                    className="flex items-center space-x-2 px-3 sm:px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors"
                  >
                    <Trash2 className="w-4 h-4" />
                    <span className="text-sm font-medium">Delete Report</span>
                  </button> */}
                </div>
              </div>
            </div>
          </div>

          {/* Video Preview and DF Results Section - Side by Side */}
          <div className="flex flex-col lg:flex-row px-4 sm:px-6 gap-4 sm:gap-6 overflow-x-hidden">
            {/* Video Preview - Left Side */}
            <div className="w-full lg:w-2/3 min-w-0">
              <div className=" rounded-xl overflow-hidden">
                <img
                  src={fileUrl || "/placeholder.svg"}
                  alt="Video preview showing analysis result"
                  className="w-full h-auto max-w-full object-contain"
                />
              </div>
            </div>

            {/* DF Results Card - Right Side */}
            {/* DF Results Card - Right Side */}
            <div className="w-full lg:w-1/3 min-w-0">
              {/* DF Results Card */}
              <div className="bg-white rounded-xl border border-gray-200 overflow-hidden min-h-[50vh] flex flex-col">
                {/* Header with DF Results and status badge */}
                <div className="bg-[#0F2FA3] text-white px-4 sm:px-6 py-3 sm:py-4 flex items-center justify-between">
                  <span className="text-sm sm:text-base font-medium">
                    Safeguardmedia Results
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

                  {/* Analysis Details */}
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
                            {analysisResult.data.real_probability.toFixed(1)}%
                          </span>
                        </div>
                        <div className="flex justify-between items-center text-xs">
                          <span className="text-gray-600">
                            Deepfake Probability:
                          </span>
                          <span className="font-medium text-red-600">
                            {analysisResult.data.deepfake_probability.toFixed(
                              1
                            )}
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

          {/* Disclaimer Section */}
          <div className="px-4 sm:px-6 pb-4 sm:pb-6 mt-5 overflow-x-hidden">
            <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
              <p className="text-sm text-blue-800 break-words">
                <span className="font-medium">Disclaimer:</span> Results are
                provided for informational purposes only and users assume full
                responsibility for any decisions based on these analyses.
              </p>
            </div>
          </div>
        </div>
      </div>
      <button
        className="fixed bottom-6 right-6 z-50 p-3 bg-[#0F2FA3] hover:bg-blue-700 text-white rounded-full shadow-lg hover:shadow-xl transition-all duration-200 group"
        onClick={() => {
          navigate("/feedback");
        }}
      >
        <MessageSquarePlus className="w-5 h-5" />

        {/* Custom Tooltip */}
        <div className="absolute bottom-full right-0 mb-2 opacity-0 group-hover:opacity-100 transition-opacity duration-200 pointer-events-none">
          <div className="bg-gray-900 text-white text-xs px-3 py-2 rounded-lg whitespace-nowrap shadow-lg">
            Send Feedback
            {/* Tooltip Arrow */}
            <div className="absolute top-full right-4 w-0 h-0 border-l-4 border-r-4 border-t-4 border-transparent border-t-gray-900"></div>
          </div>
        </div>
      </button>
    </div>
  );
};

export default ImageScreen;
