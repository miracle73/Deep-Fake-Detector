import { useState, useEffect } from "react";
import {
  Bell,
  LayoutGrid,
  Video,
  ImageIcon,
  AudioLines,
  Menu,
  X,
  // Play,
  // Pause,
  // SkipBack,
  // SkipForward,
} from "lucide-react";
import FourthImage from "../assets/images/fourthImage.png";
import { BackIcon } from "../assets/svg";
import { useNavigate, useParams, useLocation } from "react-router-dom";
import SafeguardMediaLogo from "../assets/images/SafeguardMedia8.svg";
import { CiSettings } from "react-icons/ci";
import { useSelector } from "react-redux";
import type { RootState } from "../store/store";

interface VideoAnalysisResult {
  success: boolean;
  message: string;
  thumbnailUrl: string;
  data: {
    analysis_type: string;
    overall_assessment: {
      confidence: number;
      fake_ratio: number;
      fake_segments: number;
      is_deepfake: boolean;
      predicted_class: string;
      real_ratio: number;
      real_segments: number;
      safeguard_analysis: {
        color_code: string;
        interpretation: string;
        recommended_action: string;
        risk_level: string;
      };
    };
    segment_analysis: Array<{
      duration: number;
      end_time: number;
      keyframe_info: {
        frame_number: number;
        timestamp: number;
        type: string;
      };
      prediction: {
        confidence: number;
        deepfake_probability: number;
        is_deepfake: boolean;
        predicted_class: string;
        real_probability: number;
      };
      safeguard_analysis: {
        color_code: string;
        interpretation: string;
        recommended_action: string;
        risk_level: string;
      };
      segment_id: number;
      start_time: number;
      time_range: string;
    }>;
    segment_summary: string[];
    technical_details: {
      frame_size: string;
      keyframe_extraction: string;
      model_type: string;
      processing_time_seconds: number;
      sequence_length: string;
      total_frames_analyzed: number;
    };
    video_filename: string;
    video_info: {
      duration: string;
      fps: number;
      segments_analyzed: number;
      total_frames: number;
    };
  };
}

const VideoScreen = () => {
  const navigate = useNavigate();
  const { token } = useParams();
  const location = useLocation();
  const [sidebarOpen, setSidebarOpen] = useState(false);
  // const [isPlaying, setIsPlaying] = useState(false);
  // const [currentTime, setCurrentTime] = useState(0);
  const [analysisResult, setAnalysisResult] =
    useState<VideoAnalysisResult | null>(null);
  const [fileName, setFileName] = useState("");
  const [fileSize, setFileSize] = useState("");
  // const [currentSegment, setCurrentSegment] = useState(0);
  const storedUser = useSelector((state: RootState) => state.user.user);

  useEffect(() => {
    // Get data from location state first, then fallback to localStorage
    const stateData = location.state;

    if (stateData?.analysisResult) {
      setAnalysisResult(stateData.analysisResult);
      setFileName(stateData.fileName || "Unknown File");
      setFileSize(stateData.fileSize || "Unknown Size");
    } else if (token) {
      // Fallback to localStorage
      const storedData = localStorage.getItem(`analysis_${token}`);
      if (storedData) {
        const parsedData = JSON.parse(storedData);
        setAnalysisResult(parsedData);
        setFileName(parsedData.data?.video_filename || "Unknown File");
        setFileSize("Unknown Size");
      }
    }
  }, [token, location.state]);

  // Calculate video duration from analysis data
  // const videoDuration = analysisResult?.data.video_info.duration
  //   ? parseFloat(analysisResult.data.video_info.duration.replace("s", ""))
  //   : 0;

  // Update current segment based on current time
  // useEffect(() => {
  //   if (analysisResult?.data.segment_analysis && videoDuration > 0) {
  //     const segments = analysisResult.data.segment_analysis;
  //     const currentSegmentIndex = segments.findIndex(
  //       (segment) =>
  //         currentTime >= segment.start_time && currentTime <= segment.end_time
  //     );
  //     if (currentSegmentIndex !== -1) {
  //       setCurrentSegment(currentSegmentIndex);
  //     }
  //   }
  // }, [currentTime, analysisResult, videoDuration]);

  const handleBack = () => {
    navigate("/dashboard");
  };

  // const handlePlayPause = () => {
  //   setIsPlaying(!isPlaying);
  // };

  // const formatTime = (seconds: number) => {
  //   const mins = Math.floor(seconds / 60);
  //   const secs = Math.floor(seconds % 60);
  //   return `${mins.toString().padStart(2, "0")}:${secs
  //     .toString()
  //     .padStart(2, "0")}`;
  // };

  // const handleTimelineClick = (e: React.MouseEvent<HTMLDivElement>) => {
  //   const rect = e.currentTarget.getBoundingClientRect();
  //   const clickX = e.clientX - rect.left;
  //   const percentage = clickX / rect.width;
  //   const newTime = percentage * videoDuration;
  //   setCurrentTime(newTime);
  // };

  // const getResultBadgeClass = (predicted_class: string) => {
  //   switch (predicted_class.toLowerCase()) {
  //     case "real":
  //       return "bg-green-600 text-white";
  //     case "fake":
  //     case "deepfake":
  //       return "bg-red-600 text-white";
  //     default:
  //       return "bg-yellow-600 text-white";
  //   }
  // };

  // const getSegmentColor = (predicted_class: string) => {
  //   switch (predicted_class.toLowerCase()) {
  //     case "real":
  //       return "bg-green-500";
  //     case "fake":
  //     case "deepfake":
  //       return "bg-red-500";
  //     default:
  //       return "bg-yellow-500";
  //   }
  // };

  // New functions for the improved results section
  const getRiskAssessment = () => {
    if (!analysisResult)
      return {
        riskLevel: "Unknown",
        interpretation: "Analysis not available",
        action: "Please retry analysis",
        riskColor: "gray",
        gaugeColor: "#9CA3AF",
      };

    const realRatio = analysisResult.data.overall_assessment.real_ratio;
    const fakeRatio = analysisResult.data.overall_assessment.fake_ratio;

    if (realRatio >= 90 && fakeRatio <= 10) {
      return {
        riskLevel: "Low",
        interpretation: "Very Likely Real",
        action: "Accept as authentic. Manual review optional.",
        riskColor: "green",
        gaugeColor: "#10B981",
      };
    } else if (realRatio >= 70 && fakeRatio <= 29) {
      return {
        riskLevel: "Medium",
        interpretation: "Likely Real, Some Risk",
        action: "Review manually if content is sensitive or high-stakes.",
        riskColor: "yellow",
        gaugeColor: "#F59E0B",
      };
    } else if (realRatio >= 50 && fakeRatio <= 49) {
      return {
        riskLevel: "Medium-High",
        interpretation: "Ambiguous / Uncertain",
        action:
          "Manual verification strongly recommended. Consider secondary tools.",
        riskColor: "orange",
        gaugeColor: "#F97316",
      };
    } else if (realRatio >= 30 && fakeRatio <= 69) {
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

    if (analysisResult.data.overall_assessment.is_deepfake) {
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
    return Math.round(analysisResult.data.overall_assessment.confidence);
  };

  if (!analysisResult) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-blue-600 mx-auto"></div>
          <p className="mt-4 text-gray-600">Loading analysis results...</p>
        </div>
      </div>
    );
  }

  const { data } = analysisResult;
  // const confidence = Math.round(data.overall_assessment.confidence);

  return (
    <div className={`min-h-screen bg-gray-50`}>
      {/* Full Width Header */}
      <header className="bg-white border-b border-gray-200 px-4 sm:px-6 py-4 w-full">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            {/* Mobile menu button */}
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

            <div
              className="flex items-center space-x-2 cursor-pointer rounded-[30px]"
              onClick={() => navigate("/settings")}
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

      {/* Mobile Sidebar Overlay */}
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
              <div className="flex items-center space-x-3 text-gray-400 cursor-not-allowed">
                <AudioLines className="w-6 h-6" />
                <span className="text-sm">Audio</span>
              </div>
              <div className="flex items-center space-x-3 text-gray-400  cursor-not-allowed">
                <Video className="w-6 h-6" />
                <span className="text-sm">Video</span>
              </div>
              <div className="flex items-center space-x-3 text-gray-400  cursor-not-allowed">
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
          <div className="flex flex-col items-center space-y-2 text-gray-400  cursor-not-allowed">
            <AudioLines className="w-6 h-6" />
            <span className="text-xs">Audio</span>
          </div>
          <div className="flex flex-col items-center space-y-2 text-gray-400  cursor-not-allowed">
            <Video className="w-6 h-6" />
            <span className="text-xs">Video</span>
          </div>
          <div className="flex flex-col items-center space-y-2 text-gray-400  cursor-not-allowed">
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
                    <span>Duration: {data.video_info.duration}</span>
                    <span className="mx-2">•</span>
                    <span>FPS: {data.video_info.fps}</span>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Video Preview and DF Results Section - Side by Side */}
          <div className="flex flex-col lg:flex-row px-4 sm:px-6 gap-4 sm:gap-6">
            {/* Video Preview - Left Side */}
            <div className="w-full lg:w-2/3">
              <div className=" rounded-xl overflow-hidden">
                {analysisResult.thumbnailUrl ? (
                  <img
                    src={analysisResult.thumbnailUrl}
                    alt="Video thumbnail"
                    className="w-full h-auto"
                  />
                ) : (
                  <img
                    src={FourthImage || "/placeholder.svg"}
                    alt="Video preview showing analysis result"
                    className="w-full h-auto"
                  />
                )}
              </div>
            </div>

            {/* DF Results Card - Right Side - UPDATED SECTION */}
            <div className="w-full lg:w-1/3">
              <div className="bg-white rounded-xl border border-gray-200 overflow-hidden min-h-[50vh] flex flex-col">
                {/* Header with DF Results and status badge */}
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

                  {/* Analysis Details */}
                  <div className="border-t border-gray-200 p-4 sm:p-6">
                    <h4 className="text-sm font-semibold text-[#020717] mb-3">
                      Confidence Breakdown:
                    </h4>
                    <div className="space-y-2">
                      <div className="flex justify-between items-center text-xs">
                        <span className="text-gray-600">Real Ratio:</span>
                        <span className="font-medium text-green-600">
                          {data.overall_assessment.real_ratio.toFixed(1)}%
                        </span>
                      </div>
                      <div className="flex justify-between items-center text-xs">
                        <span className="text-gray-600">Fake Ratio:</span>
                        <span className="font-medium text-red-600">
                          {data.overall_assessment.fake_ratio.toFixed(1)}%
                        </span>
                      </div>
                    </div>

                    {/* Additional Video-Specific Stats */}
                    <div className="mt-4 space-y-2">
                      <div className="flex justify-between text-xs text-gray-600">
                        <span>Real Video Frames:</span>
                        <span>
                          {data.overall_assessment.real_segments}/
                          {data.video_info.segments_analyzed}
                        </span>
                      </div>
                      <div className="flex justify-between text-xs text-gray-600">
                        <span>Fake Video Frames:</span>
                        <span>
                          {data.overall_assessment.fake_segments}/
                          {data.video_info.segments_analyzed}
                        </span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Disclaimer Section */}
          <div className="px-4 sm:px-6 pb-4 sm:pb-6 mt-5">
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

export default VideoScreen;
