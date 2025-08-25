import type React from "react";
import { useState, useEffect, useRef } from "react";
import {
  Bell,
  LayoutGrid,
  Video,
  ImageIcon,
  AudioLines,
  Menu,
  X,
  Play,
  Pause,
  SkipBack,
  SkipForward,
  Volume2,
  VolumeX,
  Shield,
} from "lucide-react";
import { BackIcon } from "../assets/svg";
import { useNavigate, useParams, useLocation } from "react-router-dom";
import { useSelector } from "react-redux";
import type { RootState } from "../store/store";
import SafeguardMediaLogo from "../assets/images/SafeguardMedia8.svg";
import { CiSettings } from "react-icons/ci";

interface AnalysisResult {
  status: string;
  confidenceScore: number;
  fileName?: string;
  fileSize?: string;
  fileUrl?: string;
  realRatio?: number;
  fakeRatio?: number;
}

const AudioScreen = () => {
  const navigate = useNavigate();
  const { token } = useParams();
  const location = useLocation();
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [volume, setVolume] = useState(0.8);
  const [isMuted, setIsMuted] = useState(false);
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
        setAnalysisResult(parsedData);
        setFileName(parsedData.fileName || "Unknown Audio File");
        setFileSize(parsedData.fileSize || "Unknown size");
        setFileUrl(parsedData.fileUrl || "");
        setUploadDate(new Date().toLocaleDateString());
      }
      // If no stored data, try to get from location state
      else if (location.state) {
        const {
          analysisResult: result,
          fileName: name,
          fileSize: size,
          fileUrl: url,
        } = location.state;
        setAnalysisResult(result);
        setFileName(name || "Unknown Audio File");
        setFileSize(size || "Unknown size");
        setFileUrl(url || "");
        setUploadDate(new Date().toLocaleDateString());
      }
    }
  }, [token, location.state]);

  // Audio event handlers
  useEffect(() => {
    const audio = audioRef.current;
    if (audio) {
      const handleLoadedMetadata = () => {
        setDuration(Math.floor(audio.duration));
      };

      const handleTimeUpdate = () => {
        setCurrentTime(Math.floor(audio.currentTime));
      };

      const handleEnded = () => {
        setIsPlaying(false);
        setCurrentTime(0);
      };

      audio.addEventListener("loadedmetadata", handleLoadedMetadata);
      audio.addEventListener("timeupdate", handleTimeUpdate);
      audio.addEventListener("ended", handleEnded);

      return () => {
        audio.removeEventListener("loadedmetadata", handleLoadedMetadata);
        audio.removeEventListener("timeupdate", handleTimeUpdate);
        audio.removeEventListener("ended", handleEnded);
      };
    }
  }, [fileUrl]);

  // Update audio volume
  useEffect(() => {
    if (audioRef.current) {
      audioRef.current.volume = isMuted ? 0 : volume;
    }
  }, [volume, isMuted]);

  const handleBack = () => {
    navigate("/dashboard");
  };

  const handlePlayPause = () => {
    if (audioRef.current) {
      if (isPlaying) {
        audioRef.current.pause();
      } else {
        audioRef.current.play();
      }
      setIsPlaying(!isPlaying);
    }
  };

  const handleVolumeToggle = () => {
    setIsMuted(!isMuted);
  };

  const handleVolumeChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newVolume = parseFloat(e.target.value);
    setVolume(newVolume);
    setIsMuted(newVolume === 0);
  };

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, "0")}:${secs
      .toString()
      .padStart(2, "0")}`;
  };

  const handleTimelineClick = (e: React.MouseEvent<HTMLDivElement>) => {
    if (audioRef.current && duration > 0) {
      const rect = e.currentTarget.getBoundingClientRect();
      const clickX = e.clientX - rect.left;
      const percentage = clickX / rect.width;
      const newTime = Math.floor(percentage * duration);
      audioRef.current.currentTime = newTime;
      setCurrentTime(newTime);
    }
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

    const confidence = analysisResult.confidenceScore;

    if (
      analysisResult.status?.toLowerCase() === "authentic" ||
      analysisResult.status?.toLowerCase() === "real"
    ) {
      if (confidence >= 90) {
        return {
          riskLevel: "Low",
          interpretation: "Very Likely Authentic",
          action: "Accept as authentic. Manual review optional.",
          riskColor: "green",
          gaugeColor: "#10B981",
        };
      } else if (confidence >= 70) {
        return {
          riskLevel: "Medium",
          interpretation: "Likely Authentic, Some Risk",
          action: "Review manually if content is sensitive or high-stakes.",
          riskColor: "yellow",
          gaugeColor: "#F59E0B",
        };
      }
    } else if (
      analysisResult.status?.toLowerCase() === "uncertain" ||
      analysisResult.status?.toLowerCase() === "inconclusive"
    ) {
      return {
        riskLevel: "Medium-High",
        interpretation: "Ambiguous / Uncertain",
        action:
          "Manual verification strongly recommended. Consider secondary tools.",
        riskColor: "orange",
        gaugeColor: "#F97316",
      };
    } else if (
      analysisResult.status?.toLowerCase() === "deepfake" ||
      analysisResult.status?.toLowerCase() === "fake"
    ) {
      if (confidence >= 70) {
        return {
          riskLevel: "Very High",
          interpretation: "Very Likely Deepfake",
          action: "Reject or flag. Notify relevant stakeholders.",
          riskColor: "red",
          gaugeColor: "#DC2626",
        };
      } else {
        return {
          riskLevel: "High",
          interpretation: "Likely Deepfake, But Not Conclusive",
          action: "Treat cautiously. Manual review required; possibly reject.",
          riskColor: "red",
          gaugeColor: "#EF4444",
        };
      }
    }

    return {
      riskLevel: "Medium",
      interpretation: "Analysis Uncertain",
      action: "Manual review recommended.",
      riskColor: "yellow",
      gaugeColor: "#F59E0B",
    };
  };

  const getResultStatus = () => {
    if (!analysisResult)
      return { text: "Unknown", color: "gray", bgColor: "bg-gray-100" };

    if (
      analysisResult.status?.toLowerCase() === "deepfake" ||
      analysisResult.status?.toLowerCase() === "fake"
    ) {
      return {
        text: "Deepfake",
        color: "red",
        bgColor: "bg-red-600",
        textColor: "text-red-600",
      };
    } else if (
      analysisResult.status?.toLowerCase() === "authentic" ||
      analysisResult.status?.toLowerCase() === "real"
    ) {
      return {
        text: "Authentic",
        color: "green",
        bgColor: "bg-green-600",
        textColor: "text-green-600",
      };
    } else {
      return {
        text: "Uncertain",
        color: "yellow",
        bgColor: "bg-yellow-600",
        textColor: "text-yellow-600",
      };
    }
  };

  const getConfidenceScore = () => {
    if (!analysisResult) return 0;
    return Math.round(analysisResult.confidenceScore);
  };

  const getResultSummary = (status: string) => {
    switch (status?.toLowerCase()) {
      case "authentic":
      case "real":
        return "Our model analysis found little to no evidence of manipulation in this audio file. The audio appears to be authentic.";
      case "uncertain":
      case "inconclusive":
        return "Our model detected some indicators of manipulation, but the evidence isn't conclusive, or the audio quality impacts certainty.";
      case "deepfake":
      case "fake":
      case "synthetic":
        return "Our model analysis found significant indicators in this audio file strongly suggesting this media has been manipulated using deepfake techniques.";
      default:
        return "Analysis completed. Please review the confidence score and other indicators for more details.";
    }
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
            <div className="w-full lg:w-2/3">
              <div className="bg-white rounded-xl border border-gray-200 p-6">
                {/* Audio waveform visualization placeholder */}
                <div className="h-48 sm:h-64 bg-gray-50 rounded-lg flex items-center justify-center border-2 border-dashed border-gray-300">
                  <div className="text-center">
                    <AudioLines className="w-16 h-16 text-gray-400 mx-auto mb-4" />
                    <p className="text-gray-500 text-sm">
                      Audio Waveform Visualization
                    </p>
                    <p className="text-gray-400 text-xs mt-1">
                      {fileName
                        ? `Playing: ${fileName}`
                        : "Waveform will be displayed here"}
                    </p>
                  </div>
                </div>
              </div>
            </div>

            {/* Enhanced Results Card - Right Side (Similar to VideoScreen) */}
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
                            opacity: getConfidenceScore() === 0 ? 0 : 1,
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
                      Analysis Summary:
                    </h4>
                    <p className="text-xs sm:text-sm text-[#020717] font-[300] leading-relaxed">
                      {analysisResult
                        ? getResultSummary(analysisResult.status)
                        : "Loading analysis results..."}
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Audio Analysis Interface - Updated to use real audio */}
          <div className="px-2 sm:px-4 md:px-6 py-4 sm:py-6">
            <div className="flex flex-col lg:flex-row gap-4 sm:gap-6">
              {/* Audio Player Interface - Left Side */}
              <div className="w-full lg:w-2/3">
                <div className="bg-white rounded-xl border border-gray-200 overflow-hidden">
                  {/* Audio Player Controls */}
                  <div className="p-4 sm:p-6">
                    {/* Time Display and Controls */}
                    <div className="flex flex-row items-center justify-center space-x-4 gap-2 sm:gap-4 mb-4">
                      <div className="flex items-center justify-between sm:justify-start space-x-4 order-1 sm:order-1">
                        <span className="text-sm font-mono text-gray-600">
                          {formatTime(currentTime)}
                        </span>
                      </div>
                      <div className="flex items-center justify-center space-x-4 order-2 sm:order-2">
                        <button
                          onClick={() => {
                            const newTime = Math.max(0, currentTime - 10);
                            if (audioRef.current) {
                              audioRef.current.currentTime = newTime;
                            }
                            setCurrentTime(newTime);
                          }}
                          className="p-2 hover:bg-gray-100 rounded-full transition-colors"
                          disabled={!fileUrl}
                        >
                          <SkipBack className="w-4 h-4 text-gray-600" />
                        </button>
                        <button
                          onClick={handlePlayPause}
                          className="p-3 bg-[#0F2FA3] hover:bg-blue-700 rounded-full transition-colors disabled:bg-gray-400"
                          disabled={!fileUrl}
                        >
                          {isPlaying ? (
                            <Pause className="w-5 h-5 text-white" />
                          ) : (
                            <Play className="w-5 h-5 text-white ml-0.5" />
                          )}
                        </button>
                        <button
                          onClick={() => {
                            const newTime = Math.min(
                              duration,
                              currentTime + 10
                            );
                            if (audioRef.current) {
                              audioRef.current.currentTime = newTime;
                            }
                            setCurrentTime(newTime);
                          }}
                          className="p-2 hover:bg-gray-100 rounded-full transition-colors"
                          disabled={!fileUrl}
                        >
                          <SkipForward className="w-4 h-4 text-gray-600" />
                        </button>
                      </div>
                      <div className="flex items-center justify-between sm:justify-end space-x-4 order-3 sm:order-3">
                        <span className="text-sm font-mono text-gray-600">
                          {formatTime(duration)}
                        </span>
                      </div>
                    </div>

                    {/* Volume Control */}
                    <div className="flex items-center justify-center space-x-3 mb-4">
                      <button
                        onClick={handleVolumeToggle}
                        className="p-2 hover:bg-gray-100 rounded-full transition-colors"
                      >
                        {isMuted || volume === 0 ? (
                          <VolumeX className="w-4 h-4 text-gray-600" />
                        ) : (
                          <Volume2 className="w-4 h-4 text-gray-600" />
                        )}
                      </button>
                      <div className="flex-1 max-w-32">
                        <input
                          type="range"
                          min="0"
                          max="1"
                          step="0.1"
                          value={isMuted ? 0 : volume}
                          onChange={handleVolumeChange}
                          className="w-full h-1 bg-gray-200 rounded-lg appearance-none cursor-pointer slider"
                        />
                      </div>
                      <span className="text-xs text-gray-500 w-8">
                        {Math.round((isMuted ? 0 : volume) * 100)}%
                      </span>
                    </div>

                    {/* Timeline Scrubber */}
                    <div className="mb-4">
                      <div
                        className="relative h-1 bg-gray-200 rounded-full cursor-pointer"
                        onClick={handleTimelineClick}
                      >
                        <div
                          className="absolute top-0 left-0 h-full bg-[#0F2FA3] rounded-full transition-all duration-150"
                          style={{
                            width:
                              duration > 0
                                ? `${(currentTime / duration) * 100}%`
                                : "0%",
                          }}
                        />
                        <div
                          className="absolute top-1/2 transform -translate-y-1/2 w-4 h-4 bg-[#0F2FA3] rounded-full border-2 border-white shadow-md transition-all duration-150"
                          style={{
                            left:
                              duration > 0
                                ? `${(currentTime / duration) * 100}%`
                                : "0%",
                            marginLeft: "-8px",
                          }}
                        />
                      </div>
                    </div>

                    {/* Timeline Markers */}
                    <div className="flex justify-between text-xs text-gray-500 mb-4 px-1">
                      <span>0s</span>
                      {duration > 60 && (
                        <span className="hidden xs:inline">1m</span>
                      )}
                      {duration > 120 && <span>2m</span>}
                      {duration > 240 && (
                        <span className="hidden xs:inline">4m</span>
                      )}
                      {duration > 360 && <span>6m</span>}
                      {duration > 480 && <span>8m</span>}
                    </div>

                    {/* Audio Analysis Segments */}
                    <div className="mb-4">
                      <h4 className="text-sm font-medium text-gray-700 mb-2">
                        Analysis Segments
                      </h4>
                      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
                        {Array.from(
                          { length: Math.min(8, Math.ceil(duration / 60)) },
                          (_, i) => (
                            <div
                              key={i}
                              className="p-2 bg-gray-50 rounded border cursor-pointer hover:bg-gray-100 transition-colors text-center"
                              onClick={() => {
                                const newTime = Math.floor(
                                  (i / Math.min(8, Math.ceil(duration / 60))) *
                                    duration
                                );
                                if (audioRef.current) {
                                  audioRef.current.currentTime = newTime;
                                }
                                setCurrentTime(newTime);
                              }}
                            >
                              <div className="text-xs text-gray-600">
                                Segment {i + 1}
                              </div>
                              <div className="text-xs text-gray-400">
                                {formatTime(
                                  Math.floor(
                                    (i /
                                      Math.min(8, Math.ceil(duration / 60))) *
                                      duration
                                  )
                                )}
                              </div>
                            </div>
                          )
                        )}
                      </div>
                    </div>

                    {/* Analysis Note */}
                    <div className="mt-4 p-3 bg-yellow-50 border border-yellow-200 rounded-lg">
                      <p className="text-xs sm:text-sm text-gray-700">
                        <span className="font-medium">Note:</span> Highlighted
                        segments indicate areas where our model detected
                        anomalies most strongly associated with known deepfake
                        audio techniques.
                      </p>
                    </div>

                    {!fileUrl && (
                      <div className="mt-4 p-3 bg-gray-50 border border-gray-200 rounded-lg text-center">
                        <p className="text-sm text-gray-600">
                          Audio file not available for playback
                        </p>
                      </div>
                    )}
                  </div>
                </div>
              </div>

              {/* Results Explanation Panel - Right Side */}
              <div className="w-full lg:w-1/3">
                <div className="bg-white rounded-xl border border-gray-200 overflow-hidden h-full">
                  {/* Header */}
                  <div className="bg-[#0F2FA3] text-white px-4 sm:px-6 py-3 sm:py-4">
                    <h3 className="text-sm sm:text-base font-medium">
                      What Do My Results Mean?
                    </h3>
                  </div>

                  {/* Results Categories */}
                  <div className="p-3 sm:p-4 md:p-6 space-y-3 sm:space-y-4">
                    {/* Authentic */}
                    <div className="flex flex-col sm:flex-row sm:items-start space-y-2 sm:space-y-0 sm:space-x-3">
                      <div className="py-1 px-3 sm:py-2 sm:px-4 bg-[#E8F8EA] rounded-full flex-shrink-0 self-start">
                        <h4 className="text-xs sm:text-sm font-semibold text-[#257933]">
                          Authentic
                        </h4>
                      </div>
                      <div className="flex-1">
                        <p className="text-xs sm:text-sm text-gray-600 leading-relaxed">
                          Our model found little to no evidence of manipulation.
                        </p>
                      </div>
                    </div>

                    {/* Uncertain */}
                    <div className="flex flex-col sm:flex-row sm:items-start space-y-2 sm:space-y-0 sm:space-x-3">
                      <div className="py-1 px-3 sm:py-2 sm:px-4 bg-[#FFF8E5] rounded-full flex-shrink-0 self-start">
                        <h4 className="text-xs sm:text-sm font-semibold text-[#8F6D00]">
                          Uncertain
                        </h4>
                      </div>
                      <div className="flex-1">
                        <p className="text-xs sm:text-sm text-gray-600 leading-relaxed">
                          Our model detected some indicators of manipulation,
                          but the evidence isn't conclusive, or the audio
                          quality impacts certainty.
                        </p>
                      </div>
                    </div>

                    {/* Deepfake */}
                    <div className="flex flex-col sm:flex-row sm:items-start space-y-2 sm:space-y-0 sm:space-x-3">
                      <div className="py-1 px-3 sm:py-2 sm:px-4 bg-[#FDEDEE] rounded-full flex-shrink-0 self-start">
                        <h4 className="text-xs sm:text-sm font-semibold text-[#B5171F]">
                          Deepfake
                        </h4>
                      </div>
                      <div className="flex-1">
                        <p className="text-xs sm:text-sm text-gray-600 leading-relaxed">
                          Our model found significant evidence suggesting this
                          audio has been manipulated.
                        </p>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
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
