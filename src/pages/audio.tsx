import type React from "react";

import { useState } from "react";
import {
  Bell,
  LayoutGrid,
  Video,
  ImageIcon,
  AudioLines,
  Menu,
  X,
  Download,
  Trash2,
  Play,
  Pause,
  SkipBack,
  SkipForward,
  Volume2,
  VolumeX,
} from "lucide-react";
import { BackIcon } from "../assets/svg";
import { useNavigate } from "react-router-dom";
import { useGetUserQuery } from "../services/apiService";
import SafeguardMediaLogo from "../assets/images/SafeguardMedia8.svg";

const AudioScreen = () => {
  const navigate = useNavigate();
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration] = useState(531);
  const [volume, setVolume] = useState(0.8);
  const [isMuted, setIsMuted] = useState(false);
  const { data: userData } = useGetUserQuery();

  const handleBack = () => {
    // Handle back navigation
    console.log("Going back...");
  };

  const handleDownloadReport = () => {
    // Handle download report
    console.log("Downloading report...");
  };

  const handleDeleteReport = () => {
    // Handle delete report
    console.log("Deleting report...");
  };

  const handlePlayPause = () => {
    setIsPlaying(!isPlaying);
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
    const rect = e.currentTarget.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const percentage = clickX / rect.width;
    const newTime = Math.floor(percentage * duration);
    setCurrentTime(newTime);
  };

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
              <span className="text-xl font-bold text-gray-900">
                Safeguardmedia
              </span>
            </div>
          </div>
          <div className="flex items-center space-x-2 sm:space-x-4">
            <div
              className="hidden sm:flex  bg-[#FBFBEF] gap-2 justify-between items-center"
              onClick={() => {
                navigate("/plans");
              }}
            >
              <button className="bg-[#0F2FA3] hover:bg-blue-700 text-white px-4 py-2 rounded-[30px] text-sm font-medium">
                Upgrade
              </button>
            </div>

            {/* Mobile upgrade button */}
            <button
              className="sm:hidden bg-[#0F2FA3] hover:bg-blue-700 text-white px-3 py-1.5 rounded-[20px] text-xs font-medium"
              onClick={() => {
                navigate("/plans");
              }}
            >
              Upgrade
            </button>

            <button
              className="p-2 text-gray-400 hover:text-gray-600 bg-[#F6F7FE] rounded-[30px] border-[0.88px] border-[#8C8C8C]"
              onClick={() => {
                navigate("/notifications");
              }}
            >
              <Bell className="w-4 h-4 sm:w-5 sm:h-5" />
            </button>

            <div
              className="flex items-center space-x-2 cursor-pointer rounded-[30px]"
              onClick={() => {
                navigate("/settings");
              }}
            >
              <div className="w-6 h-6 sm:w-8 sm:h-8 bg-gray-300 rounded-full flex items-center justify-center">
                <span className="text-xs sm:text-sm font-medium text-gray-600">
                  {userData?.data?.user?.firstName
                    ? userData.data.user.firstName.charAt(0).toUpperCase()
                    : "U"}
                </span>
              </div>
              <span className="hidden sm:inline text-sm text-gray-700">
                {userData?.data?.user?.firstName || "Username"}
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
              <div className="flex items-center space-x-3 text-gray-400 cursor-pointer">
                <AudioLines className="w-6 h-6" />
                <span className="text-sm">Audio</span>
              </div>
              <div className="flex items-center space-x-3 text-gray-400  cursor-pointer">
                <Video className="w-6 h-6" />
                <span className="text-sm">Video</span>
              </div>
              <div className="flex items-center space-x-3 text-gray-400  cursor-pointer">
                <ImageIcon className="w-6 h-6" />
                <span className="text-sm">Image</span>
              </div>
              <div className="flex flex-col items-center space-y-2 text-gray-400  cursor-pointer">
                <ImageIcon className="w-6 h-6" />
                <span className="text-xs">Settings</span>
              </div>
              <div className="flex flex-col items-center space-y-2 text-gray-400  cursor-pointer">
                <ImageIcon className="w-6 h-6" />
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
          <div className="flex flex-col items-center space-y-2 text-gray-400  cursor-pointer">
            <AudioLines className="w-6 h-6" />
            <span className="text-xs">Audio</span>
          </div>
          <div className="flex flex-col items-center space-y-2 text-gray-400  cursor-pointer">
            <Video className="w-6 h-6" />
            <span className="text-xs">Video</span>
          </div>
          <div className="flex flex-col items-center space-y-2 text-gray-400  cursor-pointer">
            <ImageIcon className="w-6 h-6" />
            <span className="text-xs">Image</span>
          </div>
          <div className="flex flex-col items-center space-y-2 text-gray-400  cursor-pointer">
            <ImageIcon className="w-6 h-6" />
            <span className="text-xs">Settings</span>
          </div>
          <div className="flex flex-col items-center space-y-2 text-gray-400  cursor-pointer">
            <ImageIcon className="w-6 h-6" />
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
                      Audio_Recording_01.mp3
                    </h2>
                  </div>
                  {/* File details */}
                  <div className="text-sm text-gray-600">
                    <span>File size: 12.34 MB</span>
                    <span className="mx-2">•</span>
                    <span>Date: 9th May, 2025, 10:34 am</span>
                  </div>
                </div>
                {/* Right side - Action buttons */}
                <div className="flex items-center space-x-2 sm:space-x-3">
                  <button
                    onClick={handleDownloadReport}
                    className="flex items-center space-x-2 px-3 sm:px-4 py-2 border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors"
                  >
                    <Download className="w-4 h-4 text-gray-600" />
                    <span className="text-sm font-medium text-gray-700">
                      Download Report
                    </span>
                  </button>
                  <button
                    onClick={handleDeleteReport}
                    className="flex items-center space-x-2 px-3 sm:px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors"
                  >
                    <Trash2 className="w-4 h-4" />
                    <span className="text-sm font-medium">Delete Report</span>
                  </button>
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
                      Waveform will be displayed here
                    </p>
                  </div>
                </div>
              </div>
            </div>

            {/* DF Results Card - Right Side */}
            <div className="w-full lg:w-1/3">
              <div className="bg-white rounded-xl border border-gray-200 overflow-hidden h-full flex flex-col">
                {/* Header with DF Results and Deepfake badge */}
                <div className="bg-[#0F2FA3] text-white px-4 sm:px-6 py-3 sm:py-4 flex items-center justify-between">
                  <span className="text-sm sm:text-base font-medium">
                    Safeguard Media Results
                  </span>
                  <span className="bg-white text-red-600 px-3 py-1 rounded-full text-xs sm:text-sm font-medium">
                    Deepfake
                  </span>
                </div>

                {/* Results Content */}
                <div className="flex-1 flex flex-col">
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
                  <div className="flex-1 p-4 sm:p-6">
                    <h4 className="text-sm sm:text-base font-semibold text-[#020717] mb-3">
                      Result Summary:
                    </h4>
                    <p className="text-xs sm:text-sm text-[#020717] font-[300] leading-relaxed">
                      Our model analysis found significant indicators in this
                      audio file strongly suggesting this media has been
                      manipulated using deepfake techniques.
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Audio Analysis Interface - New Section */}
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
                          onClick={() =>
                            setCurrentTime(Math.max(0, currentTime - 10))
                          }
                          className="p-2 hover:bg-gray-100 rounded-full transition-colors"
                        >
                          <SkipBack className="w-4 h-4 text-gray-600" />
                        </button>
                        <button
                          onClick={handlePlayPause}
                          className="p-3 bg-[#0F2FA3] hover:bg-blue-700 rounded-full transition-colors"
                        >
                          {isPlaying ? (
                            <Pause className="w-5 h-5 text-white" />
                          ) : (
                            <Play className="w-5 h-5 text-white ml-0.5" />
                          )}
                        </button>
                        <button
                          onClick={() =>
                            setCurrentTime(Math.min(duration, currentTime + 10))
                          }
                          className="p-2 hover:bg-gray-100 rounded-full transition-colors"
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
                            width: `${(currentTime / duration) * 100}%`,
                          }}
                        />
                        <div
                          className="absolute top-1/2 transform -translate-y-1/2 w-4 h-4 bg-[#0F2FA3] rounded-full border-2 border-white shadow-md transition-all duration-150"
                          style={{
                            left: `${(currentTime / duration) * 100}%`,
                            marginLeft: "-8px",
                          }}
                        />
                      </div>
                    </div>

                    {/* Timeline Markers */}
                    <div className="flex justify-between text-xs text-gray-500 mb-4 px-1">
                      <span>0s</span>
                      <span className="hidden xs:inline">1m</span>
                      <span>2m</span>
                      <span className="hidden xs:inline">4m</span>
                      <span>6m</span>
                      <span>8m</span>
                    </div>

                    {/* Audio Analysis Segments */}
                    <div className="mb-4">
                      <h4 className="text-sm font-medium text-gray-700 mb-2">
                        Analysis Segments
                      </h4>
                      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
                        {Array.from({ length: 8 }, (_, i) => (
                          <div
                            key={i}
                            className="p-2 bg-gray-50 rounded border cursor-pointer hover:bg-gray-100 transition-colors text-center"
                            onClick={() =>
                              setCurrentTime(Math.floor((i / 8) * duration))
                            }
                          >
                            <div className="text-xs text-gray-600">
                              Segment {i + 1}
                            </div>
                            <div className="text-xs text-gray-400">
                              {formatTime(Math.floor((i / 8) * duration))}
                            </div>
                          </div>
                        ))}
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
                    <h3 className="text-sm sm:text-base font-medium text-[#020717]">
                      What Do My Results Mean?
                    </h3>
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
