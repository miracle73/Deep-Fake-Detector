import type React from "react";
import { useState } from "react";
import {
  Bell,
  ChevronDown,
  LayoutGrid,
  Video,
  ImageIcon,
  // Clock,
  // FileText,
  // HelpCircle,
  AudioLines,
  MoreHorizontal,
  ChevronLeft,
  ChevronRight,
  Menu,
  X,
} from "lucide-react";
import { NoAnalysisYet, UploadIcon } from "../assets/svg";
import FirstImage from "../assets/images/firstImage.png";
import SecondImage from "../assets/images/secondImage.png";
import ThirdImage from "../assets/images/thirdImage.png";
import { useNavigate } from "react-router-dom";
// import { useGetUserQuery } from "../services/apiService";
const mockAnalyses = [
  {
    id: 1,
    fileName: "Video_Clip_01.mp4",
    thumbnail: "/placeholder.svg?height=40&width=40",
    uploadDate: "May 10, 2025, 08:15 AM",
    status: "Authentic",
    confidence: 88,
    type: "video",
    image: FirstImage,
  },
  {
    id: 2,
    fileName: "Audio_Clip_02.mp4",
    thumbnail: "/placeholder.svg?height=40&width=40",
    uploadDate: "May 10, 2025, 08:15 AM",
    status: "Uncertain",
    confidence: 20,
    type: "audio",
    image: SecondImage,
  },
  {
    id: 3,
    fileName: "Image_Clip_03.mp4",
    thumbnail: "/placeholder.svg?height=40&width=40",
    uploadDate: "May 10, 2025, 08:15 AM",
    status: "Deepfake",
    confidence: 97,
    type: "image",
    image: FirstImage,
  },
  {
    id: 4,
    fileName: "Video_Clip_04.mp4",
    thumbnail: "/placeholder.svg?height=40&width=40",
    uploadDate: "May 10, 2025, 08:15 AM",
    status: "Deepfake",
    confidence: 98,
    type: "video",
    image: SecondImage,
  },
];

const Dashboard = () => {
  const navigate = useNavigate();
  const [dragActive, setDragActive] = useState(false);
  const [hasAnalyses, setHasAnalyses] = useState(true);
  const [currentPage, setCurrentPage] = useState(1);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [uploadedFile, setUploadedFile] = useState<{
    name: string;
    size: string;
    thumbnail: string;
  } | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  // const { data: userData, isLoading: userLoading, error: userError } = useGetUserQuery();
  // Modify the handleUploadMedia function
  const handleUploadMedia = () => {
    setIsUploading(true);

    // Simulate file upload with delay
    setTimeout(() => {
      setUploadedFile({
        name: "Video_Clip_01.mp4",
        size: "17.53 MB",
        thumbnail: ThirdImage,
      });
      setIsUploading(false);
    }, 3000);
  };

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    // Handle file drop logic here
  };

  // const handleUploadMedia = () => {
  //   // Simulate file upload
  //   setUploadedFile({
  //     name: "Video_Clip_01.mp4",
  //     size: "17.53 MB",
  //     thumbnail:
  //       "https://hebbkx1anhila5yf.public.blob.vercel-storage.com/image-PfGOczEHEKWpuIxew8P36mT0KzEkji.png",
  //   });
  // };

  const handleRemoveFile = () => {
    setUploadedFile(null);
  };

  const handleAnalyseMedia = () => {
    // Handle analysis logic here
    console.log("Analysing media...");
    setUploadedFile(null);
  };
  const getStatusBadge = (status: string) => {
    const baseClasses = "px-3 py-1 rounded-full text-xs font-medium";
    switch (status) {
      case "Authentic":
        return `${baseClasses} bg-green-100 text-green-800`;
      case "Uncertain":
        return `${baseClasses} bg-yellow-100 text-yellow-800`;
      case "Deepfake":
        return `${baseClasses} bg-red-100 text-red-800`;
      default:
        return `${baseClasses} bg-gray-100 text-gray-800`;
    }
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
            <h1 className="text-lg sm:text-xl font-bold text-gray-900">
              <span className="font-bold">Safeguard</span>{" "}
              <span className="font-normal">Media</span>
            </h1>
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

            <div className="flex items-center space-x-2 cursor-pointer rounded-[30px]">
              <div className="w-6 h-6 sm:w-8 sm:h-8 bg-gray-300 rounded-full flex items-center justify-center">
                <span className="text-xs sm:text-sm font-medium text-gray-600">
                  U
                </span>
              </div>
              <span className="hidden sm:inline text-sm text-gray-700">
                Username
              </span>
              <ChevronDown className="w-3 h-3 sm:w-4 sm:h-4 text-gray-400" />
            </div>
            {/* <div className="flex items-center space-x-2 cursor-pointer rounded-[30px]">
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
              <ChevronDown className="w-3 h-3 sm:w-4 sm:h-4 text-gray-400" />
            </div> */}
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
        </div>

        {/* Main Content Container */}
        <div className="flex-1 flex flex-col">
          {/* Upper Section: Upload Area + Right Sidebar */}
          <div className="flex flex-col lg:flex-row">
            {/* Main Content Area */}
            <div className="w-full lg:w-2/3 p-4 sm:p-6">
              {/* Getting Started Section */}
              <div>
                <h2 className="text-xl sm:text-2xl font-semibold text-gray-900 mb-2">
                  Let's get started!
                </h2>
                <p className="text-sm sm:text-base text-gray-600 mb-6">
                  Upload your file for analysis, supports video, audio, and
                  image formats.
                </p>

                {/* Upload Area */}
                <div
                  className={`border-2 border-dashed rounded-xl p-6 sm:p-12 text-center transition-colors h-full ${
                    dragActive
                      ? "border-blue-400 bg-blue-50"
                      : "border-gray-300 bg-white"
                  }`}
                  onDragEnter={handleDrag}
                  onDragLeave={handleDrag}
                  onDragOver={handleDrag}
                  onDrop={handleDrop}
                >
                  {!uploadedFile ? (
                    <>
                      <div className="flex justify-center items-center mb-4">
                        <UploadIcon />
                      </div>

                      <h3 className="text-base sm:text-lg font-medium text-gray-900 mb-2">
                        Drag and drop to upload or browse files
                      </h3>
                      <p className="text-xs sm:text-sm text-red-500 mb-6">
                        Supports audio, video and image format. Max file size
                        1GB
                      </p>
                      <button
                        className="bg-[#FBFBEF] border border-[#8C8C8C] rounded-[30px] hover:bg-gray-200 text-gray-700 px-4 sm:px-6 py-2 sm:py-3 text-sm sm:text-base font-medium"
                        onClick={() => {
                          setHasAnalyses(!hasAnalyses);
                          handleUploadMedia();
                        }}
                        disabled={isUploading}
                      >
                        Upload Media
                      </button>
                    </>
                  ) : (
                    <div className="flex flex-col items-center space-y-4">
                      {/* Video Thumbnail */}
                      <div className="w-32 h-20 sm:w-40 sm:h-24 rounded-lg overflow-hidden bg-gray-200">
                        <img
                          src={ThirdImage}
                          alt="Video thumbnail"
                          className="w-full h-full object-cover"
                        />
                      </div>

                      {/* File Info */}
                      <div className="flex items-center space-x-2">
                        <div className=" flex items-center gap-4">
                          <p className="text-sm sm:text-base font-medium text-gray-900">
                            {uploadedFile.name}
                          </p>
                          <p className="text-xs sm:text-sm text-gray-500">
                            ({uploadedFile.size})
                          </p>
                        </div>
                        <button
                          onClick={handleRemoveFile}
                          className="p-1 text-red-500 hover:text-red-700 hover:bg-red-50 rounded-full"
                        >
                          <X className="w-4 h-4 sm:w-5 sm:h-5" />
                        </button>
                      </div>

                      {/* Analyse Button */}
                      <button
                        onClick={handleAnalyseMedia}
                        className="bg-gray-900 hover:bg-gray-800 text-white px-6 sm:px-8 py-2 sm:py-3 rounded-full text-sm sm:text-base font-medium transition-colors"
                      >
                        Analyse Media
                      </button>
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* Right Sidebar */}
            <div className="w-full lg:w-1/3 p-4 sm:p-6 lg:mt-22">
              {/* Combined Subscription and How it Works Card */}
              <div className="bg-white rounded-xl border border-gray-200 overflow-hidden h-full flex flex-col">
                {/* Subscription Header */}
                <div className="bg-[#0F2FA3] text-white px-4 sm:px-6 py-3 sm:py-4">
                  <span className="text-xs sm:text-sm font-medium">
                    Subscribe to Max plan and get 30% off
                  </span>
                </div>

                {/* How it Works Content */}
                <div className="p-4 sm:p-6 flex-1 flex flex-col justify-center">
                  <h3 className="text-base sm:text-lg font-semibold text-gray-900 mb-4">
                    How it Works
                  </h3>
                  <div className="space-y-3 sm:space-y-4">
                    <div className="flex items-center justify-start gap-4">
                      <div>→</div>
                      <p className="text-xs sm:text-sm text-gray-600">
                        Upload your media
                      </p>
                    </div>
                    <div className="flex items-center justify-start gap-4">
                      <div>→</div>
                      <p className="text-xs sm:text-sm text-gray-600">
                        Our model automatically processes the media to determine
                        whether it is AI generated or not.
                      </p>
                    </div>
                    <div className="flex items-center justify-start gap-4">
                      <div>→</div>
                      <p className="text-xs sm:text-sm text-gray-600">
                        Get instant and clear results with confidence.
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Recent Analyses Section - Full Width */}
          <div className="px-4 sm:px-6 pb-4 sm:pb-6">
            <div className="bg-white rounded-xl border border-gray-200 p-4 sm:p-6">
              <div className="mb-6">
                <h3 className="text-base sm:text-lg font-semibold text-gray-900 mb-2">
                  Your Recent Analyses
                </h3>
                <p className="text-xs sm:text-sm text-gray-600">
                  Review the results of your recently uploaded media files.
                  Results are stored for 30 days.
                </p>
              </div>

              {hasAnalyses ? (
                /* Analyses Table View */
                <div>
                  {/* Desktop Table Header */}
                  <div className="hidden md:grid grid-cols-12 gap-4 pb-3 border-b border-gray-200 text-sm font-medium text-gray-500">
                    <div className="col-span-4">File name/thumbnail</div>
                    <div className="col-span-3">Upload date/time</div>
                    <div className="col-span-2">Status</div>
                    <div className="col-span-2">Confidence score</div>
                    <div className="col-span-1"></div>
                  </div>

                  {/* Table Rows */}
                  <div className="space-y-3 mt-4">
                    {mockAnalyses.map((analysis) => (
                      <div key={analysis.id}>
                        {/* Desktop Row */}
                        <div className="hidden md:grid grid-cols-12 gap-4 items-center py-3 hover:bg-gray-50 rounded-lg">
                          <div className="col-span-4 flex items-center space-x-3">
                            <div className="w-10 h-10 rounded-lg flex items-center justify-center">
                              <img
                                src={analysis.image || "/placeholder.svg"}
                                alt={analysis.fileName}
                                className="w-10 h-10 rounded-lg object-cover"
                              />
                            </div>
                            <span className="text-sm font-medium text-gray-900 truncate">
                              {analysis.fileName}
                            </span>
                          </div>
                          <div className="col-span-3">
                            <span className="text-sm text-gray-600">
                              {analysis.uploadDate}
                            </span>
                          </div>
                          <div className="col-span-2">
                            <span className={getStatusBadge(analysis.status)}>
                              {analysis.status}
                            </span>
                          </div>
                          <div className="col-span-2">
                            <span className="text-sm font-medium text-gray-900">
                              {analysis.confidence}%
                            </span>
                          </div>
                          <div className="col-span-1 flex justify-end">
                            <button className="p-1 hover:bg-gray-100 rounded">
                              <MoreHorizontal className="w-4 h-4 text-gray-400" />
                            </button>
                          </div>
                        </div>

                        {/* Mobile Card */}
                        <div className="md:hidden bg-gray-50 rounded-lg p-4 hover:bg-gray-100">
                          <div className="flex items-start space-x-3">
                            <div className="w-12 h-12 rounded-lg flex items-center justify-center flex-shrink-0">
                              <img
                                src={analysis.image || "/placeholder.svg"}
                                alt={analysis.fileName}
                                className="w-12 h-12 rounded-lg object-cover"
                              />
                            </div>
                            <div className="flex-1 min-w-0">
                              <div className="flex items-start justify-between">
                                <div className="flex-1 min-w-0">
                                  <p className="text-sm font-medium text-gray-900 truncate">
                                    {analysis.fileName}
                                  </p>
                                  <p className="text-xs text-gray-600 mt-1">
                                    {analysis.uploadDate}
                                  </p>
                                </div>
                                <button className="p-1 hover:bg-gray-200 rounded ml-2">
                                  <MoreHorizontal className="w-4 h-4 text-gray-400" />
                                </button>
                              </div>
                              <div className="flex items-center justify-between mt-3">
                                <span
                                  className={getStatusBadge(analysis.status)}
                                >
                                  {analysis.status}
                                </span>
                                <span className="text-sm font-medium text-gray-900">
                                  {analysis.confidence}%
                                </span>
                              </div>
                            </div>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>

                  {/* Pagination */}
                  <div className="flex items-center justify-center space-x-1 sm:space-x-2 mt-6 pt-4 border-t border-gray-200">
                    <button className="p-1.5 sm:p-2 hover:bg-gray-100 rounded">
                      <ChevronLeft className="w-3 h-3 sm:w-4 sm:h-4 text-gray-400" />
                    </button>
                    {[1, 2, 3, 4, 5].map((page) => (
                      <button
                        key={page}
                        className={`w-6 h-6 sm:w-8 sm:h-8 rounded text-xs sm:text-sm font-medium ${
                          currentPage === page
                            ? "bg-[#0F2FA3] text-white"
                            : "text-gray-600 hover:bg-gray-100"
                        }`}
                        onClick={() => setCurrentPage(page)}
                      >
                        {page}
                      </button>
                    ))}
                    <button className="p-1.5 sm:p-2 hover:bg-gray-100 rounded">
                      <ChevronRight className="w-3 h-3 sm:w-4 sm:h-4 text-gray-400" />
                    </button>
                  </div>
                </div>
              ) : (
                <>
                  {/* Empty State */}
                  <div className="text-center py-8 sm:py-12">
                    <div className="flex justify-center items-center mb-4">
                      <NoAnalysisYet />
                    </div>
                    <h4 className="text-base sm:text-lg font-medium text-gray-900 mb-2">
                      No Analyses Yet!
                    </h4>
                    <p className="text-xs sm:text-sm text-gray-600 mb-1">
                      Upload your first video, audio, or image file to check for
                      manipulation.
                    </p>
                    <p className="text-xs sm:text-sm text-gray-600">
                      Our model will provide a quick and clear assessment.
                    </p>
                  </div>
                </>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
