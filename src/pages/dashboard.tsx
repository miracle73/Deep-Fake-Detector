import type React from "react";
import { useState, useEffect } from "react";
import {
  Bell,
  LayoutGrid,
  Video,
  ImageIcon,
  // Clock,
  // FileText,
  // HelpCircle,
  AudioLines,
  // MoreHorizontal,
  ChevronLeft,
  ChevronRight,
  Menu,
  X,
  AlertCircle,
} from "lucide-react";
import { NoAnalysisYet, UploadIcon } from "../assets/svg";
// import FirstImage from "../assets/images/firstImage.png";
// import SecondImage from "../assets/images/secondImage.png";
import ThirdImage from "../assets/images/thirdImage.png";
import { useNavigate } from "react-router-dom";
import {
  // useGetUserQuery,
  useUpdateMediaConsentMutation,
  useDetectAnalyzeMutation,
  useGetAnalysisHistoryQuery,
} from "../services/apiService";
import SafeguardMediaLogo from "../assets/images/SafeguardMedia8.svg";
import { CiSettings } from "react-icons/ci";
import { useSelector } from "react-redux";
import type { RootState } from "../store/store";

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
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [filePreview, setFilePreview] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [hasConsented, setHasConsented] = useState(true);
  const [isFirstTimeUser, setIsFirstTimeUser] = useState(true);
  const [showConsentModal, setShowConsentModal] = useState(false);
  // const { data: userData } = useGetUserQuery();
  // const { data: historyData } = useGetAnalysisHistoryQuery();
  const storedUser = useSelector((state: RootState) => state.user.user);
  const [updateMediaConsent] = useUpdateMediaConsentMutation();
  const [detectAnalyze] = useDetectAnalyzeMutation();
  const [itemsPerPage] = useState(10);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisError, setAnalysisError] = useState<string | null>(null);

  const getPageNumbers = () => {
    if (!historyData?.pagination) return [];

    const totalPages = historyData.pagination.totalPages;
    const pages = [];

    if (totalPages <= 7) {
      for (let i = 1; i <= totalPages; i++) {
        pages.push(i);
      }
    } else {
      if (currentPage <= 4) {
        for (let i = 1; i <= 5; i++) {
          pages.push(i);
        }
        pages.push("ellipsis");
        pages.push(totalPages);
      } else if (currentPage >= totalPages - 3) {
        pages.push(1);
        pages.push("ellipsis");
        for (let i = totalPages - 4; i <= totalPages; i++) {
          pages.push(i);
        }
      } else {
        pages.push(1);
        pages.push("ellipsis");
        for (let i = currentPage - 1; i <= currentPage + 1; i++) {
          pages.push(i);
        }
        pages.push("ellipsis");
        pages.push(totalPages);
      }
    }

    return pages;
  };

  const {
    data: historyData,
    isLoading: isHistoryLoading,
    error: historyError,
  } = useGetAnalysisHistoryQuery();

  const getPaginatedData = () => {
    if (!historyData?.data) return [];

    const startIndex = (currentPage - 1) * itemsPerPage;
    const endIndex = startIndex + itemsPerPage;
    return historyData.data.slice(startIndex, endIndex);
  };

  // const getTotalPages = () => {
  //   if (!historyData?.data) return 0;
  //   return Math.ceil(historyData.data.length / itemsPerPage);
  // };

  // useEffect(() => {
  //   setCurrentPage(1);
  // }, [historyData]);
  useEffect(() => {
    if (historyData?.pagination?.currentPage !== currentPage) {
      setCurrentPage(historyData?.pagination?.currentPage || 1);
    }
  }, [historyData, currentPage]);

  const generateUniqueToken = (): string => {
    return Date.now().toString(36) + Math.random().toString(36).substr(2);
  };

  const handleUploadMedia = () => {
    if (isFirstTimeUser) {
      setShowConsentModal(true);
      return;
    }

    const fileInput = document.getElementById(
      "file-upload-input"
    ) as HTMLInputElement;
    fileInput?.click();
  };

  const handleConsentSubmit = async () => {
    // Store the consent choice regardless of what user selected
    localStorage.setItem("safeguardmedia_consent", hasConsented.toString());
    localStorage.setItem("safeguardmedia_uploaded", "true");
    setIsFirstTimeUser(false);
    setShowConsentModal(false);

    // Call API with inverted logic:
    // hasConsented = true means user does NOT consent = allowStorage: false
    // hasConsented = false means user consents = allowStorage: true
    try {
      await updateMediaConsent({
        allowStorage: !hasConsented,
      }).unwrap();
    } catch (error) {
      console.error("Failed to update media consent:", error);
    }

    // Always proceed with file upload regardless of consent choice
    const fileInput = document.getElementById(
      "file-upload-input"
    ) as HTMLInputElement;
    fileInput?.click();
  };

  const handleFileSelect = (file: File) => {
    setSelectedFile(file);
    setIsUploading(true);

    // Generate preview based on file type
    if (file.type.startsWith("image/")) {
      const reader = new FileReader();
      reader.onload = (e) => {
        setFilePreview(e.target?.result as string);
      };
      reader.readAsDataURL(file);
    } else if (file.type.startsWith("video/")) {
      // For videos, create a video element to extract thumbnail
      const video = document.createElement("video");
      const canvas = document.createElement("canvas");
      const ctx = canvas.getContext("2d");

      video.onloadedmetadata = () => {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        video.currentTime = 1; // Seek to 1 second
      };

      video.onseeked = () => {
        if (ctx) {
          ctx.drawImage(video, 0, 0);
          setFilePreview(canvas.toDataURL());
        }
      };

      video.src = URL.createObjectURL(file);
    } else {
      // For audio files, use a default audio icon or create a simple preview
      setFilePreview(null);
    }

    // Format file size
    const formatFileSize = (bytes: number) => {
      if (bytes === 0) return "0 Bytes";
      const k = 1024;
      const sizes = ["Bytes", "KB", "MB", "GB"];
      const i = Math.floor(Math.log(bytes) / Math.log(k));
      return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
    };

    // Simulate upload delay
    setTimeout(() => {
      setUploadedFile({
        name: file.name,
        size: formatFileSize(file.size),
        thumbnail: filePreview || ThirdImage,
      });
      setIsUploading(false);
    }, 1000);
  };

  // Add this function to handle file input change
  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      handleFileSelect(file);
    }
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

    const files = e.dataTransfer.files;
    if (files && files[0]) {
      const file = files[0];
      // Check if file type is supported
      const supportedTypes = [
        "video/mp4",
        "video/avi",
        "video/quicktime",
        "image/jpeg",
        "image/jpg",
        "image/png",
        "image/webp",
        "audio/mp3",
        "audio/mpeg",
        "audio/wav",
        "audio/aac",
      ];

      if (
        supportedTypes.includes(file.type) ||
        file.name.match(/\.(mp4|avi|mov|jpeg|jpg|png|webp|mp3|wav|aac)$/i)
      ) {
        handleFileSelect(file);
      } else {
        alert(
          "Unsupported file type. Please upload MP4, AVI, MOV, JPEG, PNG, WEBP, MP3, WAV, or AAC files."
        );
      }
    }
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

  const handleAnalyseMedia = async () => {
    if (!selectedFile) {
      setAnalysisError("No file selected for analysis");
      return;
    }

    setIsAnalyzing(true);
    setAnalysisError(null);

    try {
      const response = await detectAnalyze({ image: selectedFile }).unwrap();

      // Generate unique token
      const token = generateUniqueToken();

      // Store the response data temporarily (you can use localStorage or state management)
      localStorage.setItem(`analysis_${token}`, JSON.stringify(response));

      // Navigate to image-detection page with token and response
      navigate(`/image-detection/${token}`, {
        state: {
          analysisResult: response,
          fileName: uploadedFile?.name || selectedFile.name,
          fileSize:
            uploadedFile?.size ||
            `${(selectedFile.size / 1024 / 1024).toFixed(2)} MB`,
        },
      });
    } catch (error: unknown) {
      console.error("Analysis failed:", error);

      if (error && typeof error === "object" && "data" in error) {
        const apiError = error as {
          data?: { message?: string };
        };
        if (apiError.data?.message) {
          setAnalysisError(apiError.data.message);
        } else {
          setAnalysisError("Failed to analyze the image. Please try again.");
        }
      } else if (error && typeof error === "object" && "message" in error) {
        const messageError = error as { message: string };
        setAnalysisError(messageError.message);
      } else {
        setAnalysisError("Failed to analyze the image. Please try again.");
      }
    } finally {
      setIsAnalyzing(false);
    }
  };
  // const getStatusBadge = (status: string) => {
  //   const baseClasses = "px-3 py-1 rounded-full text-xs font-medium";
  //   switch (status) {
  //     case "Authentic":
  //       return `${baseClasses} bg-green-100 text-green-800`;
  //     case "Uncertain":
  //       return `${baseClasses} bg-yellow-100 text-yellow-800`;
  //     case "Deepfake":
  //       return `${baseClasses} bg-red-100 text-red-800`;
  //     default:
  //       return `${baseClasses} bg-gray-100 text-gray-800`;
  //   }
  // };

  useEffect(() => {
    const userConsent = localStorage.getItem("safeguardmedia_consent");
    const userHasUploadedBefore = localStorage.getItem(
      "safeguardmedia_uploaded"
    );

    if (userConsent !== null) {
      setHasConsented(userConsent === "true");
    }

    if (userHasUploadedBefore === "true") {
      setIsFirstTimeUser(false);
    }
  }, []);
  return (
    <div className={`min-h-screen bg-gray-50 overflow-x-hidden`}>
      {/* Full Width Header */}
      <header className="bg-white border-b border-gray-200 px-4 sm:px-6 py-4 w-full">
        <div className="flex items-center justify-between min-w-0">
          <div className="flex items-center space-x-2 sm:space-x-3 min-w-0">
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
                className="h-8 sm:h-12 w-auto flex-shrink-0"
              />
              <span className="text-base sm:text-xl font-bold text-gray-900 ml-2 truncate">
                Safeguardmedia
              </span>
            </div>
          </div>
          <div className="flex items-center space-x-2 sm:space-x-4 flex-shrink-0">
            <button
              className="hidden sm:block p-2 text-gray-400 hover:text-gray-600 bg-[#F6F7FE] rounded-[30px] border-[0.88px] border-[#8C8C8C]"
              onClick={() => {
                navigate("/notifications");
              }}
            >
              <Bell className="w-4 h-4 sm:w-5 sm:h-5" />
            </button>

            <div
              className="flex items-center space-x-2 cursor-pointer rounded-[30px] min-w-0"
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
                className="flex items-center space-x-3 text-gray-400  cursor-not-allowed"
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
      <div className="flex min-h-0 overflow-x-hidden">
        {/* Desktop Sidebar */}
        <div className="hidden lg:flex w-24 bg-white border-r border-gray-200 flex-col items-center py-6 space-y-8 min-h-[calc(100vh-73px)] flex-shrink-0">
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
            className="flex flex-col items-center space-y-2 text-gray-400  cursor-not-allowed"
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
        <div className="flex-1 flex flex-col min-w-0 overflow-x-hidden">
          {/* Upper Section: Upload Area + Right Sidebar */}
          <div className="flex flex-col xl:flex-row min-h-0">
            {/* Main Content Area */}
            <div className="w-full xl:w-2/3 p-4 sm:p-6 min-w-0">
              {/* Getting Started Section */}
              <div className="min-w-0">
                <h2 className="text-xl sm:text-2xl font-semibold text-gray-900 mb-2">
                  Let's get started!
                </h2>
                <p className="text-sm sm:text-base text-gray-600 mb-6">
                  Upload your file for analysis, supports video, audio, and
                  image formats.
                </p>

                {/* Upload Area */}
                <div
                  className={`border-2 border-dashed rounded-xl p-4 sm:p-6 lg:p-12 text-center transition-colors w-full ${
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
                    <div className="w-full">
                      <div className="flex justify-center items-center mb-4">
                        <UploadIcon />
                      </div>

                      <h3 className="text-base sm:text-lg font-medium text-gray-900 mb-2">
                        Drag and drop to upload or browse files
                      </h3>
                      <p className="text-xs sm:text-sm text-red-500 mb-6">
                        Max file size 10MB
                      </p>

                      {/* Supported Formats Section */}
                      <div className="bg-gray-50 rounded-lg p-3 sm:p-4 mb-6 text-left max-w-full overflow-hidden">
                        <h4 className="text-sm font-medium text-gray-700 mb-3">
                          Currently, SafeguardMedia supports the following
                          formats:
                        </h4>
                        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 text-xs">
                          <div className="min-w-0">
                            <span className="font-medium text-gray-800 block">
                              Videos:
                            </span>
                            <div className="text-gray-600 mt-1">
                              MP4, AVI, MOV
                            </div>
                          </div>
                          <div className="min-w-0">
                            <span className="font-medium text-gray-800 block">
                              Images:
                            </span>
                            <div className="text-gray-600 mt-1">
                              JPEG, PNG, WEBP
                            </div>
                          </div>
                          <div className="min-w-0">
                            <span className="font-medium text-gray-800 block">
                              Audio:
                            </span>
                            <div className="text-gray-600 mt-1">
                              MP3, WAV, AAC
                            </div>
                          </div>
                        </div>
                      </div>

                      {!isFirstTimeUser && (
                        <div className="mb-4 max-w-full">
                          <label className="flex items-start space-x-3 cursor-pointer text-left">
                            <input
                              type="checkbox"
                              checked={hasConsented}
                              onChange={async (e) => {
                                const newConsentValue = e.target.checked;
                                setHasConsented(newConsentValue);

                                localStorage.setItem(
                                  "safeguardmedia_consent",
                                  newConsentValue.toString()
                                );

                                try {
                                  await updateMediaConsent({
                                    allowStorage: !newConsentValue,
                                  }).unwrap();
                                } catch (error) {
                                  console.error(
                                    "Failed to update media consent:",
                                    error
                                  );
                                }
                              }}
                              className="mt-1 w-4 h-4 text-blue-600 border-gray-300 rounded focus:ring-blue-500 flex-shrink-0"
                            />
                            <span className="text-xs text-gray-600 leading-relaxed break-words">
                              I do not consent to SafeguardMedia using my
                              uploaded media for AI model training or research.
                              My upload should be used only for analysis and
                              detection.
                            </span>
                          </label>
                        </div>
                      )}

                      <button
                        className="bg-[#FBFBEF] border border-[#8C8C8C] rounded-[30px] hover:bg-gray-200 text-gray-700 px-4 sm:px-6 py-2 sm:py-3 text-sm sm:text-base font-medium w-full sm:w-auto max-w-xs"
                        onClick={() => {
                          setHasAnalyses(!hasAnalyses);
                          handleUploadMedia();
                        }}
                        disabled={isUploading}
                      >
                        {isUploading ? "Uploading..." : "Upload Media"}
                      </button>
                    </div>
                  ) : (
                    <div className="flex flex-col items-center space-y-4 w-full max-w-full">
                      {/* Video Thumbnail */}
                      <div className="w-32 h-20 sm:w-40 sm:h-24 rounded-lg overflow-hidden bg-gray-200 flex-shrink-0">
                        {filePreview ? (
                          <img
                            src={filePreview}
                            alt="File preview"
                            className="w-full h-full object-cover"
                          />
                        ) : selectedFile?.type.startsWith("audio/") ? (
                          <div className="w-full h-full flex items-center justify-center bg-gray-300">
                            <AudioLines className="w-8 h-8 text-gray-600" />
                          </div>
                        ) : (
                          <img
                            src={ThirdImage}
                            alt="File thumbnail"
                            className="w-full h-full object-cover"
                          />
                        )}
                      </div>

                      {/* File Info */}
                      <div className="flex flex-col sm:flex-row items-center justify-center w-full min-w-0 space-y-2 sm:space-y-0 sm:space-x-4">
                        <div className="flex flex-col sm:flex-row items-center gap-2 min-w-0 max-w-full">
                          <p className="text-sm sm:text-base font-medium text-gray-900 truncate max-w-full text-center sm:text-left">
                            {uploadedFile.name}
                          </p>
                          <p className="text-xs sm:text-sm text-gray-500 flex-shrink-0">
                            ({uploadedFile.size})
                          </p>
                        </div>
                        <button
                          onClick={handleRemoveFile}
                          className="p-1 text-red-500 hover:text-red-700 hover:bg-red-50 rounded-full flex-shrink-0"
                          title="Remove file"
                        >
                          <X className="w-4 h-4 sm:w-5 sm:h-5" />
                        </button>
                      </div>

                      {/* Analyse Button */}
                      <button
                        onClick={handleAnalyseMedia}
                        disabled={isAnalyzing || !selectedFile}
                        className="bg-gray-900 hover:bg-gray-800 disabled:bg-gray-400 disabled:cursor-not-allowed text-white px-6 sm:px-8 py-2 sm:py-3 rounded-full text-sm sm:text-base font-medium transition-colors w-full sm:w-auto max-w-xs"
                      >
                        {isAnalyzing ? "Analyzing..." : "Analyse Media"}
                      </button>

                      {analysisError && (
                        <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-lg w-full max-w-full">
                          <div className="flex items-start">
                            <AlertCircle className="w-4 h-4 mr-2 flex-shrink-0 mt-0.5 text-red-500" />
                            <div className="flex-1 min-w-0">
                              <p className="text-sm text-red-700 break-words">
                                {analysisError}
                              </p>
                            </div>
                            <button
                              type="button"
                              onClick={() => setAnalysisError(null)}
                              className="ml-2 text-red-400 hover:text-red-600 flex-shrink-0"
                              title="Dismiss error"
                            >
                              <X className="w-4 h-4" />
                            </button>
                          </div>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* Consent Modal */}
            {showConsentModal && (
              <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
                <div className="bg-white rounded-xl max-w-md w-full mx-4 p-6 max-h-[90vh] overflow-y-auto">
                  <h3 className="text-lg font-semibold text-gray-900 mb-4">
                    Processing Consent
                  </h3>

                  <div className="mb-6">
                    <label className="flex items-start space-x-3 cursor-pointer">
                      <input
                        type="checkbox"
                        checked={hasConsented}
                        onChange={(e) => setHasConsented(e.target.checked)}
                        className="mt-1 w-4 h-4 text-blue-600 border-gray-300 rounded focus:ring-blue-500 flex-shrink-0"
                      />
                      <span className="text-sm text-gray-700 leading-relaxed break-words">
                        I do not consent to SafeguardMedia using my uploaded
                        media for AI model training or research. My upload
                        should be used only for analysis and detection.
                      </span>
                    </label>
                  </div>

                  <div className="flex flex-col sm:flex-row gap-3">
                    <button
                      onClick={() => setShowConsentModal(false)}
                      className="flex-1 px-4 py-2 text-gray-700 border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors"
                    >
                      Cancel
                    </button>
                    <button
                      onClick={handleConsentSubmit}
                      className="flex-1 px-4 py-2 text-white rounded-lg font-medium bg-[#0F2FA3] hover:bg-blue-700 transition-colors"
                    >
                      Continue
                    </button>
                  </div>
                </div>
              </div>
            )}

            {/* Hidden file input */}
            <input
              type="file"
              ref={(input) => {
                if (input) {
                  input.onclick = () => {
                    input.value = "";
                  };
                }
              }}
              onChange={handleFileInputChange}
              accept=".mp4,.avi,.mov,.jpeg,.jpg,.png,.webp,.mp3,.wav,.aac"
              style={{ display: "none" }}
              id="file-upload-input"
            />

            {/* Right Sidebar */}
            <div className="w-full xl:w-1/3 p-4 sm:p-6 min-w-0 xl:pt-[100px]">
              {/* Combined Subscription and How it Works Card */}
              <div className="bg-white rounded-xl border border-gray-200 overflow-hidden flex flex-col w-full h-[300px]">
                {/* Subscription Header */}
                <div className="bg-[#0F2FA3] text-white px-4 sm:px-6 py-3 sm:py-4 flex-shrink-0">
                  <span className="text-xs sm:text-sm font-medium leading-tight">
                    Upgrade to Max plan for unlimited analysis
                  </span>
                </div>

                {/* How it Works Content */}
                <div className="p-4 sm:p-6 flex-1 flex flex-col justify-start overflow-y-auto">
                  <h3 className="text-base sm:text-lg font-semibold text-gray-900 mb-4">
                    How it Works
                  </h3>
                  <div className="space-y-3 sm:space-y-4">
                    <div className="flex items-start gap-3 sm:gap-4">
                      <div className="flex-shrink-0 text-gray-600 font-medium">
                        →
                      </div>
                      <p className="text-xs sm:text-sm text-gray-600 leading-relaxed">
                        Upload your media
                      </p>
                    </div>
                    <div className="flex items-start gap-3 sm:gap-4">
                      <div className="flex-shrink-0 text-gray-600 font-medium">
                        →
                      </div>
                      <p className="text-xs sm:text-sm text-gray-600 leading-relaxed">
                        Our model automatically processes the media to determine
                        whether it is AI generated or not.
                      </p>
                    </div>
                    <div className="flex items-start gap-3 sm:gap-4">
                      <div className="flex-shrink-0 text-gray-600 font-medium">
                        →
                      </div>
                      <p className="text-xs sm:text-sm text-gray-600 leading-relaxed">
                        Get instant and clear results with confidence.
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Recent Analyses Section - Full Width */}
          <div className="px-4 sm:px-6 pb-4 sm:pb-6 min-w-0">
            <div className="bg-white rounded-xl border border-gray-200 p-4 sm:p-6 w-full overflow-hidden">
              <div className="mb-6">
                <h3 className="text-base sm:text-lg font-semibold text-gray-900 mb-2">
                  Your Recent Analyses
                </h3>
                <p className="text-xs sm:text-sm text-gray-600 leading-relaxed">
                  Review the results of your recently uploaded media files.
                  Results are stored for 30 days.
                </p>
              </div>
              {/* Loading State */}
              {isHistoryLoading && (
                <div className="text-center py-8 sm:py-12 w-full">
                  <div className="flex justify-center items-center mb-4">
                    <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
                  </div>
                  <p className="text-sm text-gray-600">
                    Loading your analysis history...
                  </p>
                </div>
              )}

              {/* Error State */}
              {historyError && (
                <div className="text-center py-8 sm:py-12 w-full">
                  <div className="flex justify-center items-center mb-4">
                    <AlertCircle className="w-8 h-8 text-red-500" />
                  </div>
                  <h4 className="text-base sm:text-lg font-medium text-gray-900 mb-2">
                    Error Loading History
                  </h4>
                  <p className="text-xs sm:text-sm text-red-600">
                    Unable to load your analysis history. Please try refreshing
                    the page.
                  </p>
                </div>
              )}
              {historyData?.data && historyData.data.length > 0 ? (
                <div className="w-full">
                  {/* Desktop Table Header */}
                  <div className="hidden md:grid grid-cols-12 gap-4 pb-3 border-b border-gray-200 text-sm font-medium text-gray-500 min-w-0">
                    <div className="col-span-4">File name/thumbnail</div>
                    <div className="col-span-3">Upload date/time</div>
                    <div className="col-span-2">Status</div>
                    <div className="col-span-2">Confidence score</div>
                    <div className="col-span-1"></div>
                  </div>

                  {/* Analysis History Items */}
                  <div className="space-y-3 mt-4 w-full">
                    {getPaginatedData().map((analysis) => {
                      // Helper function to get file type icon
                      const getFileTypeIcon = (
                        fileName: string | undefined
                      ) => {
                        if (!fileName)
                          return (
                            <ImageIcon className="w-6 h-6 text-gray-600" />
                          );

                        const extension = fileName
                          .split(".")
                          .pop()
                          ?.toLowerCase();
                        switch (extension) {
                          case "mp4":
                          case "avi":
                          case "mov":
                            return <Video className="w-6 h-6 text-gray-600" />;
                          case "mp3":
                          case "wav":
                          case "aac":
                            return (
                              <AudioLines className="w-6 h-6 text-gray-600" />
                            );
                          case "jpg":
                          case "jpeg":
                          case "png":
                          case "webp":
                          default:
                            return (
                              <ImageIcon className="w-6 h-6 text-gray-600" />
                            );
                        }
                      };

                      // Helper function to format date
                      const formatDate = (dateString: string): string => {
                        const date = new Date(dateString);
                        return date.toLocaleDateString("en-US", {
                          month: "short",
                          day: "2-digit",
                          year: "numeric",
                          hour: "2-digit",
                          minute: "2-digit",
                          hour12: true,
                        });
                      };

                      // Helper function to get status badge
                      const getStatusBadge = (
                        status: string | undefined
                      ): string => {
                        const baseClasses =
                          "px-3 py-1 rounded-full text-xs font-medium whitespace-nowrap";
                        switch (status?.toLowerCase()) {
                          case "authentic":
                          case "real":
                            return `${baseClasses} bg-green-100 text-green-800`;
                          case "uncertain":
                          case "inconclusive":
                            return `${baseClasses} bg-yellow-100 text-yellow-800`;
                          case "deepfake":
                          case "fake":
                          case "synthetic":
                            return `${baseClasses} bg-red-100 text-red-800`;
                          default:
                            return `${baseClasses} bg-gray-100 text-gray-800`;
                        }
                      };

                      return (
                        <div key={analysis._id} className="w-full">
                          {/* Desktop View */}
                          <div className="hidden md:grid grid-cols-12 gap-4 items-center py-3 hover:bg-gray-50 rounded-lg min-w-0 transition-colors">
                            <div className="col-span-4 flex items-center space-x-3 min-w-0">
                              <div className="w-10 h-10 rounded-lg flex items-center justify-center bg-gray-100 flex-shrink-0">
                                {analysis.thumbnailUrl ? (
                                  <img
                                    src={analysis.thumbnailUrl}
                                    alt="File thumbnail"
                                    className="w-full h-full object-cover rounded-lg"
                                    onError={(e) => {
                                      // Fallback to icon if thumbnail fails to load
                                      const target =
                                        e.target as HTMLImageElement;
                                      target.style.display = "none";
                                      target.nextElementSibling?.classList.remove(
                                        "hidden"
                                      );
                                    }}
                                  />
                                ) : null}
                                <div
                                  className={
                                    analysis.thumbnailUrl ? "hidden" : ""
                                  }
                                >
                                  {getFileTypeIcon(analysis.fileName)}
                                </div>
                              </div>
                              <span
                                className="text-sm font-medium text-gray-900 truncate"
                                title={
                                  analysis.fileName ||
                                  `Media_${analysis._id.slice(-8)}`
                                }
                              >
                                {analysis.fileName ||
                                  `Media_${analysis._id.slice(-8)}`}
                              </span>
                            </div>
                            <div className="col-span-3 min-w-0">
                              <span className="text-sm text-gray-600 break-words">
                                {formatDate(analysis.uploadDate)}
                              </span>
                            </div>
                            <div className="col-span-2 min-w-0">
                              <span className={getStatusBadge(analysis.status)}>
                                {analysis.status || "Unknown"}
                              </span>
                            </div>
                            <div className="col-span-2 min-w-0">
                              <span className="text-sm font-medium text-gray-900">
                                {analysis.confidenceScore
                                  ? `${Math.round(analysis.confidenceScore)}%`
                                  : "N/A"}
                              </span>
                            </div>
                            <div className="col-span-1"></div>
                          </div>

                          {/* Mobile View */}
                          <div className="md:hidden bg-gray-50 rounded-lg p-4 hover:bg-gray-100 w-full transition-colors">
                            <div className="flex items-start space-x-3 min-w-0">
                              <div className="w-12 h-12 rounded-lg flex items-center justify-center bg-gray-100 flex-shrink-0">
                                {analysis.thumbnailUrl ? (
                                  <img
                                    src={analysis.thumbnailUrl}
                                    alt="File thumbnail"
                                    className="w-full h-full object-cover rounded-lg"
                                    onError={(e) => {
                                      // Fallback to icon if thumbnail fails to load
                                      const target =
                                        e.target as HTMLImageElement;
                                      target.style.display = "none";
                                      target.nextElementSibling?.classList.remove(
                                        "hidden"
                                      );
                                    }}
                                  />
                                ) : null}
                                <div
                                  className={
                                    analysis.thumbnailUrl ? "hidden" : ""
                                  }
                                >
                                  {getFileTypeIcon(analysis.fileName)}
                                </div>
                              </div>
                              <div className="flex-1 min-w-0">
                                <div className="flex flex-col space-y-2">
                                  <div className="min-w-0">
                                    <p
                                      className="text-sm font-medium text-gray-900 truncate"
                                      title={
                                        analysis.fileName ||
                                        `Media_${analysis._id.slice(-8)}`
                                      }
                                    >
                                      {analysis.fileName ||
                                        `Media_${analysis._id.slice(-8)}`}
                                    </p>
                                    <p className="text-xs text-gray-600 mt-1 break-words">
                                      {formatDate(analysis.uploadDate)}
                                    </p>
                                  </div>
                                  <div className="flex items-center justify-between gap-2 flex-wrap">
                                    <span
                                      className={getStatusBadge(
                                        analysis.status
                                      )}
                                    >
                                      {analysis.status || "Unknown"}
                                    </span>
                                    <span className="text-sm font-medium text-gray-900 flex-shrink-0">
                                      {analysis.confidenceScore
                                        ? `${Math.round(
                                            analysis.confidenceScore
                                          )}%`
                                        : "N/A"}
                                    </span>
                                  </div>
                                </div>
                              </div>
                            </div>
                          </div>
                        </div>
                      );
                    })}
                  </div>

                  {/* Pagination - Show pagination info using API response */}
                  {historyData.pagination &&
                    historyData.pagination.totalPages > 1 && (
                      <div className="flex items-center justify-between mt-6 pt-4 border-t border-gray-200">
                        {/* Items per page info - Using API pagination data */}
                        <div className="text-sm text-gray-600">
                          Showing{" "}
                          {(historyData.pagination.currentPage - 1) *
                            historyData.pagination.itemsPerPage +
                            1}{" "}
                          to{" "}
                          {Math.min(
                            historyData.pagination.currentPage *
                              historyData.pagination.itemsPerPage,
                            historyData.pagination.totalItems
                          )}{" "}
                          of {historyData.pagination.totalItems} analyses
                        </div>

                        {/* Pagination controls */}
                        <div className="flex items-center space-x-1 sm:space-x-2">
                          {/* Previous button */}
                          <button
                            className="p-1.5 sm:p-2 hover:bg-gray-100 rounded disabled:opacity-50 disabled:cursor-not-allowed"
                            onClick={() => {
                              // For now, just update local state since API doesn't accept query params yet
                              if (historyData.pagination.hasPreviousPage) {
                                setCurrentPage(
                                  historyData.pagination.previousPage ||
                                    currentPage - 1
                                );
                              }
                            }}
                            disabled={!historyData.pagination.hasPreviousPage}
                          >
                            <ChevronLeft className="w-3 h-3 sm:w-4 sm:h-4 text-gray-400" />
                          </button>

                          {/* Dynamic page numbers */}
                          <div className="flex space-x-1 overflow-x-auto">
                            {getPageNumbers().map((page, index) =>
                              page === "ellipsis" ? (
                                <span
                                  key={`ellipsis-${index}`}
                                  className="px-2 py-1 text-gray-400"
                                >
                                  ...
                                </span>
                              ) : (
                                <button
                                  key={page}
                                  className={`w-6 h-6 sm:w-8 sm:h-8 rounded text-xs sm:text-sm font-medium flex-shrink-0 transition-colors ${
                                    historyData.pagination.currentPage === page
                                      ? "bg-[#0F2FA3] text-white"
                                      : "text-gray-600 hover:bg-gray-100"
                                  }`}
                                  onClick={() => {
                                    // For now, just update local state since API doesn't accept query params yet
                                    setCurrentPage(page as number);
                                  }}
                                >
                                  {page}
                                </button>
                              )
                            )}
                          </div>

                          {/* Next button */}
                          <button
                            className="p-1.5 sm:p-2 hover:bg-gray-100 rounded disabled:opacity-50 disabled:cursor-not-allowed"
                            onClick={() => {
                              // For now, just update local state since API doesn't accept query params yet
                              if (historyData.pagination.hasNextPage) {
                                setCurrentPage(
                                  historyData.pagination.nextPage ||
                                    currentPage + 1
                                );
                              }
                            }}
                            disabled={!historyData.pagination.hasNextPage}
                          >
                            <ChevronRight className="w-3 h-3 sm:w-4 sm:h-4 text-gray-400" />
                          </button>
                        </div>
                      </div>
                    )}
                </div>
              ) : (
                // No Analyses Yet - Empty State
                <div className="text-center py-8 sm:py-12 w-full">
                  <div className="flex justify-center items-center mb-4">
                    <NoAnalysisYet />
                  </div>
                  <h4 className="text-base sm:text-lg font-medium text-gray-900 mb-2">
                    No Analyses Yet!
                  </h4>
                  <div className="space-y-1 max-w-md mx-auto">
                    <p className="text-xs sm:text-sm text-gray-600">
                      Upload your first video, audio, or image file to check for
                      manipulation.
                    </p>
                    <p className="text-xs sm:text-sm text-gray-600">
                      Our model will provide a quick and clear assessment.
                    </p>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
