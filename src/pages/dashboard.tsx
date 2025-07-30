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
  MoreHorizontal,
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
  useGetUserQuery,
  useUpdateMediaConsentMutation,
  useDetectAnalyzeMutation,
} from "../services/apiService";
import SafeguardMediaLogo from "../assets/images/SafeguardMedia8.svg";
import type { AnalysisHistory } from "../services/apiService";
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
  const { data: userData } = useGetUserQuery();
  const storedUser = useSelector((state: RootState) => state.user.user);
  const [updateMediaConsent] = useUpdateMediaConsentMutation();
  const [detectAnalyze] = useDetectAnalyzeMutation();

  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisError, setAnalysisError] = useState<string | null>(null);

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
                        Max file size 10MB
                      </p>
                      {/* <button
                        className="bg-[#FBFBEF] border border-[#8C8C8C] rounded-[30px] hover:bg-gray-200 text-gray-700 px-4 sm:px-6 py-2 sm:py-3 text-sm sm:text-base font-medium"
                        onClick={() => {
                          setHasAnalyses(!hasAnalyses);
                          handleUploadMedia();
                        }}
                        disabled={isUploading}
                      >
                        Upload Media
                      </button> */}
                      {/* Supported Formats Section */}
                      <div className="bg-gray-50 rounded-lg p-4 mb-6 text-left">
                        <h4 className="text-sm font-medium text-gray-700 mb-3">
                          Currently, SafeguardMedia supports the following
                          formats:
                        </h4>
                        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 text-xs">
                          <div>
                            <span className="font-medium text-gray-800">
                              Videos:
                            </span>
                            <div className="text-gray-600 mt-1">
                              MP4, AVI, MOV
                            </div>
                          </div>
                          <div>
                            <span className="font-medium text-gray-800">
                              Images:
                            </span>
                            <div className="text-gray-600 mt-1">
                              JPEG, PNG, WEBP
                            </div>
                          </div>
                          <div>
                            <span className="font-medium text-gray-800">
                              Audio:
                            </span>
                            <div className="text-gray-600 mt-1">
                              MP3, WAV, AAC
                            </div>
                          </div>
                        </div>
                      </div>
                      {!isFirstTimeUser && (
                        <div className="mb-4">
                          <label className="flex items-start space-x-3 cursor-pointer text-left">
                            <input
                              type="checkbox"
                              checked={hasConsented}
                              onChange={async (e) => {
                                const newConsentValue = e.target.checked;
                                setHasConsented(newConsentValue);

                                // Store consent choice when user changes it
                                localStorage.setItem(
                                  "safeguardmedia_consent",
                                  newConsentValue.toString()
                                );

                                // Call API with inverted logic:
                                // checked = user does NOT consent = allowStorage: false
                                // unchecked = user consents = allowStorage: true
                                try {
                                  await updateMediaConsent({
                                    allowStorage: !newConsentValue,
                                  }).unwrap();
                                } catch (error) {
                                  console.error(
                                    "Failed to update media consent:",
                                    error
                                  );
                                  // Optionally revert the state if API call fails
                                  // setHasConsented(!newConsentValue);
                                  // localStorage.setItem("safeguardmedia_consent", (!newConsentValue).toString());
                                }
                              }}
                              className="mt-1 w-4 h-4 text-blue-600 border-gray-300 rounded focus:ring-blue-500"
                            />
                            <span className="text-xs text-gray-600 leading-relaxed">
                              I do not consent to SafeguardMedia using my
                              uploaded media for AI model training or research.
                              My upload should be used only for analysis and
                              detection.
                            </span>
                          </label>
                        </div>
                      )}

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
                        disabled={isAnalyzing || !selectedFile}
                        className="bg-gray-900 hover:bg-gray-800 disabled:bg-gray-400 disabled:cursor-not-allowed text-white px-6 sm:px-8 py-2 sm:py-3 rounded-full text-sm sm:text-base font-medium transition-colors"
                      >
                        {isAnalyzing ? "Analyzing..." : "Analyse Media"}
                      </button>

                      {analysisError && (
                        <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-lg">
                          <AlertCircle className="w-4 h-4 mr-2 flex-shrink-0" />
                          <p className="text-sm text-red-700">
                            {analysisError}
                          </p>

                          <button
                            type="button"
                            onClick={() => setAnalysisError(null)}
                            className="ml-auto text-red-400 hover:text-red-600"
                          >
                            <X className="w-4 h-4" />
                          </button>
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
                <div className="bg-white rounded-xl max-w-md w-full p-6">
                  <h3 className="text-lg font-semibold text-gray-900 mb-4">
                    Processing Consent
                  </h3>

                  <div className="mb-6">
                    <label className="flex items-start space-x-3 cursor-pointer">
                      <input
                        type="checkbox"
                        checked={hasConsented}
                        onChange={(e) => setHasConsented(e.target.checked)}
                        className="mt-1 w-4 h-4 text-blue-600 border-gray-300 rounded focus:ring-blue-500"
                      />

                      <span className="text-sm text-gray-700 leading-relaxed">
                        I do not consent to SafeguardMedia using my uploaded
                        media for AI model training or research. My upload
                        should be used only for analysis and detection.
                      </span>
                    </label>
                  </div>

                  <div className="flex space-x-3">
                    <button
                      onClick={() => setShowConsentModal(false)}
                      className="flex-1 px-4 py-2 text-gray-700 border border-gray-300 rounded-lg hover:bg-gray-50"
                    >
                      Cancel
                    </button>
                    <button
                      onClick={handleConsentSubmit}
                      className="flex-1 px-4 py-2 text-white rounded-lg font-medium bg-[#0F2FA3] hover:bg-blue-700"
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

            <div className="w-full lg:w-1/3 p-4 sm:p-6 lg:mt-22">
              {/* Combined Subscription and How it Works Card */}
              <div
                className="bg-white rounded-xl border border-gray-200 overflow-hidden flex flex-col"
                style={{ height: "300px" }}
              >
                {/* Subscription Header */}
                <div className="bg-[#0F2FA3] text-white px-4 sm:px-6 py-3 sm:py-4 flex-shrink-0">
                  <span className="text-xs sm:text-sm font-medium">
                    Upgrade to Max plan for unlimited analysis
                  </span>
                </div>

                {/* How it Works Content */}
                <div className="p-4 sm:p-6 flex-1 flex flex-col justify-start">
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

              {userData?.data?.user?.analysisHistory &&
              userData.data.user.analysisHistory.length > 0 ? (
                <div>
                  <div className="hidden md:grid grid-cols-12 gap-4 pb-3 border-b border-gray-200 text-sm font-medium text-gray-500">
                    <div className="col-span-4">File name/thumbnail</div>
                    <div className="col-span-3">Upload date/time</div>
                    <div className="col-span-2">Status</div>
                    <div className="col-span-2">Confidence score</div>
                    <div className="col-span-1"></div>
                  </div>

                  <div className="space-y-3 mt-4">
                    {userData.data.user.analysisHistory.map(
                      (analysis: AnalysisHistory) => {
                        // Helper function to get file type icon
                        const getFileTypeIcon = (type: string | undefined) => {
                          switch (type?.toLowerCase()) {
                            case "video":
                              return (
                                <Video className="w-6 h-6 text-gray-600" />
                              );
                            case "audio":
                              return (
                                <AudioLines className="w-6 h-6 text-gray-600" />
                              );
                            case "image":
                              return (
                                <ImageIcon className="w-6 h-6 text-gray-600" />
                              );
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
                          result: string | undefined
                        ): string => {
                          const baseClasses =
                            "px-3 py-1 rounded-full text-xs font-medium";
                          switch (result?.toLowerCase()) {
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

                        // Helper function to extract confidence from result if available
                        const getConfidence = (
                          analysis: AnalysisHistory
                        ): string => {
                          // If there's a separate confidence field, use it
                          if (analysis.confidence !== undefined) {
                            return `${analysis.confidence}%`;
                          }
                          // Otherwise, return a placeholder or extract from result if it contains confidence info
                          return "N/A";
                        };

                        return (
                          <div key={analysis.id}>
                            {/* Desktop View */}
                            <div className="hidden md:grid grid-cols-12 gap-4 items-center py-3 hover:bg-gray-50 rounded-lg">
                              <div className="col-span-4 flex items-center space-x-3">
                                <div className="w-10 h-10 rounded-lg flex items-center justify-center bg-gray-100">
                                  {getFileTypeIcon(analysis.type)}
                                </div>
                                <span className="text-sm font-medium text-gray-900 truncate">
                                  {analysis.fileName ||
                                    `${
                                      analysis.type || "Media"
                                    }_${analysis.id.slice(-8)}`}
                                </span>
                              </div>
                              <div className="col-span-3">
                                <span className="text-sm text-gray-600">
                                  {formatDate(analysis.createdAt)}
                                </span>
                              </div>
                              <div className="col-span-2">
                                <span
                                  className={getStatusBadge(analysis.result)}
                                >
                                  {analysis.result || "Unknown"}
                                </span>
                              </div>
                              <div className="col-span-2">
                                <span className="text-sm font-medium text-gray-900">
                                  {getConfidence(analysis)}
                                </span>
                              </div>
                              <div className="col-span-1 flex justify-end">
                                <button className="p-1 hover:bg-gray-100 rounded">
                                  <MoreHorizontal className="w-4 h-4 text-gray-400" />
                                </button>
                              </div>
                            </div>

                            {/* Mobile View */}
                            <div className="md:hidden bg-gray-50 rounded-lg p-4 hover:bg-gray-100">
                              <div className="flex items-start space-x-3">
                                <div className="w-12 h-12 rounded-lg flex items-center justify-center bg-gray-100 flex-shrink-0">
                                  {getFileTypeIcon(analysis.type)}
                                </div>
                                <div className="flex-1 min-w-0">
                                  <div className="flex items-start justify-between">
                                    <div className="flex-1 min-w-0">
                                      <p className="text-sm font-medium text-gray-900 truncate">
                                        {analysis.fileName ||
                                          `${
                                            analysis.type || "Media"
                                          }_${analysis.id.slice(-8)}`}
                                      </p>
                                      <p className="text-xs text-gray-600 mt-1">
                                        {formatDate(analysis.createdAt)}
                                      </p>
                                    </div>
                                    <button className="p-1 hover:bg-gray-200 rounded ml-2">
                                      <MoreHorizontal className="w-4 h-4 text-gray-400" />
                                    </button>
                                  </div>
                                  <div className="flex items-center justify-between mt-3">
                                    <span
                                      className={getStatusBadge(
                                        analysis.result
                                      )}
                                    >
                                      {analysis.result || "Unknown"}
                                    </span>
                                    <span className="text-sm font-medium text-gray-900">
                                      {getConfidence(analysis)}
                                    </span>
                                  </div>
                                </div>
                              </div>
                            </div>
                          </div>
                        );
                      }
                    )}
                  </div>

                  {/* Pagination - Show only if there are many analyses */}
                  {userData.data.user.analysisHistory.length > 5 && (
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
                  )}
                </div>
              ) : (
                // No Analyses Yet - Show when analysisHistory is empty or undefined
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
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
