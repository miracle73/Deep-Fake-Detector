import { useState, useEffect } from "react";
import {
  Bell,
  LayoutGrid,
  Video,
  ImageIcon,
  AudioLines,
  Menu,
  X,
  AlertCircle,
  Loader,
} from "lucide-react";
import {
  useUpdateUserMutation,
  useDeleteUserMutation,
} from "../services/apiService";
import { useNavigate } from "react-router-dom";
import { useGetUserQuery } from "../services/apiService";
import SafeguardMediaLogo from "../assets/images/SafeguardMedia8.svg";
import { CiSettings } from "react-icons/ci";

const Settings = () => {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [personalInfo, setPersonalInfo] = useState({
    firstName: "",
    lastName: "",
    email: "",
    phone: "",
  });
  const [emailSettings, setEmailSettings] = useState({
    unsubscribeAll: false,
  });
  const navigate = useNavigate();
  const handlePersonalInfoChange = (field: string, value: string) => {
    setPersonalInfo((prev) => ({
      ...prev,
      [field]: value,
    }));
  };
  const [updateUser] = useUpdateUserMutation();
  const [deleteUser] = useDeleteUserMutation();
  const [isUpdatingPersonalInfo, setIsUpdatingPersonalInfo] = useState(false);
  const [isDeletingAccount, setIsDeletingAccount] = useState(false);
  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const [deleteReason, setDeleteReason] = useState("");
  const [otherReason, setOtherReason] = useState("");
  const [errors, setErrors] = useState({
    personalInfo: "",
    deleteAccount: "",
  });
  const [successMessages, setSuccessMessages] = useState({
    personalInfo: "",
  });
  const { data: userData } = useGetUserQuery();
  useEffect(() => {
    if (errors.personalInfo) {
      const timer = setTimeout(() => {
        setErrors((prev) => ({ ...prev, personalInfo: "" }));
      }, 3000);

      return () => clearTimeout(timer);
    }
  }, [errors.personalInfo]);

  useEffect(() => {
    if (errors.deleteAccount) {
      const timer = setTimeout(() => {
        setErrors((prev) => ({ ...prev, deleteAccount: "" }));
      }, 3000);

      return () => clearTimeout(timer);
    }
  }, [errors.deleteAccount]);
  // Replace the existing handlePersonalInfoSave function
  const handlePersonalInfoSave = async () => {
    // Clear previous messages
    setErrors((prev) => ({ ...prev, personalInfo: "" }));
    setSuccessMessages((prev) => ({ ...prev, personalInfo: "" }));

    // Basic validation
    if (
      !personalInfo.firstName.trim() ||
      !personalInfo.lastName.trim() ||
      !personalInfo.email.trim()
    ) {
      setErrors((prev) => ({
        ...prev,
        personalInfo: "Please fill in all required fields",
      }));
      return;
    }

    // Email validation
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(personalInfo.email)) {
      setErrors((prev) => ({
        ...prev,
        personalInfo: "Please enter a valid email address",
      }));
      return;
    }

    setIsUpdatingPersonalInfo(true);

    try {
      const updateData = {
        firstName: personalInfo.firstName.trim(),
        lastName: personalInfo.lastName.trim(),
        email: personalInfo.email.trim().toLowerCase(),
        phone: personalInfo.phone.trim(),
      };

      const result = await updateUser(updateData).unwrap();

      console.log("Personal info updated successfully:", result);
      setSuccessMessages((prev) => ({
        ...prev,
        personalInfo: "Personal information updated successfully!",
      }));
      navigate("/dashboard"); // Redirect to dashboard after successful update
      // Clear success message after 3 seconds
      setTimeout(() => {
        setSuccessMessages((prev) => ({ ...prev, personalInfo: "" }));
      }, 3000);
    } catch (error) {
      console.error("Update failed:", error);

      if (error && typeof error === "object" && "data" in error) {
        const apiError = error as { data?: { message?: string } };
        if (apiError.data?.message) {
          setErrors((prev) => ({
            ...prev,
            personalInfo:
              apiError.data?.message ??
              "Failed to update personal information. Please try again.",
          }));
        } else {
          setErrors((prev) => ({
            ...prev,
            personalInfo:
              "Failed to update personal information. Please try again.",
          }));
        }
      } else if (error && typeof error === "object" && "message" in error) {
        const messageError = error as { message?: string };
        setErrors((prev) => ({
          ...prev,
          personalInfo:
            messageError.message ??
            "Failed to update personal information. Please try again.",
        }));
      } else {
        setErrors((prev) => ({
          ...prev,
          personalInfo:
            "Failed to update personal information. Please try again.",
        }));
      }
    } finally {
      setIsUpdatingPersonalInfo(false);
    }
  };

  const handleDeleteModalSubmit = () => {
    if (!deleteReason && !otherReason) {
      alert("Please select a reason before proceeding.");
      return;
    }

    const finalReason = deleteReason === "other" ? otherReason : deleteReason;
    console.log("Delete reason:", finalReason);

    handleDeleteAccount();
  };

  const handleDeleteAccount = async () => {
    // Clear previous messages
    setErrors((prev) => ({ ...prev, deleteAccount: "" }));

    setIsDeletingAccount(true);

    try {
      const result = await deleteUser().unwrap();

      console.log("Account deleted successfully:", result);

      // Redirect to signin page or home page after successful deletion
      navigate("/signup");
    } catch (error) {
      console.error("Delete account failed:", error);

      if (error && typeof error === "object" && "data" in error) {
        const apiError = error as { data?: { message?: string } };
        if (
          apiError.data &&
          typeof apiError.data === "object" &&
          "message" in apiError.data
        ) {
          setErrors((prev) => ({
            ...prev,
            deleteAccount:
              (apiError.data as { message?: string }).message ??
              "Failed to delete account. Please try again.",
          }));
        } else {
          setErrors((prev) => ({
            ...prev,
            deleteAccount: "Failed to delete account. Please try again.",
          }));
        }
      } else if (error && typeof error === "object" && "message" in error) {
        const messageError = error;
        setErrors((prev) => ({
          ...prev,
          deleteAccount:
            typeof messageError.message === "string"
              ? messageError.message
              : "Failed to delete account. Please try again.",
        }));
      } else {
        setErrors((prev) => ({
          ...prev,
          deleteAccount: "Failed to delete account. Please try again.",
        }));
      }
    } finally {
      setIsDeletingAccount(false);
      setShowDeleteModal(false);
    }
  };

  const handleEmailSettingsSave = () => {
    console.log("Saving email settings:", emailSettings);
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
              <span className="text-xl max-lg:text-sm font-bold text-gray-900">
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
          {/* Main Content Area - Full Width */}
          <div className="w-full p-4 sm:p-6">
            {/* Getting Started Section */}
            <div className="mb-8">
              <h2 className="text-xl sm:text-2xl font-semibold text-gray-900 mb-2">
                Settings
              </h2>
              <p className="text-sm sm:text-base text-gray-600 mb-6">
                {/* Welcome back, Username */}
                Welcome back {userData?.data?.user?.firstName || "Username"}
              </p>
            </div>

            {/* Settings Form */}
            <div className="w-full bg-white rounded-xl border border-[#8C8C8C] p-6 sm:p-8 space-y-8">
              {/* Personal Information Section */}
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                {/* Left Column - Section Info and Button */}
                <div className="space-y-6">
                  <div>
                    <h3 className="text-lg font-semibold text-gray-900 mb-2">
                      Personal Information
                    </h3>
                    <p className="text-sm text-gray-600 mb-6">
                      Update your personal details here
                    </p>
                  </div>

                  {/* Error Message for Personal Info */}
                  {errors.personalInfo && (
                    <div className="flex items-center p-3 text-sm text-red-600 bg-red-50 border border-red-200 rounded-lg">
                      <AlertCircle className="w-4 h-4 mr-2 flex-shrink-0" />
                      <span>{errors.personalInfo}</span>
                    </div>
                  )}

                  {/* Success Message for Personal Info */}
                  {successMessages.personalInfo && (
                    <div className="flex items-center p-3 text-sm text-green-600 bg-green-50 border border-green-200 rounded-lg">
                      <svg
                        className="w-4 h-4 mr-2 flex-shrink-0"
                        fill="currentColor"
                        viewBox="0 0 20 20"
                      >
                        <path
                          fillRule="evenodd"
                          d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z"
                          clipRule="evenodd"
                        />
                      </svg>
                      <span>{successMessages.personalInfo}</span>
                    </div>
                  )}
                  {/* Save Changes Button */}
                  <div className="flex justify-start">
                    <button
                      onClick={handlePersonalInfoSave}
                      disabled={isUpdatingPersonalInfo}
                      className="bg-[#FBFBEF] border border-[#8C8C8C] hover:bg-gray-200 disabled:bg-gray-100 disabled:cursor-not-allowed text-gray-700 px-10 py-2 rounded-full text-sm font-medium transition-colors flex items-center"
                    >
                      {isUpdatingPersonalInfo ? (
                        <>
                          <Loader className="animate-spin -ml-1 mr-2 h-4 w-4" />
                          Saving...
                        </>
                      ) : (
                        "Save Changes"
                      )}
                    </button>
                  </div>
                </div>

                {/* Right Column - Form Fields */}
                <div className="space-y-4">
                  {/* First Name and Last Name Row */}
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                    <div>
                      <label
                        htmlFor="firstName"
                        className="block text-sm font-medium text-gray-700 mb-2"
                      >
                        First name
                      </label>
                      <input
                        type="text"
                        id="firstName"
                        placeholder="placeholder"
                        value={personalInfo.firstName}
                        onChange={(e) =>
                          handlePersonalInfoChange("firstName", e.target.value)
                        }
                        className="w-full px-4 py-3 border border-gray-300 rounded-full focus:outline-none focus:ring-2 focus:ring-[#0F2FA3] focus:border-transparent text-sm"
                      />
                    </div>
                    <div>
                      <label
                        htmlFor="lastName"
                        className="block text-sm font-medium text-gray-700 mb-2"
                      >
                        Last name
                      </label>
                      <input
                        type="text"
                        id="lastName"
                        placeholder="placeholder"
                        value={personalInfo.lastName}
                        onChange={(e) =>
                          handlePersonalInfoChange("lastName", e.target.value)
                        }
                        className="w-full px-4 py-3 border border-gray-300 rounded-full focus:outline-none focus:ring-2 focus:ring-[#0F2FA3] focus:border-transparent text-sm"
                      />
                    </div>
                  </div>

                  {/* Email Address */}
                  <div>
                    <label
                      htmlFor="email"
                      className="block text-sm font-medium text-gray-700 mb-2"
                    >
                      Email address
                    </label>
                    <input
                      type="email"
                      id="email"
                      placeholder="placeholder"
                      value={personalInfo.email}
                      onChange={(e) =>
                        handlePersonalInfoChange("email", e.target.value)
                      }
                      className="w-full px-4 py-3 border border-gray-300 rounded-full focus:outline-none focus:ring-2 focus:ring-[#0F2FA3] focus:border-transparent text-sm"
                    />
                  </div>

                  {/* Phone Number */}
                  <div>
                    <label
                      htmlFor="phone"
                      className="block text-sm font-medium text-gray-700 mb-2"
                    >
                      Phone number
                    </label>
                    <input
                      type="tel"
                      id="phone"
                      placeholder="placeholder"
                      value={personalInfo.phone}
                      onChange={(e) =>
                        handlePersonalInfoChange("phone", e.target.value)
                      }
                      className="w-full px-4 py-3 border border-gray-300 rounded-full focus:outline-none focus:ring-2 focus:ring-[#0F2FA3] focus:border-transparent text-sm"
                    />
                  </div>
                </div>
              </div>

              {/* Email Settings Section */}
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 border-t border-[#8C8C8C]  pt-8">
                {/* Left Column - Section Info and Button */}

                <div className="space-y-6">
                  <div>
                    <h3 className="text-lg font-semibold text-gray-900 mb-2">
                      Email Settings
                    </h3>
                    <p className="text-sm text-gray-600 mb-6">
                      Get firsthand information on product updates
                    </p>
                  </div>

                  {/* Save Changes Button */}
                  <div className="flex justify-start">
                    <button
                      onClick={handleEmailSettingsSave}
                      className="bg-[#FBFBEF] border border-[#8C8C8C] hover:bg-gray-200 text-gray-700 px-10 py-2 rounded-full text-sm font-medium transition-colors"
                    >
                      Save Changes
                    </button>
                  </div>
                </div>

                {/* Right Column - Checkbox */}
                <div className="flex items-start pt-8">
                  <div className="flex items-center space-x-3">
                    <input
                      type="checkbox"
                      id="unsubscribeAll"
                      checked={emailSettings.unsubscribeAll}
                      onChange={(e) =>
                        setEmailSettings((prev) => ({
                          ...prev,
                          unsubscribeAll: e.target.checked,
                        }))
                      }
                      className="w-4 h-4 text-[#0F2FA3] bg-gray-100 border-gray-300 rounded focus:ring-[#0F2FA3] focus:ring-2"
                    />
                    <label
                      htmlFor="unsubscribeAll"
                      className="text-sm text-gray-700"
                    >
                      Unsubscribe all emails
                    </label>
                  </div>
                </div>
              </div>

              {/* Sign Out Section */}
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 border-t border-[#8C8C8C] pt-8">
                {/* Left Column - Section Info and Button */}
                <div className="space-y-6">
                  <div>
                    <h3 className="text-lg font-semibold text-gray-900 mb-2">
                      Sign Out
                    </h3>
                    <p className="text-sm text-gray-600 mb-6">
                      Sign out of your account. You can always sign back in
                      later.
                    </p>
                  </div>

                  {/* Sign Out Button */}
                  <div className="flex justify-start">
                    <button
                      onClick={() => {
                        // Add your signout logic here
                        localStorage.removeItem("authToken"); // or however you handle auth
                        navigate("/signin");
                      }}
                      className="bg-[#FBFBEF] border border-[#8C8C8C] hover:bg-gray-200 text-gray-700 px-10 py-2 rounded-full text-sm font-medium transition-colors"
                    >
                      Sign Out
                    </button>
                  </div>
                </div>
              </div>

              <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 border-t border-[#8C8C8C] pt-8">
                {/* Left Column - Section Info and Button */}
                <div className="space-y-6">
                  <div>
                    <h3 className="text-lg font-semibold text-gray-900 mb-2">
                      Delete Account
                    </h3>
                    <p className="text-sm text-gray-600 mb-6">
                      Once you delete your account, there is no going back.
                      Please be certain. This action will permanently delete
                      your account, all your data, and cannot be undone.
                    </p>
                  </div>
                  {errors.deleteAccount && (
                    <div className="flex items-center p-3 text-sm text-red-600 bg-red-50 border border-red-200 rounded-lg">
                      <AlertCircle className="w-4 h-4 mr-2 flex-shrink-0" />
                      <span>{errors.deleteAccount}</span>
                    </div>
                  )}

                  {/* Delete Account Button */}
                  <div className="flex justify-start">
                    <button
                      onClick={() => setShowDeleteModal(true)}
                      disabled={isDeletingAccount}
                      className="bg-red-50 border border-red-300 hover:bg-red-100 disabled:bg-gray-100 disabled:cursor-not-allowed text-red-700 px-10 py-2 rounded-full text-sm font-medium transition-colors flex items-center"
                    >
                      {isDeletingAccount ? (
                        <>
                          <Loader className="animate-spin -ml-1 mr-2 h-4 w-4" />
                          Deleting...
                        </>
                      ) : (
                        "Delete Account"
                      )}
                    </button>
                  </div>
                </div>

                {/* Right Column - Warning Message */}
                <div className="bg-red-50 border border-red-200 rounded-lg p-4">
                  <div className="flex items-start space-x-3">
                    <div className="flex-shrink-0">
                      <svg
                        className="w-5 h-5 text-red-400"
                        fill="currentColor"
                        viewBox="0 0 20 20"
                      >
                        <path
                          fillRule="evenodd"
                          d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z"
                          clipRule="evenodd"
                        />
                      </svg>
                    </div>
                    <div>
                      <h4 className="text-sm font-medium text-red-800">
                        Warning: This action is irreversible
                      </h4>
                      <p className="text-sm text-red-700 mt-1">
                        Deleting your account will:
                      </p>
                      <ul className="text-sm text-red-700 mt-2 space-y-1">
                        <li>• Permanently delete all your personal data</li>
                        <li>• Remove access to all your processed media</li>
                        <li>• Cancel any active subscriptions</li>
                        <li>• Delete your account history</li>
                      </ul>
                    </div>
                  </div>
                </div>
              </div>

              {showDeleteModal && (
                <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
                  <div
                    className="fixed inset-0 backdrop-blur-sm bg-black bg-opacity-50"
                    onClick={() => setShowDeleteModal(false)}
                  />
                  <div className="relative bg-white rounded-lg max-w-md w-full p-6 space-y-6">
                    <div className="flex items-center justify-between">
                      <h3 className="text-lg font-semibold text-gray-900">
                        Account Deletion Request
                      </h3>
                      <button
                        onClick={() => setShowDeleteModal(false)}
                        className="p-2 text-gray-400 hover:text-gray-600"
                      >
                        <X className="w-5 h-5" />
                      </button>
                    </div>

                    <p className="text-sm text-gray-600">
                      We're sorry to see you go! Please help us understand your
                      reason for deleting your account:
                    </p>

                    <div className="space-y-4">
                      <p className="text-sm font-medium text-gray-700">
                        Select a reason:
                      </p>

                      <div className="space-y-3">
                        {[
                          {
                            value: "no-longer-need",
                            label: "I no longer need the service",
                          },
                          {
                            value: "privacy-concerns",
                            label: "I have privacy concerns",
                          },
                          {
                            value: "better-alternative",
                            label: "I found a better alternative",
                          },
                          {
                            value: "features-not-met",
                            label: "The features did not meet my expectations",
                          },
                        ].map((reason) => (
                          <label
                            key={reason.value}
                            className="flex items-center space-x-3 cursor-pointer"
                          >
                            <input
                              type="radio"
                              name="deleteReason"
                              value={reason.value}
                              checked={deleteReason === reason.value}
                              onChange={(e) => setDeleteReason(e.target.value)}
                              className="w-4 h-4 text-[#0F2FA3] border-gray-300 focus:ring-[#0F2FA3]"
                            />
                            <span className="text-sm text-gray-700">
                              {reason.label}
                            </span>
                          </label>
                        ))}

                        <label className="flex items-start space-x-3 cursor-pointer">
                          <input
                            type="radio"
                            name="deleteReason"
                            value="other"
                            checked={deleteReason === "other"}
                            onChange={(e) => setDeleteReason(e.target.value)}
                            className="w-4 h-4 text-[#0F2FA3] border-gray-300 focus:ring-[#0F2FA3] mt-0.5"
                          />
                          <div className="flex-1">
                            <span className="text-sm text-gray-700">
                              Other:
                            </span>
                            {deleteReason === "other" && (
                              <input
                                type="text"
                                value={otherReason}
                                onChange={(e) => setOtherReason(e.target.value)}
                                placeholder="Please specify..."
                                className="w-full mt-2 px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-[#0F2FA3] focus:border-transparent text-sm"
                              />
                            )}
                          </div>
                        </label>
                      </div>
                    </div>

                    <div className="bg-blue-50 border border-blue-200 rounded-lg p-3">
                      <div className="flex items-start space-x-2">
                        <svg
                          className="w-4 h-4 text-blue-600 mt-0.5 flex-shrink-0"
                          fill="currentColor"
                          viewBox="0 0 20 20"
                        >
                          <path
                            fillRule="evenodd"
                            d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
                            clipRule="evenodd"
                          />
                        </svg>
                        <p className="text-xs text-blue-800">
                          By submitting this form, your account and associated
                          data will be permanently deleted in accordance with
                          our Privacy Policy.
                        </p>
                      </div>
                    </div>

                    <div className="flex space-x-3 pt-4">
                      <button
                        onClick={() => setShowDeleteModal(false)}
                        className="flex-1 px-4 py-2 border border-gray-300 rounded-lg text-sm font-medium text-gray-700 hover:bg-gray-50 transition-colors"
                      >
                        Cancel
                      </button>
                      <button
                        // onClick={handleDeleteModalSubmit}
                        // disabled={isDeletingAccount}
                        className="flex-1 px-4 py-2 bg-red-600 border border-transparent rounded-lg text-sm font-medium text-white hover:bg-red-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors flex items-center justify-center"
                      >
                        {isDeletingAccount ? (
                          <>
                            <Loader className="animate-spin -ml-1 mr-2 h-4 w-4" />
                            Deleting...
                          </>
                        ) : (
                          "Delete Account"
                        )}
                      </button>
                    </div>
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

export default Settings;
