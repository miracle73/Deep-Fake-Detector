import { useState } from "react";
import {
  Bell,
  ChevronDown,
  LayoutGrid,
  Video,
  ImageIcon,
  Clock,
  FileText,
  HelpCircle,
  AudioLines,
  Menu,
  X,
} from "lucide-react";

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

  const handlePersonalInfoChange = (field: string, value: string) => {
    setPersonalInfo((prev) => ({
      ...prev,
      [field]: value,
    }));
  };

  const handlePersonalInfoSave = () => {
    console.log("Saving personal info:", personalInfo);
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
            <h1 className="text-lg sm:text-xl font-bold text-gray-900">
              <span className="font-bold">Df</span>{" "}
              <span className="font-normal">Detector</span>
            </h1>
          </div>
          <div className="flex items-center space-x-2 sm:space-x-4">
            <div className="hidden sm:flex border border-[#8C8C8C] bg-[#FBFBEF] rounded-[30px] pl-2 gap-2 justify-between items-center">
              <span className="text-sm text-gray-600">4,000 sec left</span>
              <button className="bg-[#0F2FA3] hover:bg-blue-700 text-white px-4 py-2 rounded-[30px] text-sm font-medium">
                Upgrade
              </button>
            </div>

            {/* Mobile upgrade button */}
            <button className="sm:hidden bg-[#0F2FA3] hover:bg-blue-700 text-white px-3 py-1.5 rounded-[20px] text-xs font-medium">
              Upgrade
            </button>

            <button className="p-2 text-gray-400 hover:text-gray-600 bg-[#F6F7FE] rounded-[30px] border-[0.88px] border-[#8C8C8C]">
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
              <div className="flex items-center space-x-3 text-gray-600 hover:text-blue-600 cursor-pointer">
                <LayoutGrid className="w-6 h-6" />
                <span className="text-sm">Dashboard</span>
              </div>
              <div className="flex items-center space-x-3 text-gray-400 hover:text-blue-600 cursor-pointer">
                <AudioLines className="w-6 h-6" />
                <span className="text-sm">Audio</span>
              </div>
              <div className="flex items-center space-x-3 text-gray-400 hover:text-blue-600 cursor-pointer">
                <Video className="w-6 h-6" />
                <span className="text-sm">Video</span>
              </div>
              <div className="flex items-center space-x-3 text-gray-400 hover:text-blue-600 cursor-pointer">
                <ImageIcon className="w-6 h-6" />
                <span className="text-sm">Image</span>
              </div>
              <div className="flex items-center space-x-3 text-gray-400 hover:text-blue-600 cursor-pointer">
                <Clock className="w-6 h-6" />
                <span className="text-sm">History</span>
              </div>
              <div className="border-t border-gray-200 pt-6 space-y-6">
                <div className="flex items-center space-x-3 text-gray-400 hover:text-blue-600 cursor-pointer">
                  <FileText className="w-6 h-6" />
                  <span className="text-sm">Documentation</span>
                </div>
                <div className="flex items-center space-x-3 text-gray-400 hover:text-blue-600 cursor-pointer">
                  <HelpCircle className="w-6 h-6" />
                  <span className="text-sm">Help</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Content Area with Sidebar */}
      <div className="flex">
        {/* Desktop Sidebar */}
        <div className="hidden lg:flex w-24 bg-white border-r border-gray-200 flex-col items-center py-6 space-y-8 min-h-[calc(100vh-73px)]">
          <div className="flex flex-col items-center space-y-2 text-gray-600 hover:text-blue-600 cursor-pointer">
            <LayoutGrid className="w-6 h-6" />
            <span className="text-xs">Dashboard</span>
          </div>
          <div className="flex flex-col items-center space-y-2 text-gray-400 hover:text-blue-600 cursor-pointer">
            <AudioLines className="w-6 h-6" />
            <span className="text-xs">Audio</span>
          </div>
          <div className="flex flex-col items-center space-y-2 text-gray-400 hover:text-blue-600 cursor-pointer">
            <Video className="w-6 h-6" />
            <span className="text-xs">Video</span>
          </div>
          <div className="flex flex-col items-center space-y-2 text-gray-400 hover:text-blue-600 cursor-pointer">
            <ImageIcon className="w-6 h-6" />
            <span className="text-xs">Image</span>
          </div>
          <div className="flex flex-col items-center space-y-2 text-gray-400 pb-20 hover:text-blue-600 cursor-pointer">
            <Clock className="w-6 h-6" />
            <span className="text-xs">History</span>
          </div>
          <div className="mt-auto space-y-8 border-t border-[#8C8C8C] pt-8 pb-12">
            <div className="flex flex-col items-center space-y-2 text-gray-400 hover:text-blue-600 cursor-pointer">
              <FileText className="w-6 h-6" />
              <span className="text-xs">Documentation</span>
            </div>
            <div className="flex flex-col items-center space-y-2 text-gray-400 hover:text-blue-600 cursor-pointer">
              <HelpCircle className="w-6 h-6" />
              <span className="text-xs">Help</span>
            </div>
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
                Welcome back, Username
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

                  {/* Save Changes Button */}
                  <div className="flex justify-start">
                    <button
                      onClick={handlePersonalInfoSave}
                      className="bg-[#FBFBEF] border border-[#8C8C8C] hover:bg-gray-200 text-gray-700 px-10 py-2 rounded-full text-sm font-medium transition-colors"
                    >
                      Save Changes
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
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-8  pt-8">
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
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Settings;
