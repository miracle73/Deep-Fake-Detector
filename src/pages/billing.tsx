import { useState } from "react";
import {
  Bell,
  ChevronDown,
  LayoutGrid,
  Video,
  ImageIcon,
  AudioLines,
  Menu,
  X,
  Check,
} from "lucide-react";
import { useNavigate } from "react-router-dom";
import { useGetUserQuery } from "../services/apiService";

const Billing = () => {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const navigate = useNavigate();
  const currentPlanFeatures = [
    "Analyze media for up to 4,000 seconds each month.",
    "Basic customer support",
    "Basic customer support",
  ];
  const { data: userData } = useGetUserQuery();
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

            {/* <div className="flex items-center space-x-2 cursor-pointer rounded-[30px]">
              <div className="w-6 h-6 sm:w-8 sm:h-8 bg-gray-300 rounded-full flex items-center justify-center">
                <span className="text-xs sm:text-sm font-medium text-gray-600">
                  U
                </span>
              </div>
              <span className="hidden sm:inline text-sm text-gray-700">
                Username
              </span>
              <ChevronDown className="w-3 h-3 sm:w-4 sm:h-4 text-gray-400" />
            </div> */}
            <div className="flex items-center space-x-2 cursor-pointer rounded-[30px]">
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
          {/* Main Content Area - Full Width */}
          <div className="w-full p-4 sm:p-6">
            {/* Getting Started Section */}
            <div className="mb-8">
              <h2 className="text-xl sm:text-2xl font-semibold text-gray-900 mb-2">
                Billing & Subscription
              </h2>
              <p className="text-sm sm:text-base text-gray-600 mb-6">
                Current Subscription Plan: <b>Free</b>
              </p>
            </div>

            {/* Current Plan Features */}
            <div className="w-full bg-white rounded-xl border border-[#8C8C8C] p-6 sm:p-8">
              <div className="space-y-6">
                <h3 className="text-base sm:text-lg font-semibold text-gray-900">
                  Current Plan Features:
                </h3>

                <div className="space-y-4">
                  {currentPlanFeatures.map((feature, index) => (
                    <div key={index} className="flex items-start space-x-3">
                      <Check className="w-4 h-4 sm:w-5 sm:h-5 text-green-600 mt-0.5 flex-shrink-0" />
                      <span className="text-sm sm:text-base text-gray-700 leading-relaxed">
                        {feature}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Billing;
