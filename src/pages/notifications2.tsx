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
} from "lucide-react";
import { NoAnalysisYet } from "../assets/svg";
import { useNavigate } from "react-router-dom";

// Mock notifications data - replace with your actual data source
const mockNotifications = [
  {
    id: 1,
    title: "Notification Header Text",
    content:
      "Notification content text for instance: you haven't updated your record for a while now. Perform activities using our model to stay alert in today's world of deepfake contents.",
    timestamp: "5 days ago",
    category: "Earlier",
  },
  {
    id: 2,
    title: "Security Alert",
    content:
      "Your account security settings have been updated. Please review the changes and ensure they are correct.",
    timestamp: "3 days ago",
    category: "Earlier",
  },
  {
    id: 3,
    title: "System Update",
    content:
      "New features have been added to your dashboard. Check out the latest improvements to enhance your experience.",
    timestamp: "1 day ago",
    category: "Earlier",
  },
];

const Notifications2 = () => {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const navigate = useNavigate();

  const [selectedNotification, setSelectedNotification] =
    useState<Notification | null>(null);
  // Toggle this to show/hide notifications for testing
  const [hasNotifications] = useState(true);

  const handleGoBack = () => {
    navigate("/dashboard");
  };

  interface Notification {
    id: number;
    title: string;
    content: string;
    timestamp: string;
    category: string;
  }

  interface NotificationClickHandler {
    (notification: Notification): void;
  }

  const handleNotificationClick: NotificationClickHandler = (notification) => {
    setSelectedNotification(notification);
  };

  const handleCloseDetails = () => {
    setSelectedNotification(null);
  };

  // Group notifications by category
  const groupedNotifications = mockNotifications.reduce((acc, notification) => {
    if (!acc[notification.category]) {
      acc[notification.category] = [];
    }
    acc[notification.category].push(notification);
    return acc;
  }, {} as Record<string, typeof mockNotifications>);

  return (
    <div className={`min-h-screen bg-gray-50`}>
      {selectedNotification && (
        <div className="fixed inset-0 bg-gray-50 bg-opacity-30 backdrop-blur-sm z-40"></div>
      )}
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
              className="hidden sm:flex bg-[#FBFBEF] gap-2 justify-between items-center"
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
              <div className="flex items-center space-x-3 text-gray-400 cursor-pointer">
                <Video className="w-6 h-6" />
                <span className="text-sm">Video</span>
              </div>
              <div className="flex items-center space-x-3 text-gray-400 cursor-pointer">
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
          <div className="flex flex-col items-center space-y-2 text-gray-400 cursor-pointer">
            <AudioLines className="w-6 h-6" />
            <span className="text-xs">Audio</span>
          </div>
          <div className="flex flex-col items-center space-y-2 text-gray-400 cursor-pointer">
            <Video className="w-6 h-6" />
            <span className="text-xs">Video</span>
          </div>
          <div className="flex flex-col items-center space-y-2 text-gray-400 cursor-pointer">
            <ImageIcon className="w-6 h-6" />
            <span className="text-xs">Image</span>
          </div>
        </div>

        {/* Main Content Container */}
        <div className="flex-1 flex flex-col">
          {/* Main Content Area - Full Width */}
          <div className="w-full p-4 sm:p-6">
            {/* Header Section */}
            <div className="mb-8">
              <h2 className="text-xl sm:text-2xl font-semibold text-gray-900 mb-2">
                Notifications
              </h2>
              <p className="text-sm sm:text-base text-gray-600 mb-6">
                Recent update: 9th May, 2025; 10:34am
              </p>
            </div>

            {/* Conditional Content */}
            {hasNotifications && mockNotifications.length > 0 ? (
              /* Notifications List */
              <div className="w-full bg-white rounded-xl border border-[#8C8C8C] overflow-hidden">
                {Object.entries(groupedNotifications).map(
                  ([category, notifications]) => (
                    <div key={category}>
                      {/* Category Header */}
                      <div className="px-6 py-4 bg-gray-50 border-b border-gray-200">
                        <h3 className="text-sm font-medium text-gray-700">
                          {category}
                        </h3>
                      </div>

                      {/* Notifications in this category */}
                      <div className="divide-y divide-gray-200">
                        {notifications.map((notification) => (
                          <div
                            key={notification.id}
                            className="px-6 py-4 hover:bg-gray-50 transition-colors cursor-pointer"
                            onClick={() =>
                              handleNotificationClick(notification)
                            }
                          >
                            <div className="flex items-start space-x-4">
                              {/* Bell Icon */}
                              <div className="flex-shrink-0 mt-1">
                                <div className="w-8 h-8 bg-gray-100 rounded-full flex items-center justify-center">
                                  <Bell className="w-4 h-4 text-gray-600" />
                                </div>
                              </div>

                              {/* Notification Content */}
                              <div className="flex-1 min-w-0">
                                <div className="flex items-start justify-between">
                                  <h4 className="text-sm font-medium text-gray-900 mb-1">
                                    {notification.title}
                                  </h4>
                                  <span className="text-xs text-gray-500 flex-shrink-0 ml-4">
                                    {notification.timestamp}
                                  </span>
                                </div>
                                <p className="text-sm text-gray-600 leading-relaxed">
                                  {notification.content}
                                </p>
                              </div>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )
                )}
              </div>
            ) : (
              /* Empty State */
              <div className="w-full bg-white rounded-xl border border-[#8C8C8C] p-12 sm:p-16 py-16 sm:py-24 text-center">
                <div className="mb-6">
                  <div className="relative inline-block">
                    <NoAnalysisYet />
                  </div>
                </div>
                <p className="text-gray-600 text-sm sm:text-base mb-8">
                  Whoops, looks like there is no notification.
                </p>
                <button
                  onClick={handleGoBack}
                  className="bg-[#FBFBEF] hover:bg-gray-200 border border-[#8C8C8C] text-gray-700 px-8 py-2 rounded-full text-sm font-medium transition-colors"
                >
                  Go Back
                </button>
              </div>
            )}
          </div>
        </div>
      </div>
      {/* Notification Details Modal */}
      {selectedNotification && (
        <div className="fixed inset-0 backdrop-blur-sm bg-opacity-25 flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-xl max-w-2xl w-full max-h-[80vh] overflow-y-auto">
            {/* Header */}
            <div className="flex items-center justify-between p-6 border-b border-gray-200">
              <h4 className="text-lg font-medium text-gray-900 mb-2">
                {selectedNotification.title}
              </h4>
              <button
                onClick={handleCloseDetails}
                className="p-2 text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded-full"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            {/* Content */}
            <div className="p-6 space-y-6">
              {/* Title and timestamp */}
              <div>
                <p className="text-sm text-gray-500">
                  {selectedNotification.timestamp}
                </p>
              </div>

              {/* Full message */}
              <div>
                <h5 className="text-sm font-medium text-gray-900 mb-2">
                  Message
                </h5>
                <p className="text-sm text-gray-600 leading-relaxed">
                  {selectedNotification.content}
                </p>
              </div>

              {/* Action buttons */}
              <div className="flex flex-col sm:flex-row gap-3">
                <button
                  onClick={handleCloseDetails}
                  className="bg-gray-100 hover:bg-gray-200 text-gray-700 px-6 py-2 rounded-full text-sm font-medium"
                >
                  Close
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default Notifications2;
