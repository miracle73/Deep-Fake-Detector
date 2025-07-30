import { useState } from "react";
import {
  Bell,
  LayoutGrid,
  Video,
  ImageIcon,
  AudioLines,
  Menu,
  X,
} from "lucide-react";
import { NoAnalysisYet } from "../assets/svg";
import { useNavigate } from "react-router-dom";
import { CiSettings } from "react-icons/ci";
import SafeguardMediaLogo from "../assets/images/SafeguardMedia8.svg";
import { useSelector } from "react-redux";
import type { RootState } from "../store/store";
import {
  useGetNotificationsQuery,
  useMarkNotificationReadMutation,
} from "../services/apiService";

const Notifications = () => {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const navigate = useNavigate();
  const [selectedNotification, setSelectedNotification] =
    useState<Notification | null>(null);
  const storedUser = useSelector((state: RootState) => state.user.user);
  // Toggle  to show/hide notifications for testing
  const {
    data: notificationsData,
    isLoading,
    error,
  } = useGetNotificationsQuery();
  const [markNotificationRead] = useMarkNotificationReadMutation();
  const notifications = notificationsData?.notifications || [];
  const hasNotifications = notifications.length > 0;

  const handleGoBack = () => {
    navigate("/dashboard");
  };

  interface Notification {
    _id: string;
    userId: string;
    type: string;
    title: string;
    message: string;
    read: boolean;
    expiresAt: string;
    createdAt: string;
    updatedAt: string;
    __v: number;
    id: string;
  }

  interface NotificationClickHandler {
    (notification: Notification): Promise<void>;
  }

  const handleNotificationClick: NotificationClickHandler = async (
    notification
  ) => {
    setSelectedNotification(notification);

    // Mark notification as read if it's unread
    if (!notification.read) {
      try {
        await markNotificationRead(notification._id).unwrap();
        // Updating the local state to reflect the read status immediately
        // This will prevent the notification from appearing unread until page refresh
      } catch (error) {
        console.error("Failed to mark notification as read:", error);
      }
    }
  };

  const handleCloseDetails = () => {
    setSelectedNotification(null);
  };

  const formatTimestamp = (dateString: string) => {
    const date = new Date(dateString);
    const now = new Date();
    const diffInDays = Math.floor(
      (now.getTime() - date.getTime()) / (1000 * 60 * 60 * 24)
    );

    if (diffInDays === 0) return "Today";
    if (diffInDays === 1) return "1 day ago";
    if (diffInDays < 7) return `${diffInDays} days ago`;
    if (diffInDays < 30) return `${Math.floor(diffInDays / 7)} weeks ago`;
    return `${Math.floor(diffInDays / 30)} months ago`;
  };

  const groupedNotifications = notifications.reduce((acc, notification) => {
    const category = notification.read ? "Read" : "Unread";
    if (!acc[category]) {
      acc[category] = [];
    }
    acc[category].push(notification);
    return acc;
  }, {} as Record<string, Notification[]>);

  const sortedCategories = Object.entries(groupedNotifications).sort(
    ([a], [b]) => {
      if (a === "Unread" && b === "Read") return -1;
      if (a === "Read" && b === "Unread") return 1;
      return 0;
    }
  );

  if (isLoading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <p className="text-gray-600">Loading notifications...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <p className="text-red-600 mb-4">Failed to load notifications</p>
          <button
            onClick={() => window.location.reload()}
            className="bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

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
              className="hidden sm:flex bg-[#FBFBEF] gap-2 justify-between items-center"
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
          {/* Main Content Area - Full Width */}
          <div className="w-full p-4 sm:p-6">
            {/* Header Section */}
            <div className="mb-8">
              <div className="flex items-center justify-between">
                <div>
                  <h2 className="text-xl sm:text-2xl font-semibold text-gray-900 mb-2">
                    Notifications
                    {notifications.filter((n) => !n.read).length > 0 && (
                      <span className="ml-3 inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
                        {notifications.filter((n) => !n.read).length} unread
                      </span>
                    )}
                  </h2>
                  <p className="text-sm sm:text-base text-gray-600 mb-6">
                    Recent update: 9th May, 2025; 10:34am
                  </p>
                </div>
              </div>
            </div>

            {/* Conditional Content */}
            {hasNotifications ? (
              /* Notifications List */
              <div className="w-full bg-white rounded-xl border border-[#8C8C8C] overflow-hidden">
                {sortedCategories.map(([category, categoryNotifications]) => (
                  <div key={category}>
                    {/* Category Header */}
                    <div
                      className={`px-6 py-4 border-b border-gray-200 ${
                        category === "Unread" ? "bg-blue-50" : "bg-gray-50"
                      }`}
                    >
                      <h3
                        className={`text-sm font-medium ${
                          category === "Unread"
                            ? "text-blue-700"
                            : "text-gray-700"
                        }`}
                      >
                        {category} ({categoryNotifications.length})
                      </h3>
                    </div>

                    {/* Notifications in this category */}
                    <div className="divide-y divide-gray-200">
                      {categoryNotifications.map((notification) => (
                        <div
                          key={notification._id}
                          className={`px-6 py-4 hover:bg-gray-50 transition-colors cursor-pointer relative ${
                            !notification.read
                              ? "bg-blue-50 border-l-4 border-l-blue-500 shadow-sm"
                              : "bg-white opacity-75"
                          }`}
                          onClick={() => handleNotificationClick(notification)}
                        >
                          <div className="flex items-start space-x-4">
                            {/* Bell Icon with read/unread styling */}
                            <div className="flex-shrink-0 mt-1">
                              <div
                                className={`w-8 h-8 rounded-full flex items-center justify-center ${
                                  !notification.read
                                    ? "bg-blue-100 ring-2 ring-blue-500 ring-opacity-30"
                                    : "bg-gray-100"
                                }`}
                              >
                                <Bell
                                  className={`w-4 h-4 ${
                                    !notification.read
                                      ? "text-blue-600"
                                      : "text-gray-400"
                                  }`}
                                />
                              </div>
                            </div>

                            {/* Notification Content */}
                            <div className="flex-1 min-w-0">
                              <div className="flex items-start justify-between">
                                <div className="flex items-center space-x-2">
                                  <h4
                                    className={`text-sm mb-1 ${
                                      !notification.read
                                        ? "text-gray-900 font-semibold"
                                        : "text-gray-600 font-normal"
                                    }`}
                                  >
                                    {notification.title}
                                  </h4>
                                  {!notification.read && (
                                    <div className="w-3 h-3 bg-blue-500 rounded-full animate-pulse"></div>
                                  )}
                                </div>
                                <span
                                  className={`text-xs flex-shrink-0 ml-4 ${
                                    !notification.read
                                      ? "text-gray-600"
                                      : "text-gray-400"
                                  }`}
                                >
                                  {formatTimestamp(notification.createdAt)}
                                </span>
                              </div>
                              <p
                                className={`text-sm leading-relaxed ${
                                  !notification.read
                                    ? "text-gray-700"
                                    : "text-gray-500"
                                }`}
                              >
                                {notification.message}
                              </p>
                            </div>
                          </div>

                          {/* Unread indicator line */}
                          {!notification.read && (
                            <div className="absolute left-0 top-0 bottom-0 w-1 bg-blue-500"></div>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                ))}
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
              <div className="flex items-center space-x-3">
                <h4 className="text-lg font-medium text-gray-900">
                  {selectedNotification.title}
                </h4>
                {!selectedNotification.read && (
                  <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
                    Unread
                  </span>
                )}
              </div>
              <button
                onClick={handleCloseDetails}
                className="p-2 text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded-full"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            {/* Content */}
            <div className="p-6 space-y-6">
              {/* Metadata */}
              <div className="flex items-center space-x-4 text-sm text-gray-500">
                <span>Type: {selectedNotification.type}</span>
                <span>•</span>
                <span>{formatTimestamp(selectedNotification.createdAt)}</span>
              </div>

              {/* Full message */}
              <div>
                <h5 className="text-sm font-medium text-gray-900 mb-2">
                  Message
                </h5>
                <p className="text-sm text-gray-600 leading-relaxed">
                  {selectedNotification.message}
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

export default Notifications;
