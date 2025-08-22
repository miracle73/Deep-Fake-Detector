import { useState, useEffect } from "react";
import {
  Search,
  Star,
  MessageSquarePlus,
  ChevronDown,
  ChevronUp,
  Calendar,
  Mail,
  User,
  AlertCircle,
  ArrowLeft,
  X,
} from "lucide-react";
import {
  useGetFeedbacksQuery,
  useGetFeedbackStatsQuery,
  useUpdateFeedbackMutation,
} from "../services/apiService";

type Feedback = {
  id: string;
  _id: string;
  userId: string;
  userName: string;
  email: string;
  type: string;
  rating: number;
  message: string;
  description: string;
  status: string;
  priority: string;
  timestamp: string;
  createdAt: string;
  updatedAt: string;
  __v: number;
};

// type FeedbackType = "general" | "bug" | "feature" | "improvement";
// type FeedbackStatus = "pending" | "in_progress" | "reviewed" | "resolved";
// type FeedbackPriority = "critical" | "high" | "medium" | "low";

const AdminFeedbackDashboard = () => {
  const [feedbacks, setFeedbacks] = useState<Feedback[]>([]);
  const [filteredFeedbacks, setFilteredFeedbacks] = useState<Feedback[]>([]);
  const [selectedFilter, setSelectedFilter] = useState("all");
  const [searchTerm, setSearchTerm] = useState("");
  const [expandedFeedback, setExpandedFeedback] = useState<string | null>(null);
  const [selectedStatus, setSelectedStatus] = useState("all");
  const [sortBy, setSortBy] = useState("newest");
  const {
    data: feedbacksData,
    isLoading,
    refetch,
  } = useGetFeedbacksQuery({ page: 1, limit: 100 });
  const { data: statsData } = useGetFeedbackStatsQuery();
  const [updateFeedback] = useUpdateFeedbackMutation();
  const [updateSuccessMessage, setUpdateSuccessMessage] = useState<string>("");
  const [updateErrorMessage, setUpdateErrorMessage] = useState<string>("");

  useEffect(() => {
    if (updateErrorMessage) {
      const timer = setTimeout(() => {
        setUpdateErrorMessage("");
      }, 3000);
      return () => clearTimeout(timer);
    }
  }, [updateErrorMessage]);

  useEffect(() => {
    if (updateSuccessMessage) {
      const timer = setTimeout(() => {
        setUpdateSuccessMessage("");
      }, 3000);
      return () => clearTimeout(timer);
    }
  }, [updateSuccessMessage]);

  useEffect(() => {
    if (feedbacksData?.data) {
      const transformedFeedbacks: Feedback[] = feedbacksData.data.map(
        (feedback) => ({
          ...feedback, // Keep all original properties
          id: feedback._id,
          userId: feedback._id, // Using _id as userId since it's not provided
          userName: feedback.email ? feedback.email.split("@")[0] : "Anonymous",
          email: feedback.email || "",
          type: feedback.type.toLowerCase(),
          rating: feedback.rating,
          message: feedback.description,
          timestamp: feedback.createdAt,
          status: feedback.status.replace(" ", "_"),
          priority:
            feedback.rating <= 2
              ? "high"
              : feedback.rating === 3
              ? "medium"
              : "low",
        })
      );
      setFeedbacks(transformedFeedbacks);
      setFilteredFeedbacks(transformedFeedbacks);
    }
  }, [feedbacksData]);
  // Filter and search logic
  useEffect(() => {
    let filtered = [...feedbacks];

    // Filter by type
    if (selectedFilter !== "all") {
      filtered = filtered.filter((f) => f.type === selectedFilter);
    }

    // Filter by status
    if (selectedStatus !== "all") {
      filtered = filtered.filter((f) => f.status === selectedStatus);
    }

    // Search
    if (searchTerm) {
      filtered = filtered.filter(
        (f) =>
          f.message.toLowerCase().includes(searchTerm.toLowerCase()) ||
          f.userName.toLowerCase().includes(searchTerm.toLowerCase()) ||
          (f.email && f.email.toLowerCase().includes(searchTerm.toLowerCase()))
      );
    }

    // Sort - Fixed Date arithmetic by converting to getTime()
    filtered.sort((a, b) => {
      if (sortBy === "newest") {
        return (
          new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()
        );
      } else if (sortBy === "oldest") {
        return (
          new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime()
        );
      } else if (sortBy === "rating-high") {
        return b.rating - a.rating;
      } else if (sortBy === "rating-low") {
        return a.rating - b.rating;
      }
      return 0;
    });

    setFilteredFeedbacks(filtered);
  }, [feedbacks, selectedFilter, selectedStatus, searchTerm, sortBy]);

  const getStatusColor = (status: string): string => {
    switch (status) {
      case "pending":
        return "bg-yellow-100 text-yellow-800";
      case "in_progress":
        return "bg-blue-100 text-blue-800";
      case "reviewed":
        return "bg-purple-100 text-purple-800";
      case "resolved":
        return "bg-green-100 text-green-800";
      default:
        return "bg-gray-100 text-gray-800";
    }
  };

  const getPriorityColor = (priority: string): string => {
    switch (priority) {
      case "critical":
        return "bg-red-500";
      case "high":
        return "bg-orange-500";
      case "medium":
        return "bg-yellow-500";
      case "low":
        return "bg-green-500";
      default:
        return "bg-gray-500";
    }
  };

  const stats = {
    total: statsData?.data?.total || 0,
    pending: statsData?.data?.pending || 0,
    avgRating: statsData?.data?.averageRating?.toFixed(1) || "0",
    critical:
      (statsData?.data?.pending || 0) + (statsData?.data?.inProgress || 0),
  };

  const updateStatus = async (
    feedbackId: string,
    newStatus: string
  ): Promise<void> => {
    try {
      const apiStatus = newStatus.replace("_", " ");
      await updateFeedback({
        id: feedbackId,
        status: apiStatus,
      }).unwrap();

      // Update local state
      setFeedbacks((prev) =>
        prev.map((f) => (f.id === feedbackId ? { ...f, status: newStatus } : f))
      );

      // Refetch to get updated data
      refetch();

      // Show success message
      setUpdateSuccessMessage(
        `Feedback status updated to ${apiStatus} successfully!`
      );
    } catch (error) {
      console.error("Failed to update feedback status:", error);

      if (error && typeof error === "object" && "data" in error) {
        const apiError = error as {
          data?: { message?: string };
        };
        if (apiError.data?.message) {
          setUpdateErrorMessage(apiError.data.message);
        } else {
          setUpdateErrorMessage(
            "Failed to update feedback status. Please try again."
          );
        }
      } else if (error && typeof error === "object" && "message" in error) {
        const messageError = error as { message: string };
        setUpdateErrorMessage(messageError.message);
      } else {
        setUpdateErrorMessage(
          "Failed to update feedback status. Please try again."
        );
      }
    }
  };
  if (isLoading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-gray-50 via-white to-gray-50 flex items-center justify-center">
        <div className="text-center">
          <div className="w-8 h-8 border-4 border-[#0F2FA3] border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
          <p className="text-gray-600">Loading feedback...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 via-white to-gray-50">
      {/* Header */}
      <header className="bg-white border-b border-gray-200 sticky top-0 z-10 shadow-sm">
        <div className="px-4 sm:px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2 sm:space-x-3 min-w-0 flex-1">
              <button
                onClick={() => window.history.back()}
                className="p-2 hover:bg-gray-100 rounded-lg transition-colors flex-shrink-0"
              >
                <ArrowLeft className="w-4 h-4 sm:w-5 sm:h-5 text-gray-600" />
              </button>
              <div className="min-w-0 flex-1">
                <h1 className="text-lg sm:text-2xl font-bold text-gray-900 truncate">
                  Feedback Dashboard
                </h1>
                <p className="text-xs sm:text-sm text-gray-500 truncate">
                  Manage and respond to user feedback
                </p>
              </div>
            </div>
            <div className="flex items-center ml-2 flex-shrink-0">
              <span className="px-2 sm:px-3 py-1 bg-[#0F2FA3]/10 text-center border border-[#0F2FA3]/20 bg-opacity-10 text-[#0F2FA3] rounded-full text-xs sm:text-sm font-medium">
                Admin Panel
              </span>
            </div>
          </div>
        </div>
      </header>

      {/* Success Message */}
      {updateSuccessMessage && (
        <div className="px-4 sm:px-6 pt-4">
          <div className="flex items-center p-3 text-sm text-green-600 bg-green-50 border border-green-200 rounded-lg">
            <div className="w-4 h-4 mr-2 flex-shrink-0">
              <svg viewBox="0 0 20 20" fill="currentColor" className="w-4 h-4">
                <path
                  fillRule="evenodd"
                  d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
                  clipRule="evenodd"
                />
              </svg>
            </div>
            <span>{updateSuccessMessage}</span>
            <button
              type="button"
              onClick={() => setUpdateSuccessMessage("")}
              className="ml-auto text-green-400 hover:text-green-600"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        </div>
      )}

      {/* Error Message */}
      {updateErrorMessage && (
        <div className="px-4 sm:px-6 pt-4">
          <div className="flex items-center p-3 text-sm text-red-600 bg-red-50 border border-red-200 rounded-lg">
            <AlertCircle className="w-4 h-4 mr-2 flex-shrink-0" />
            <span>{updateErrorMessage}</span>
            <button
              type="button"
              onClick={() => setUpdateErrorMessage("")}
              className="ml-auto text-red-400 hover:text-red-600"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        </div>
      )}

      {/* Stats Cards */}
      <div className="px-4 sm:px-6 py-6">
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 transform transition-all duration-300 hover:shadow-lg hover:scale-105">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-500">Total Feedback</p>
                <p className="text-2xl font-bold text-gray-900 mt-1">
                  {stats.total}
                </p>
              </div>
            </div>
          </div>

          <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 transform transition-all duration-300 hover:shadow-lg hover:scale-105">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-500">Pending Review</p>
                <p className="text-2xl font-bold text-yellow-600 mt-1">
                  {stats.pending}
                </p>
              </div>
            </div>
          </div>

          <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 transform transition-all duration-300 hover:shadow-lg hover:scale-105">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-500">Avg Rating</p>
                <p className="text-2xl font-bold text-gray-900 mt-1">
                  {stats.avgRating}
                </p>
              </div>
            </div>
          </div>

          <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 transform transition-all duration-300 hover:shadow-lg hover:scale-105">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-500">High Priority</p>
                <p className="text-2xl font-bold text-red-600 mt-1">
                  {stats.critical}
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Filters and Search */}
      <div className="px-4 sm:px-6 pb-4">
        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-4">
          <div className="flex flex-col lg:flex-row gap-4">
            {/* Search */}

            <div className="flex-1">
              <div className="relative">
                <Search className="absolute left-3 top-1/4 transform -translate-y-1/2 text-gray-400 w-5 h-5" />
                <input
                  type="text"
                  placeholder="Search feedback..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-[#0F2FA3] focus:border-transparent"
                />
              </div>
            </div>

            {/* Filter by Type */}
            <select
              value={selectedFilter}
              onChange={(e) => setSelectedFilter(e.target.value)}
              className="px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-[#0F2FA3] focus:border-transparent"
            >
              <option value="all">All Types</option>
              <option value="general">General</option>
              <option value="bug">Bug Report</option>
              <option value="feature">Feature Request</option>
              <option value="improvement">Improvement</option>
            </select>

            {/* Filter by Status */}
            <select
              value={selectedStatus}
              onChange={(e) => setSelectedStatus(e.target.value)}
              className="px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-[#0F2FA3] focus:border-transparent"
            >
              <option value="all">All Status</option>
              <option value="pending">Pending</option>
              <option value="in_progress">In Progress</option>
              <option value="reviewed">Reviewed</option>
              <option value="resolved">Resolved</option>
            </select>

            {/* Sort */}
            <select
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value)}
              className="px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-[#0F2FA3] focus:border-transparent"
            >
              <option value="newest">Newest First</option>
              <option value="oldest">Oldest First</option>
              <option value="rating-high">Highest Rating</option>
              <option value="rating-low">Lowest Rating</option>
            </select>
          </div>
        </div>
      </div>

      {/* Feedback List */}
      <div className="px-4 sm:px-6 pb-6">
        <div className="space-y-4">
          {filteredFeedbacks.map((feedback) => {
            const isExpanded = expandedFeedback === feedback.id;

            return (
              <div
                key={feedback.id}
                className="bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden transform transition-all duration-300 hover:shadow-lg"
              >
                <div className="p-6">
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      {/* Header */}
                      <div className="flex items-center space-x-4 mb-3">
                        <div
                          className={`w-2 h-8 rounded-full ${getPriorityColor(
                            feedback.priority
                          )}`}
                        />

                        <span className="font-semibold text-gray-900">
                          {feedback.userName}
                        </span>
                        <span
                          className={`px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(
                            feedback.status
                          )}`}
                        >
                          {feedback.status.replace("_", " ")}
                        </span>
                        <div className="flex items-center">
                          {[...Array(5)].map((_, i) => (
                            <Star
                              key={i}
                              className={`w-4 h-4 ${
                                i < feedback.rating
                                  ? "fill-yellow-400 text-yellow-400"
                                  : "text-gray-300"
                              }`}
                            />
                          ))}
                        </div>
                      </div>

                      {/* Message Preview */}
                      <p className="text-gray-700 mb-3">
                        {isExpanded
                          ? feedback.message
                          : `${feedback.message.substring(0, 100)}...`}
                      </p>

                      {/* Meta Info */}
                      <div className="flex items-center space-x-4 text-sm text-gray-500">
                        <div className="flex items-center space-x-1">
                          <Calendar className="w-4 h-4" />
                          <span>
                            {new Date(feedback.timestamp).toLocaleDateString()}
                          </span>
                        </div>
                        {feedback.email && (
                          <div className="flex items-center space-x-1">
                            <Mail className="w-4 h-4" />
                            <span>{feedback.email}</span>
                          </div>
                        )}
                        <div className="flex items-center space-x-1">
                          <User className="w-4 h-4" />
                          <span>ID: {feedback.userId}</span>
                        </div>
                      </div>

                      {/* Expanded Actions */}
                      {/* Expanded Actions */}
                      {isExpanded && (
                        <div className="mt-4 pt-4 border-t border-gray-200">
                          <div className="flex items-center space-x-3">
                            <span className="text-sm font-medium text-gray-700">
                              Update Status:
                            </span>
                            <button
                              onClick={() =>
                                updateStatus(feedback.id, "in_progress")
                              }
                              className="px-3 py-1 text-xs bg-blue-100 text-blue-700 rounded-lg hover:bg-blue-200 transition-colors"
                            >
                              In Progress
                            </button>
                            <button
                              onClick={() =>
                                updateStatus(feedback.id, "reviewed")
                              }
                              className="px-3 py-1 text-xs bg-purple-100 text-purple-700 rounded-lg hover:bg-purple-200 transition-colors"
                            >
                              Reviewed
                            </button>
                            <button
                              onClick={() =>
                                updateStatus(feedback.id, "resolved")
                              }
                              className="px-3 py-1 text-xs bg-green-100 text-green-700 rounded-lg hover:bg-green-200 transition-colors"
                            >
                              Resolved
                            </button>
                            {feedback.email && (
                              <button className="px-3 py-1 text-xs bg-[#0F2FA3] text-white rounded-lg hover:bg-blue-700 transition-colors flex items-center space-x-1">
                                <Mail className="w-3 h-3" />
                                <span>Reply</span>
                              </button>
                            )}
                          </div>
                        </div>
                      )}
                    </div>

                    {/* Expand/Collapse Button */}
                    {/* Expand/Collapse Button */}
                    <button
                      onClick={() =>
                        setExpandedFeedback(isExpanded ? null : feedback.id)
                      }
                      className="ml-4 p-2 hover:bg-gray-100 rounded-lg transition-colors"
                    >
                      {isExpanded ? (
                        <ChevronUp className="w-5 h-5 text-gray-500" />
                      ) : (
                        <ChevronDown className="w-5 h-5 text-gray-500" />
                      )}
                    </button>
                  </div>
                </div>
              </div>
            );
          })}

          {filteredFeedbacks.length === 0 && (
            <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-12 text-center">
              <MessageSquarePlus className="w-12 h-12 text-gray-400 mx-auto mb-4" />
              <p className="text-gray-500">
                No feedback found matching your filters
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default AdminFeedbackDashboard;
