import { useState, useEffect } from "react";
import {
  ArrowLeft,
  Star,
  Send,
  CheckCircle,
  MessageSquarePlus,
  Sparkles,
  ThumbsUp,
  ThumbsDown,
  AlertCircle,
  X,
} from "lucide-react";
import { useCreateFeedbackMutation } from "../services/apiService";

const Feedback = () => {
  const [rating, setRating] = useState(0);
  const [hoveredRating, setHoveredRating] = useState(0);
  const [feedbackType, setFeedbackType] = useState("General Feedback");
  const [feedbackText, setFeedbackText] = useState("");
  const [email, setEmail] = useState("");
  const [isSubmitted, setIsSubmitted] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [createFeedback] = useCreateFeedbackMutation();
  const [errorMessage, setErrorMessage] = useState<string>("");

  useEffect(() => {
    if (errorMessage) {
      const timer = setTimeout(() => {
        setErrorMessage("");
      }, 3000);

      return () => clearTimeout(timer);
    }
  }, [errorMessage]);

  const feedbackTypes = [
    {
      id: "General Feedback",
      label: "General Feedback",
      icon: MessageSquarePlus,
    },
    { id: "Bug Report", label: "Bug Report", icon: ThumbsDown },
    { id: "Feature Request", label: "Feature Request", icon: Sparkles },
    { id: "Improvement", label: "Improvement", icon: ThumbsUp },
  ];

  const handleSubmit = async () => {
    if (!feedbackText) return;

    setIsSubmitting(true);

    try {
      const feedbackData = {
        type: feedbackType,
        rating,
        description: feedbackText,
        email: email || undefined,
      };

      await createFeedback(feedbackData).unwrap();

      setIsSubmitted(true);

      // Reset after showing success
      setTimeout(() => {
        setIsSubmitted(false);
        setRating(0);
        setFeedbackText("");
        setEmail("");
        setFeedbackType("General Feedback");
      }, 3000);
    } catch (error) {
      console.error("Failed to submit feedback:", error);

      if (error && typeof error === "object" && "data" in error) {
        const apiError = error as {
          data?: { message?: string };
        };
        if (apiError.data?.message) {
          setErrorMessage(apiError.data.message);
        } else {
          setErrorMessage("Failed to submit feedback. Please try again.");
        }
      } else if (error && typeof error === "object" && "message" in error) {
        const messageError = error as { message: string };
        setErrorMessage(messageError.message);
      } else {
        setErrorMessage("Failed to submit feedback. Please try again.");
      }
    } finally {
      setIsSubmitting(false);
    }
  };

  if (isSubmitted) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-blue-50 flex items-center justify-center p-4">
        <div className="bg-white rounded-2xl shadow-xl p-8 max-w-md w-full text-center transform animate-fadeIn">
          <div className="w-20 h-20 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-4">
            <CheckCircle className="w-10 h-10 text-green-600" />
          </div>
          <h2 className="text-2xl font-bold text-gray-900 mb-2">Thank You!</h2>
          <p className="text-gray-600">
            Your feedback has been successfully submitted. We appreciate your
            input!
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-blue-50">
      {/* Header */}
      <header className="bg-white border-b border-gray-200 sticky top-0 z-10 backdrop-blur-lg bg-opacity-90">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <button
                onClick={() => window.history.back()}
                className="p-2 hover:bg-gray-100 rounded-lg transition-all duration-200 group"
              >
                <ArrowLeft className="w-5 h-5 text-gray-600 group-hover:text-[#0F2FA3] transition-colors" />
              </button>
              <div>
                <h1 className="text-xl font-bold text-gray-900">
                  Send Feedback
                </h1>
                <p className="text-sm text-gray-500">
                  Help us improve Safeguardmedia
                </p>
              </div>
            </div>
            <div className="hidden sm:block">
              <div className="px-4 py-2 bg-[#0F2FA3]/10 rounded-full border border-[#0F2FA3]/20">
                <span className="text-sm font-medium text-[#0F2FA3]">
                  Your opinion matters!
                </span>
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Error Message */}
      {errorMessage && (
        <div className="max-w-4xl mx-auto px-4 sm:px-6 pt-4">
          <div className="flex items-center p-3 text-sm text-red-600 bg-red-50 border border-red-200 rounded-lg">
            <AlertCircle className="w-4 h-4 mr-2 flex-shrink-0" />
            <span>{errorMessage}</span>
            <button
              type="button"
              onClick={() => setErrorMessage("")}
              className="ml-auto text-red-400 hover:text-red-600"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        </div>
      )}

      {/* Main Content */}
      <div className="max-w-4xl mx-auto px-4 sm:px-6 py-8">
        <div className="space-y-6">
          {/* Feedback Type Selection */}
          <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 transform transition-all duration-300 hover:shadow-lg">
            <label className="block text-sm font-semibold text-gray-700 mb-4">
              What type of feedback do you have?
            </label>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              {feedbackTypes.map((type) => {
                const Icon = type.icon;
                return (
                  <button
                    key={type.id}
                    onClick={() => setFeedbackType(type.id)}
                    className={`p-3 rounded-lg border-2 transition-all duration-200 transform hover:scale-105 ${
                      feedbackType === type.id
                        ? "border-[#0F2FA3] bg-[#0F2FA3]/10 shadow-md"
                        : "border-gray-200 hover:border-gray-300"
                    }`}
                  >
                    <Icon
                      className={`w-5 h-5 mx-auto mb-2 ${
                        feedbackType === type.id
                          ? "text-[#0F2FA3]"
                          : "text-gray-500"
                      }`}
                    />
                    <span
                      className={`text-xs font-medium ${
                        feedbackType === type.id
                          ? "text-[#0F2FA3]"
                          : "text-gray-600"
                      }`}
                    >
                      {type.label}
                    </span>
                  </button>
                );
              })}
            </div>
          </div>

          {/* Rating Section */}
          <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 transform transition-all duration-300 hover:shadow-lg">
            <label className="block text-sm font-semibold text-gray-700 mb-4">
              How would you rate your experience?
            </label>
            <div className="flex justify-center space-x-2">
              {[1, 2, 3, 4, 5].map((star) => (
                <button
                  key={star}
                  onClick={() => setRating(star)}
                  onMouseEnter={() => setHoveredRating(star)}
                  onMouseLeave={() => setHoveredRating(0)}
                  className="transform transition-all duration-200 hover:scale-110"
                >
                  <Star
                    className={`w-8 h-8 transition-colors duration-200 ${
                      star <= (hoveredRating || rating)
                        ? "fill-yellow-400 text-yellow-400"
                        : "text-gray-300"
                    }`}
                  />
                </button>
              ))}
            </div>
            <p className="text-center mt-3 text-sm text-gray-500">
              {rating === 0 && "Click to rate"}
              {rating === 1 && "Poor"}
              {rating === 2 && "Fair"}
              {rating === 3 && "Good"}
              {rating === 4 && "Very Good"}
              {rating === 5 && "Excellent"}
            </p>
          </div>

          {/* Feedback Text */}
          <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 transform transition-all duration-300 hover:shadow-lg">
            <label className="block text-sm font-semibold text-gray-700 mb-2">
              Tell us more
            </label>
            <textarea
              value={feedbackText}
              onChange={(e) => setFeedbackText(e.target.value)}
              placeholder="Share your thoughts, suggestions, or report issues..."
              className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-[#0F2FA3] focus:border-transparent resize-none transition-all duration-200"
              rows={6}
            />
            <div className="mt-2 text-right">
              <span
                className={`text-xs ${
                  feedbackText.length > 500 ? "text-red-500" : "text-gray-400"
                }`}
              >
                {feedbackText.length}/500
              </span>
            </div>
          </div>

          {/* Email Input */}
          <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 transform transition-all duration-300 hover:shadow-lg">
            <label className="block text-sm font-semibold text-gray-700 mb-2">
              Email (optional)
            </label>
            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="your.email@example.com"
              className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-[#0F2FA3] focus:border-transparent transition-all duration-200"
            />
            <p className="mt-2 text-xs text-gray-500">
              We'll only use this to follow up on your feedback if needed
            </p>
          </div>

          {/* Submit Button */}
          <div className="flex justify-end space-x-3">
            <button
              onClick={() => window.history.back()}
              className="px-6 py-3 border border-gray-300 rounded-lg hover:bg-gray-50 transition-all duration-200 font-medium text-gray-700"
            >
              Cancel
            </button>
            <button
              onClick={handleSubmit}
              disabled={!feedbackText || isSubmitting}
              className={`px-6 py-3 rounded-lg font-medium transition-all duration-200 transform hover:scale-105 flex items-center space-x-2 ${
                !feedbackText || isSubmitting
                  ? "bg-gray-300 text-gray-500 cursor-not-allowed"
                  : "bg-[#0F2FA3] text-white hover:bg-blue-700 shadow-lg hover:shadow-xl"
              }`}
            >
              {isSubmitting ? (
                <>
                  <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                  <span>Sending...</span>
                </>
              ) : (
                <>
                  <Send className="w-4 h-4" />
                  <span>Send Feedback</span>
                </>
              )}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Feedback;
