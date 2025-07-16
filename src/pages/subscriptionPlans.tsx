import { Check, Plus, X, Mail, Loader2 } from "lucide-react";
import { useState, useEffect } from "react";
import {
  useSubscriptionPlansQuery,
  useCheckoutMutation,
} from "../services/apiService";
import { useGetUserQuery } from "../services/apiService";
import SafeguardMediaLogo from "../assets/images/SafeguardMedia8.svg";

// Type definitions matching the API service
interface SubscriptionPlan {
  id: string;
  object: string;
  active: boolean;
  attributes: unknown[];
  created: number;
  default_price: string;
  description: string;
  images: unknown[];
  livemode: boolean;
  marketing_features: unknown[];
  metadata: Record<string, unknown>;
  name: string;
  package_dimensions: unknown;
  shippable: unknown;
  statement_descriptor: unknown;
  tax_code: unknown;
  type: string;
  unit_label: unknown;
  updated: number;
  url: unknown;
}

interface SubscriptionPlansResponse {
  success: boolean;
  message: string;
  data: {
    object: string;
    data: SubscriptionPlan[];
    has_more: boolean;
    url: string;
  };
}

const SubscriptionPlans = () => {
  const [selectedTab, setSelectedTab] = useState("Individual");
  const [expandedFAQ, setExpandedFAQ] = useState<number | null>(null);
  const [loadingPlanId, setLoadingPlanId] = useState<string | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const {
    data: userData,
    // isLoading: userLoading,
    // error: userError,
  } = useGetUserQuery();
  // Fetch subscription plans from API
  const {
    data: plansData,
    isLoading,
    error,
  } = useSubscriptionPlansQuery() as {
    data: SubscriptionPlansResponse | undefined;
    isLoading: boolean;
    error: unknown;
  };

  useEffect(() => {
    if (errorMessage) {
      const timer = setTimeout(() => {
        setErrorMessage(null);
      }, 5000); // Auto-dismiss after 5 seconds

      return () => clearTimeout(timer);
    }
  }, [errorMessage]);
  // Checkout mutation
  const [checkout] = useCheckoutMutation();

  // Helper function to map plan names to display information
  const getPlanDisplayInfo = (planName: string) => {
    switch (planName) {
      case "SAFEGUARD_FREE":
        return {
          displayName: "Free",
          price: "$0",
          billing: "Try Safeguard Media for free",
          features: ["Analyze media for up to 4,000 seconds each month."],
        };
      case "SAFEGUARD_PRO":
        return {
          displayName: "Pro",
          price: "$10",
          billing: "per month, billed annually",
          features: [
            "All features in Free mode.",
            "30 media analysis each month.",
          ],
        };
      case "SAFEGUARD_MAX":
        return {
          displayName: "Max",
          price: "$25",
          billing: "per month, billed annually",
          features: [
            "All features in Pro mode.",
            "Unlimited access to all features each month.",
          ],
        };
      default:
        return {
          displayName: planName,
          price: "Contact Us",
          billing: "Custom pricing",
          features: ["Custom features available"],
        };
    }
  };

  // Helper function to get current user plan
  const getCurrentUserPlan = () => {
    console.log(userData?.data?.user?.plan);
    return userData?.data?.user?.plan || null;
  };

  // Helper function to check if plan is current user's plan
  // const isCurrentPlan = (planName: string) => {
  //   const currentPlan = getCurrentUserPlan();
  //   return currentPlan === planName;
  // };

  // Helper function to check if plan is current user's plan
  const isCurrentPlan = (planName: string) => {
    const currentPlan = getCurrentUserPlan();

    // Handle the naming mismatch between API response and plan names
    if (!currentPlan) {
      return false;
    }
    const normalizedCurrentPlan = currentPlan.toUpperCase();
    const normalizedPlanName = planName;

    console.log("Current plan from API:", currentPlan);
    console.log("Normalized current plan:", normalizedCurrentPlan);
    console.log("Plan name to check:", normalizedPlanName);

    return normalizedCurrentPlan === normalizedPlanName;
  };

  // Helper function to get button text and state
  const getButtonConfig = (planName: string) => {
    if (isCurrentPlan(planName)) {
      return {
        text: "Current Plan",
        disabled: true,
      };
    }
    return {
      text: "Upgrade",
      disabled: false,
    };
  };
  // Handle checkout process
  const handleGetStarted = async (plan: SubscriptionPlan) => {
    // Clear any previous error messages
    setErrorMessage(null);

    if (isCurrentPlan(plan.name)) {
      return;
    }
    // For free plan, you might want to handle differently
    if (plan.name === "SAFEGUARD_FREE") {
      // Handle free plan signup logic here
      console.log("Free plan selected");
      return;
    }

    try {
      setLoadingPlanId(plan.id);

      const response = await checkout({
        priceId: plan.default_price,
      }).unwrap();

      if (response.success && response.data.sessionUrl) {
        // Redirect to Stripe checkout
        window.location.href = response.data.sessionUrl;
      } else {
        console.error("Checkout failed:", response);
        setErrorMessage("Failed to initialize checkout. Please try again.");
      }
    } catch (error: unknown) {
      console.error("Checkout error:", error);

      // Extract error message from the API response
      let errorMsg = "An unexpected error occurred. Please try again.";

      if (error && typeof error === "object" && "data" in error) {
        const apiError = error as { data?: { message?: string } };
        if (apiError.data?.message) {
          errorMsg = apiError.data.message;
        }
      } else if (error && typeof error === "object" && "message" in error) {
        const messageError = error as { message: string };
        errorMsg = messageError.message;
      }

      setErrorMessage(errorMsg);
    } finally {
      setLoadingPlanId(null);
    }
  };

  // Sort plans in desired order: Free, Pro, Max with proper null checking
  const sortedPlans = plansData?.data?.data
    ? [...plansData.data.data].sort(
        (a: SubscriptionPlan, b: SubscriptionPlan) => {
          const order = ["SAFEGUARD_FREE", "SAFEGUARD_PRO", "SAFEGUARD_MAX"];
          return order.indexOf(a.name) - order.indexOf(b.name);
        }
      )
    : [];

  return (
    <div className="min-h-screen bg-white">
      {/* Header */}
      <header className="bg-white">
        <div className="max-w-7xl mx-auto px-5 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center">
              <img
                src={SafeguardMediaLogo}
                alt="Safeguardmedia Logo"
                className="h-12 w-auto"
              />
              <span className="text-xl font-bold text-gray-900">
                Safeguardmedia
              </span>
            </div>
            <div></div>
            {/* Mobile menu button */}
            <button className="md:hidden bg-[#0F2FA3] hover:bg-blue-700 text-white px-3 py-1.5 rounded-lg text-sm font-medium">
              Get Started
            </button>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-5 sm:px-6 lg:px-8 py-12">
        {/*  error message section */}
        {errorMessage && (
          <div className="mb-6 max-w-3xl mx-auto">
            <div className="flex items-center p-4 text-sm text-red-600 bg-red-50 border border-red-200 rounded-lg">
              <X className="w-5 h-5 mr-3 flex-shrink-0" />
              <span className="flex-1">{errorMessage}</span>
              <button
                type="button"
                onClick={() => setErrorMessage(null)}
                className="ml-3 text-red-400 hover:text-red-600"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
          </div>
        )}
        {/* Pricing Section */}
        <div className="text-center mb-12">
          <h2 className="text-3xl sm:text-4xl font-bold text-gray-900 mb-8">
            Explore Plans
          </h2>

          {/* Tab Toggle */}
          <div className="inline-flex bg-[#E6E6E6] rounded-full p-1 mb-12">
            <button
              onClick={() => setSelectedTab("Individual")}
              className={`px-6 py-2 rounded-full text-sm font-medium transition-colors ${
                selectedTab === "Individual"
                  ? "bg-[#FBFBEF] text-gray-900 shadow-sm"
                  : "text-gray-600 hover:text-gray-900"
              }`}
            >
              Individual
            </button>
            <button
              onClick={() => setSelectedTab("Teams & Enterprises")}
              className={`px-6 py-2 rounded-full text-sm font-medium transition-colors ${
                selectedTab === "Teams & Enterprises"
                  ? "bg-white text-gray-900 shadow-sm"
                  : "text-gray-600 hover:text-gray-900"
              }`}
            >
              Teams & Enterprises
            </button>
          </div>

          {selectedTab === "Teams & Enterprises" && (
            <div className="w-full flex justify-center">
              <div className="bg-gray-100 border w-1/2 border-blue-200 rounded-xl p-6 text-center">
                <div className="flex justify-center mb-4">
                  <div className="bg-gray-200 p-3 rounded-full">
                    <Mail className="w-6 h-6 text-black" />
                  </div>
                </div>
                <h3 className="text-lg font-semibold text-gray-900 mb-2">
                  Ready for Enterprise Solutions?
                </h3>
                <p className="text-gray-600 mb-4">
                  Get in touch with our team to discuss custom plans, volume
                  discounts, and enterprise features.
                </p>
                <div className="text-sm text-gray-700">
                  <p className="mb-2">Contact us at:</p>
                  <a
                    href="mailto:info@safeguardmedia.io"
                    className="text-black hover:text-gray-700 font-medium hover:underline"
                  >
                    info@safeguardmedia.io
                  </a>
                </div>
              </div>
            </div>
          )}

          {/* Pricing Cards */}
          {selectedTab === "Individual" && (
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 max-w-5xl mx-auto">
              {isLoading ? (
                <div className="col-span-3 flex justify-center items-center py-12">
                  <Loader2 className="w-8 h-8 animate-spin text-gray-500" />
                  <span className="ml-2 text-gray-500">Loading plans...</span>
                </div>
              ) : error ? (
                <div className="col-span-3 text-center py-12">
                  <p className="text-red-500 mb-4">
                    Failed to load subscription plans
                  </p>
                  <p className="text-gray-600">Please try again later</p>
                </div>
              ) : sortedPlans && sortedPlans.length > 0 ? (
                sortedPlans.map((plan: SubscriptionPlan) => {
                  const displayInfo = getPlanDisplayInfo(plan.name);

                  return (
                    <div
                      key={plan.id}
                      className="bg-white rounded-xl border border-gray-200 p-6 sm:p-8"
                    >
                      <div className="text-left h-full flex flex-col">
                        <h3 className="text-lg font-semibold text-gray-900 mb-2">
                          {displayInfo.displayName}
                        </h3>
                        <div className="mb-4">
                          <span className="text-4xl font-bold text-gray-900">
                            {displayInfo.price}
                          </span>
                        </div>
                        <p className="text-sm text-gray-600 mb-6">
                          {displayInfo.billing}
                        </p>

                        <div className="space-y-3 mb-8 flex-1">
                          {displayInfo.features.map((feature, index) => (
                            <div
                              key={index}
                              className="flex items-start space-x-3"
                            >
                              <Check className="w-5 h-5 text-green-500 mt-0.5 flex-shrink-0" />
                              <span className="text-sm text-gray-700">
                                {feature}
                              </span>
                            </div>
                          ))}
                          {plan.description && (
                            <div className="flex items-start space-x-3">
                              <Check className="w-5 h-5 text-green-500 mt-0.5 flex-shrink-0" />
                              <span className="text-sm text-gray-700">
                                {plan.description}
                              </span>
                            </div>
                          )}
                        </div>

                        <div className="w-full flex justify-center items-center">
                          {(() => {
                            const buttonConfig = getButtonConfig(plan.name);
                            const isLoadingThis = loadingPlanId === plan.id;
                            const isDisabled =
                              isLoadingThis || buttonConfig.disabled;

                            return (
                              <button
                                onClick={() => handleGetStarted(plan)}
                                disabled={isDisabled}
                                className={`py-3 px-6 rounded-[40px] font-medium transition-colors flex items-center justify-center min-w-[120px] ${
                                  buttonConfig.disabled
                                    ? "bg-gray-900 text-white cursor-not-allowed opacity-50"
                                    : "bg-gray-900 hover:bg-gray-800 disabled:bg-gray-400 disabled:cursor-not-allowed text-white"
                                }`}
                              >
                                {isLoadingThis ? (
                                  <>
                                    <Loader2 className="animate-spin -ml-1 mr-2 h-4 w-4" />
                                    Processing...
                                  </>
                                ) : (
                                  buttonConfig.text
                                )}
                              </button>
                            );
                          })()}
                        </div>
                      </div>
                    </div>
                  );
                })
              ) : (
                <div className="col-span-3 text-center py-12">
                  <p className="text-gray-500">No plans available</p>
                </div>
              )}
            </div>
          )}
        </div>

        {/* FAQ Section */}
        <div
          id="faq-section"
          className="max-w-3xl mx-auto mb-20 mt-40 max-md:mt-20"
        >
          <h2 className="text-3xl md:text-4xl font-bold text-gray-900 mb-12 text-center">
            Frequently Asked Questions
          </h2>

          <div className="space-y-6">
            {/* FAQ Item 2 - File formats supported */}
            <div className="border-b border-gray-200 pb-6">
              <div className="flex justify-between items-center mb-4">
                <h3 className="text-lg font-semibold text-gray-900">
                  What file formats are supported?
                </h3>
                <button
                  className="text-gray-400 hover:text-gray-600"
                  onClick={() => setExpandedFAQ(expandedFAQ === 1 ? null : 1)}
                >
                  {expandedFAQ === 1 ? (
                    <X className="w-5 h-5" />
                  ) : (
                    <Plus className="w-5 h-5" />
                  )}
                </button>
              </div>
              {expandedFAQ === 1 && (
                <div className="text-gray-600">
                  <p className="mb-3">
                    Currently, SafeguardMedia supports the following formats:
                  </p>
                  <ul className="space-y-2 ml-4">
                    <li>
                      <strong>Videos:</strong> MP4, AVI, MOV
                    </li>
                    <li>
                      <strong>Images:</strong> JPEG, PNG, WEBP
                    </li>
                    <li>
                      <strong>Audio:</strong> MP3, WAV, AAC
                    </li>
                  </ul>
                  <p className="mt-3">
                    We are continuously expanding compatibility based on user
                    feedback and technological needs.
                  </p>
                </div>
              )}
            </div>

            {/* FAQ Item 3 - Content types detection */}
            <div className="border-b border-gray-200 pb-6">
              <div className="flex justify-between items-center mb-4">
                <h3 className="text-lg font-semibold text-gray-900">
                  Does SafeguardMedia detect deepfakes in videos, audio, and
                  images?
                </h3>
                <button
                  className="text-gray-400 hover:text-gray-600"
                  onClick={() => setExpandedFAQ(expandedFAQ === 2 ? null : 2)}
                >
                  {expandedFAQ === 2 ? (
                    <X className="w-5 h-5" />
                  ) : (
                    <Plus className="w-5 h-5" />
                  )}
                </button>
              </div>
              {expandedFAQ === 2 && (
                <div className="text-gray-600">
                  <p className="mb-3">
                    Yes. SafeguardMedia is built to detect manipulation across{" "}
                    three primary content types:
                  </p>
                  <ul className="space-y-2 ml-4">
                    <li>
                      <strong>Videos</strong> (frame-by-frame and audio sync
                      analysis)
                    </li>
                    <li>
                      <strong>Audio</strong> (voice cloning and manipulation
                      detection)
                    </li>
                    <li>
                      <strong>Static images</strong> (facial and pixel-level
                      inconsistencies)
                    </li>
                  </ul>
                </div>
              )}
            </div>

            {/* FAQ Item 4 - Performance across content types */}
            <div className="border-b border-gray-200 pb-6">
              <div className="flex justify-between items-center mb-4">
                <h3 className="text-lg font-semibold text-gray-900">
                  How does performance vary across different content types?
                </h3>
                <button
                  className="text-gray-400 hover:text-gray-600"
                  onClick={() => setExpandedFAQ(expandedFAQ === 3 ? null : 3)}
                >
                  {expandedFAQ === 3 ? (
                    <X className="w-5 h-5" />
                  ) : (
                    <Plus className="w-5 h-5" />
                  )}
                </button>
              </div>
              {expandedFAQ === 3 && (
                <div className="text-gray-600">
                  <p className="mb-3">
                    Performance may vary depending on content quality, length,
                    and type:
                  </p>
                  <ul className="space-y-2 ml-4">
                    <li>
                      <strong>Images:</strong> Near real-time results with high
                      accuracy for manipulated facial features.
                    </li>
                    <li>
                      <strong>Videos:</strong> Slightly longer processing due to
                      frame-by-frame analysis, with strong detection accuracy.
                    </li>
                    <li>
                      <strong>Audio:</strong> Accuracy improves with longer
                      samples (over 5 seconds) and clear speech patterns.
                    </li>
                  </ul>
                  <p className="mt-3">
                    Regardless of format, SafeguardMedia provides a confidence
                    score and visual explanation to help users understand the
                    results.
                  </p>
                </div>
              )}
            </div>

            {/* FAQ Item 5 - Security of uploaded media */}
            <div className="border-b border-gray-200 pb-6">
              <div className="flex justify-between items-center mb-4">
                <h3 className="text-lg font-semibold text-gray-900">
                  How is the security of my uploaded media handled?
                </h3>
                <button
                  className="text-gray-400 hover:text-gray-600"
                  onClick={() => setExpandedFAQ(expandedFAQ === 4 ? null : 4)}
                >
                  {expandedFAQ === 4 ? (
                    <X className="w-5 h-5" />
                  ) : (
                    <Plus className="w-5 h-5" />
                  )}
                </button>
              </div>
              {expandedFAQ === 4 && (
                <div className="text-gray-600">
                  <p>
                    SafeguardMedia prioritizes your privacy and data security.
                    All uploaded content is processed through encrypted
                    channels, and no media is stored without user consent. Our
                    infrastructure complies with industry standards for
                    cybersecurity and data protection to ensure your files
                    remain confidential and secure.
                  </p>
                </div>
              )}
            </div>

            {/* FAQ Item 6 - Staying effective against evolving techniques */}
            <div className="border-b border-gray-200 pb-6">
              <div className="flex justify-between items-center mb-4">
                <h3 className="text-lg font-semibold text-gray-900">
                  How does SafeguardMedia stay effective against evolving
                  deepfake techniques?
                </h3>
                <button
                  className="text-gray-400 hover:text-gray-600"
                  onClick={() => setExpandedFAQ(expandedFAQ === 5 ? null : 5)}
                >
                  {expandedFAQ === 5 ? (
                    <X className="w-5 h-5" />
                  ) : (
                    <Plus className="w-5 h-5" />
                  )}
                </button>
              </div>
              {expandedFAQ === 5 && (
                <div className="text-gray-600">
                  <p>
                    We continuously update our detection algorithms using
                    real-world datasets and the latest advancements in AI
                    research. Our system is built on adaptive learning
                    principles, allowing it to detect even the most recent and
                    sophisticated forms of synthetic media, including
                    adversarially crafted deepfakes.
                  </p>
                </div>
              )}
            </div>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="bg-gray-900 text-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
          <div className="grid md:grid-cols-3 gap-8">
            {/* Company Column */}
            <div>
              <h3 className="text-sm font-medium text-gray-400 mb-4">
                Company
              </h3>
              <ul className="space-y-3">
                <li>
                  <a href="" className="text-gray-300 hover:text-white text-sm">
                    About Us
                  </a>
                </li>
                <li>
                  <a href="" className="text-gray-300 hover:text-white text-sm">
                    Our Technology
                  </a>
                </li>
                <li>
                  <a
                    href="#"
                    className="text-gray-300 hover:text-white text-sm"
                  >
                    Industries
                  </a>
                </li>
                <li>
                  <a
                    href="#"
                    className="text-gray-300 hover:text-white text-sm"
                  >
                    Help Center
                  </a>
                </li>
              </ul>
            </div>

            {/* Cookies Column */}
            <div>
              <h3 className="text-sm font-medium text-gray-400 mb-4">
                Cookies
              </h3>
              <ul className="space-y-3">
                <li>
                  <a
                    href="/terms-and-conditions"
                    className="text-gray-300 hover:text-white text-sm"
                  >
                    Terms of Use
                  </a>
                </li>
                <li>
                  <a
                    href="/terms-and-conditions"
                    className="text-gray-300 hover:text-white text-sm"
                  >
                    Privacy Policies
                  </a>
                </li>
                <li>
                  <a
                    href="/terms-and-conditions"
                    className="text-gray-300 hover:text-white text-sm"
                  >
                    Responsible Disclosure Policy
                  </a>
                </li>
                <li>
                  <a
                    href="/terms-and-conditions"
                    className="text-gray-300 hover:text-white text-sm"
                  >
                    Compliance
                  </a>
                </li>
              </ul>
            </div>

            {/* Contacts Column */}
            <div>
              <h3 className="text-sm font-medium text-gray-400 mb-4">
                Contacts
              </h3>
              <ul className="space-y-3">
                <li>
                  <a
                    href="https://www.instagram.com/safe_guard_media/"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-gray-300 hover:text-white text-sm"
                  >
                    Instagram
                  </a>
                </li>
                <li>
                  <a
                    href="https://x.com/safeguardmedia1"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-gray-300 hover:text-white text-sm"
                  >
                    X
                  </a>
                </li>
                <li>
                  <a
                    href="https://www.linkedin.com/company/safeguardmedia/about/?viewAsMember=true"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-gray-300 hover:text-white text-sm"
                  >
                    LinkedIn
                  </a>
                </li>
                <li>
                  <a
                    href="mailto:info@safeguardmedia.io"
                    className="text-gray-300 hover:text-white text-sm"
                  >
                    Contact Us: info@safeguardmedia.io
                  </a>
                </li>
              </ul>
            </div>
          </div>

          {/* Bottom Section */}
          <div className="border-t border-gray-700 mt-12 pt-8">
            <p className="text-center text-sm text-gray-400">
              © 2025, All Rights Reserved
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default SubscriptionPlans;
