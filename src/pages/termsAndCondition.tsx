import { useNavigate } from "react-router-dom";
import { Button } from "../components/ui/button";
import SafeguardMediaLogo from "../assets/images/SafeguardMedia8.svg";

function TermsOfService() {
  const navigate = useNavigate();

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
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
            <nav className="hidden md:flex items-center space-x-8">
              <a
                href="/plans"
                className="text-gray-600 hover:text-gray-900"
                onClick={() => navigate("/plans")}
              >
                Pricing
              </a>

              <Button
                className="bg-[#0F2FA3] hover:bg-blue-700 text-white px-4 py-2 rounded-[50px]"
                onClick={() => navigate("/signin")}
              >
                Get Started
              </Button>
            </nav>
          </div>
        </div>
      </header>

      {/* Content */}
      <div className="max-w-4xl mx-auto px-4 py-8">
        <div className="bg-white rounded-lg shadow-sm">
          {/* Header Section */}
          <div className="p-8 border-b border-gray-200">
            <div className="flex items-center justify-center space-x-3 mb-4">
              <div className="flex items-center flex-col justify-center">
                <h1 className="text-3xl font-bold text-center text-gray-900">
                  Terms of Service
                </h1>
              </div>
            </div>
            <p className="text-gray-600 text-center">
              Please read these Terms of Service carefully before using the
              SafeguardMedia platform. By accessing or using our service, you
              agree to be bound by these terms.
            </p>
          </div>

          {/* Terms Content */}
          <div className="p-8 prose max-w-none">
            <p className="text-gray-700 mb-6">
              Welcome to SafeguardMedia ("SafeguardMedia," "we," "our," or
              "us"). These Terms of Service ("Terms") govern your use of our
              website located at [www.safeguardmedia.org] (the "Site"), browser
              extension, detection tools, and related services (collectively,
              the "Service").
            </p>
            <p className="text-gray-700 mb-8">
              By accessing or using the Service, you agree to be bound by these
              Terms and our [Privacy Policy]. If you do not agree, please do not
              use the Service.
            </p>

            <section className="mb-8">
              <h2 className="text-2xl font-bold text-gray-900 mb-4">
                1. The Service
              </h2>
              <p className="text-gray-700 mb-4">
                SafeguardMedia provides users with tools to analyze and verify
                multimedia content, including images, video, and audio, for
                signs of manipulation using AI-based detection techniques. The
                Service is intended to help users better understand the
                credibility of content online.
              </p>
              <p className="text-gray-700">
                The platform is currently in beta testing and functionality may
                evolve or change without notice.
              </p>
            </section>

            <section className="mb-8">
              <h2 className="text-2xl font-bold text-gray-900 mb-4">
                2. Eligibility
              </h2>
              <p className="text-gray-700">
                To use the Service, you must be at least 18 years old and have
                the legal capacity to enter into a binding agreement. If you use
                the Service on behalf of an organization, you represent and
                warrant that you have authority to bind that organization.
              </p>
            </section>

            <section className="mb-8">
              <h2 className="text-2xl font-bold text-gray-900 mb-4">
                3. Acceptable Use
              </h2>
              <p className="text-gray-700 mb-4">You agree that you will not:</p>
              <ul className="list-disc list-inside text-gray-700 space-y-2 ml-4">
                <li>
                  Use the Service in violation of any local, state, national, or
                  international law.
                </li>
                <li>
                  Upload content that contains personal data, copyrighted
                  material, or sensitive information without lawful
                  authorization.
                </li>
                <li>
                  Attempt to access or reverse engineer the Service's underlying
                  code, models, or infrastructure.
                </li>
                <li>
                  Upload or submit malicious content, malware, or disinformation
                  to the Service.
                </li>
                <li>
                  Use the Service for surveillance, profiling, or to harm or
                  mislead others.
                </li>
              </ul>
            </section>

            <section className="mb-8">
              <h2 className="text-2xl font-bold text-gray-900 mb-4">
                4. User Submissions
              </h2>
              <p className="text-gray-700 mb-4">
                You may submit media files ("Submissions") to the Service for
                analysis. By submitting content, you affirm that:
              </p>
              <ul className="list-disc list-inside text-gray-700 space-y-2 ml-4 mb-4">
                <li>
                  You either own the content or have permission to upload and
                  analyze it.
                </li>
                <li>
                  The content does not contain confidential or private
                  information unless you have consent or legal authority to
                  include it.
                </li>
                <li>
                  You grant SafeguardMedia a non-exclusive, royalty-free, and
                  revocable license to process the submission for the sole
                  purpose of providing analysis and improving our Service.
                </li>
              </ul>
              <p className="text-gray-700">
                We do not claim ownership of your uploaded content, but you
                acknowledge that SafeguardMedia may retain a record of processed
                content for model improvement, research, or quality assurance
                unless otherwise requested.
              </p>
            </section>

            <section className="mb-8">
              <h2 className="text-2xl font-bold text-gray-900 mb-4">
                5. Account and Security
              </h2>
              <p className="text-gray-700 mb-4">
                Some features may require account registration. You agree to:
              </p>
              <ul className="list-disc list-inside text-gray-700 space-y-2 ml-4 mb-4">
                <li>Provide accurate and complete information</li>
                <li>Keep your login credentials confidential</li>
                <li>
                  Notify us immediately of any unauthorized access or security
                  breach
                </li>
              </ul>
              <p className="text-gray-700">
                You are responsible for all activity under your account.
              </p>
            </section>

            <section className="mb-8">
              <h2 className="text-2xl font-bold text-gray-900 mb-4">
                6. Intellectual Property
              </h2>
              <p className="text-gray-700 mb-4">
                All intellectual property rights in the Service, including but
                not limited to models, code, logos, interfaces, and datasets,
                are owned by or licensed to SafeguardMedia. These Terms do not
                grant you any right to use the SafeguardMedia brand or
                trademarks without written permission.
              </p>
              <p className="text-gray-700">
                You agree not to copy, distribute, modify, or create derivative
                works from the Service or its components.
              </p>
            </section>

            <section className="mb-8">
              <h2 className="text-2xl font-bold text-gray-900 mb-4">
                7. Feedback
              </h2>
              <p className="text-gray-700">
                We welcome suggestions and feedback. By submitting feedback, you
                grant us the unrestricted right to use it without compensation
                or obligation.
              </p>
            </section>

            <section className="mb-8">
              <h2 className="text-2xl font-bold text-gray-900 mb-4">
                8. Disclaimers
              </h2>
              <p className="text-gray-700 mb-4">
                The Service is provided "as is" and "as available."
                SafeguardMedia makes no guarantees regarding:
              </p>
              <ul className="list-disc list-inside text-gray-700 space-y-2 ml-4 mb-4">
                <li>The accuracy or completeness of results</li>
                <li>The reliability of detection models</li>
                <li>Uninterrupted or error-free service</li>
              </ul>
              <p className="text-gray-700">
                SafeguardMedia does not guarantee that flagged content is
                definitively manipulated, nor does a clean result guarantee
                authenticity. Our Service is a tool—not a final judgment
                platform.
              </p>
            </section>

            <section className="mb-8">
              <h2 className="text-2xl font-bold text-gray-900 mb-4">
                9. Limitation of Liability
              </h2>
              <p className="text-gray-700 mb-4">
                To the fullest extent permitted by law, SafeguardMedia shall not
                be liable for:
              </p>
              <ul className="list-disc list-inside text-gray-700 space-y-2 ml-4 mb-4">
                <li>Any loss of data, profits, business, or goodwill</li>
                <li>Any indirect, incidental, or consequential damages</li>
                <li>
                  Any harm caused by misinformation, false negatives, or false
                  positives from our detection tools
                </li>
              </ul>
              <p className="text-gray-700">
                Your use of the Service is at your own risk.
              </p>
            </section>

            <section className="mb-8">
              <h2 className="text-2xl font-bold text-gray-900 mb-4">
                10. Termination
              </h2>
              <p className="text-gray-700">
                We reserve the right to suspend or terminate access to the
                Service at our discretion, without notice, if we believe a user
                has violated these Terms or poses a risk to the platform or
                others.
              </p>
            </section>

            <section className="mb-8">
              <h2 className="text-2xl font-bold text-gray-900 mb-4">
                11. Changes to the Service
              </h2>
              <p className="text-gray-700">
                We may update, enhance, or discontinue parts of the Service at
                any time without notice. Continued use of the Service after
                changes constitutes acceptance of the updated Terms.
              </p>
            </section>

            <section className="mb-8">
              <h2 className="text-2xl font-bold text-gray-900 mb-4">
                12. Governing Law
              </h2>
              <p className="text-gray-700">
                These Terms are governed by the laws of the State of New Jersey
                (or your local jurisdiction if required). You agree that any
                legal action must be brought in courts located in New Jersey,
                unless otherwise required by law.
              </p>
            </section>

            {/* Contact Section */}
            <div className="bg-gray-50 rounded-lg p-6 mt-8">
              <h2 className="text-xl font-semibold text-gray-900 mb-4">
                13. Contact
              </h2>
              <div className="text-gray-700 space-y-2">
                <p>
                  If you have questions about these Terms of Service, please
                  contact us at:
                </p>
                <div className="space-y-1">
                  <p>
                    <strong>Email:</strong> info@safeguardmedia.org
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

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
              <h3 className="text-sm font-medium text-gray-400 mb-4">Legal</h3>
              <ul className="space-y-3">
                <li>
                  <a
                    href="/terms-of-service"
                    className="text-gray-300 hover:text-white text-sm"
                  >
                    Terms of Service
                  </a>
                </li>
                <li>
                  <a
                    href="/privacy-policy"
                    className="text-gray-300 hover:text-white text-sm"
                  >
                    Privacy Policy
                  </a>
                </li>
                <li>
                  <a
                    href="/responsible-disclosure"
                    className="text-gray-300 hover:text-white text-sm"
                  >
                    Responsible Disclosure Policy
                  </a>
                </li>
                <li>
                  <a
                    href="/compliance"
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
                    href="mailto:info@safeguardmedia.org"
                    className="text-gray-300 hover:text-white text-sm"
                  >
                    Contact Us: info@safeguardmedia.org
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
}

export default TermsOfService;
