import {
  // ArrowLeft,
  FileText,
  Calendar,
} from "lucide-react";
import { useNavigate } from "react-router-dom";

function TermsAndConditions() {
  const navigate = useNavigate();

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      {/* <div className="bg-white border-b border-gray-200">
        <div className="max-w-4xl mx-auto px-4 py-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <button
                type="button"
                className="flex items-center text-sm text-gray-600 hover:text-gray-900"
                onClick={() => navigate(-1)}
              >
                <ArrowLeft className="w-4 h-4 mr-2" />
                Back
              </button>
              <div className="text-xl font-bold text-gray-900">
                <span className="font-bold">Safeguard</span>{" "}
                <span className="font-normal">Media</span>
              </div>
            </div>
          </div>
        </div>
      </div> */}

      {/* Content */}
      <div className="max-w-4xl mx-auto px-4 py-8">
        <div className="bg-white rounded-lg shadow-sm border border-gray-200">
          {/* Header Section */}
          <div className="border-b border-gray-200 p-8">
            <div className="flex items-center space-x-3 mb-4">
              <div className="bg-blue-100 p-3 rounded-full">
                <FileText className="w-6 h-6 text-blue-600" />
              </div>
              <div>
                <h1 className="text-3xl font-bold text-gray-900">
                  Terms and Conditions
                </h1>
                <div className="flex items-center text-sm text-gray-500 mt-1">
                  <Calendar className="w-4 h-4 mr-1" />
                  Last updated: June 23, 2025
                </div>
              </div>
            </div>
            <p className="text-gray-600">
              Please read these Terms and Conditions carefully before using the
              Safeguard Media platform. By accessing or using our service, you
              agree to be bound by these terms.
            </p>
          </div>

          {/* Terms Content */}
          <div className="p-8 space-y-8">
            {/* Section 1 */}
            <section>
              <h2 className="text-xl font-semibold text-gray-900 mb-4">
                1. Acceptance of Terms
              </h2>
              <div className="text-gray-700 space-y-3">
                <p>
                  By creating an account, accessing, or using Safeguard Media's
                  deepfake detection services ("Service"), you acknowledge that
                  you have read, understood, and agree to be bound by these
                  Terms and Conditions ("Terms"). If you do not agree to these
                  Terms, you may not use our Service.
                </p>
                <p>
                  These Terms constitute a legally binding agreement between you
                  ("User," "you," or "your") and Safeguard Media ("Company,"
                  "we," "us," or "our").
                </p>
              </div>
            </section>

            {/* Section 2 */}
            <section>
              <h2 className="text-xl font-semibold text-gray-900 mb-4">
                2. Description of Service
              </h2>
              <div className="text-gray-700 space-y-3">
                <p>
                  Safeguard Media provides AI-powered deepfake detection and
                  media authentication services. Our platform analyzes uploaded
                  images, videos, and audio files to identify potential
                  synthetic or manipulated content.
                </p>
                <p>The Service includes but is not limited to:</p>
                <ul className="list-disc list-inside ml-4 space-y-1">
                  <li>Deepfake detection algorithms</li>
                  <li>Media analysis and reporting</li>
                  <li>API access for developers</li>
                  <li>Dashboard and analytics tools</li>
                  <li>Customer support and documentation</li>
                </ul>
              </div>
            </section>

            {/* Section 3 */}
            <section>
              <h2 className="text-xl font-semibold text-gray-900 mb-4">
                3. User Accounts and Responsibilities
              </h2>
              <div className="text-gray-700 space-y-3">
                <p>
                  <strong>Account Creation:</strong> You must provide accurate,
                  current, and complete information when creating your account.
                  You are responsible for maintaining the confidentiality of
                  your account credentials.
                </p>
                <p>
                  <strong>Account Security:</strong> You are solely responsible
                  for all activities that occur under your account. Notify us
                  immediately of any unauthorized use of your account.
                </p>
                <p>
                  <strong>Eligibility:</strong> You must be at least 18 years
                  old or the age of majority in your jurisdiction to use our
                  Service.
                </p>
              </div>
            </section>

            {/* Section 4 */}
            <section>
              <h2 className="text-xl font-semibold text-gray-900 mb-4">
                4. Acceptable Use Policy
              </h2>
              <div className="text-gray-700 space-y-3">
                <p>You agree NOT to use the Service to:</p>
                <ul className="list-disc list-inside ml-4 space-y-1">
                  <li>
                    Upload, analyze, or distribute illegal, harmful, or
                    inappropriate content
                  </li>
                  <li>
                    Violate any applicable laws, regulations, or third-party
                    rights
                  </li>
                  <li>
                    Attempt to reverse engineer, hack, or compromise our systems
                  </li>
                  <li>
                    Use the Service for harassment, defamation, or malicious
                    purposes
                  </li>
                  <li>
                    Share your account credentials with unauthorized parties
                  </li>
                  <li>
                    Exceed usage limits or attempt to bypass security measures
                  </li>
                </ul>
                <p>
                  We reserve the right to suspend or terminate accounts that
                  violate this policy.
                </p>
              </div>
            </section>

            {/* Section 5 */}
            <section>
              <h2 className="text-xl font-semibold text-gray-900 mb-4">
                5. Privacy and Data Handling
              </h2>
              <div className="text-gray-700 space-y-3">
                <p>
                  <strong>Data Processing:</strong> By using our Service, you
                  consent to our collection, processing, and analysis of
                  uploaded media files for the purpose of deepfake detection.
                </p>
                <p>
                  <strong>Data Retention:</strong> We may retain uploaded files
                  and analysis results for a limited time to improve our
                  services and provide support.
                </p>
                <p>
                  <strong>Privacy Policy:</strong> Our Privacy Policy,
                  incorporated by reference, governs our data collection and
                  processing practices.
                </p>
              </div>
            </section>

            {/* Section 6 */}
            <section>
              <h2 className="text-xl font-semibold text-gray-900 mb-4">
                6. Intellectual Property Rights
              </h2>
              <div className="text-gray-700 space-y-3">
                <p>
                  <strong>Our IP:</strong> The Service, including all
                  technology, algorithms, software, and documentation, is owned
                  by Safeguard Media and protected by intellectual property
                  laws.
                </p>
                <p>
                  <strong>Your Content:</strong> You retain ownership of media
                  files you upload. By using our Service, you grant us a limited
                  license to process your content for detection purposes.
                </p>
                <p>
                  <strong>Restrictions:</strong> You may not copy, modify,
                  distribute, or create derivative works of our proprietary
                  technology.
                </p>
              </div>
            </section>

            {/* Section 7 */}
            <section>
              <h2 className="text-xl font-semibold text-gray-900 mb-4">
                7. Payment Terms
              </h2>
              <div className="text-gray-700 space-y-3">
                <p>
                  <strong>Subscription Plans:</strong> Paid plans are billed in
                  advance on a monthly or annual basis. All fees are
                  non-refundable except as required by law.
                </p>
                <p>
                  <strong>Payment Processing:</strong> Payments are processed
                  through secure third-party payment processors. You authorize
                  us to charge your payment method.
                </p>
                <p>
                  <strong>Price Changes:</strong> We may modify pricing with 30
                  days' notice. Changes apply to subsequent billing cycles.
                </p>
              </div>
            </section>

            {/* Section 8 */}
            <section>
              <h2 className="text-xl font-semibold text-gray-900 mb-4">
                8. Service Availability and Limitations
              </h2>
              <div className="text-gray-700 space-y-3">
                <p>
                  <strong>Availability:</strong> We strive to maintain high
                  service availability but cannot guarantee uninterrupted
                  access. Maintenance and updates may cause temporary downtime.
                </p>
                <p>
                  <strong>Detection Accuracy:</strong> While we use advanced AI
                  technology, our detection results are not 100% accurate.
                  Results should be used as guidance, not definitive proof.
                </p>
                <p>
                  <strong>Usage Limits:</strong> Plans include specific usage
                  limits. Exceeding limits may result in additional charges or
                  service restrictions.
                </p>
              </div>
            </section>

            {/* Section 9 */}
            <section>
              <h2 className="text-xl font-semibold text-gray-900 mb-4">
                9. Limitation of Liability
              </h2>
              <div className="text-gray-700 space-y-3">
                <p>
                  TO THE MAXIMUM EXTENT PERMITTED BY LAW, SAFEGUARD MEDIA SHALL
                  NOT BE LIABLE FOR ANY INDIRECT, INCIDENTAL, SPECIAL,
                  CONSEQUENTIAL, OR PUNITIVE DAMAGES, INCLUDING BUT NOT LIMITED
                  TO LOSS OF PROFITS, DATA, OR BUSINESS INTERRUPTION.
                </p>
                <p>
                  OUR TOTAL LIABILITY TO YOU FOR ANY CLAIMS ARISING FROM OR
                  RELATED TO THE SERVICE SHALL NOT EXCEED THE AMOUNT YOU PAID US
                  IN THE TWELVE MONTHS PRECEDING THE CLAIM.
                </p>
              </div>
            </section>

            {/* Section 10 */}
            <section>
              <h2 className="text-xl font-semibold text-gray-900 mb-4">
                10. Termination
              </h2>
              <div className="text-gray-700 space-y-3">
                <p>
                  <strong>By You:</strong> You may terminate your account at any
                  time through your account settings or by contacting support.
                </p>
                <p>
                  <strong>By Us:</strong> We may suspend or terminate your
                  account for violation of these Terms, non-payment, or other
                  legitimate business reasons.
                </p>
                <p>
                  <strong>Effect of Termination:</strong> Upon termination, your
                  access to the Service will cease, and we may delete your
                  account data in accordance with our data retention policies.
                </p>
              </div>
            </section>

            {/* Section 11 */}
            <section>
              <h2 className="text-xl font-semibold text-gray-900 mb-4">
                11. Changes to Terms
              </h2>
              <div className="text-gray-700 space-y-3">
                <p>
                  We may update these Terms from time to time. We will notify
                  users of material changes via email or through the Service.
                  Your continued use after changes constitutes acceptance of the
                  updated Terms.
                </p>
              </div>
            </section>

            {/* Section 12 */}
            <section>
              <h2 className="text-xl font-semibold text-gray-900 mb-4">
                12. Governing Law and Disputes
              </h2>
              <div className="text-gray-700 space-y-3">
                <p>
                  These Terms are governed by the laws of [Your Jurisdiction].
                  Any disputes arising from these Terms or the Service shall be
                  resolved through binding arbitration or in the courts of [Your
                  Jurisdiction].
                </p>
              </div>
            </section>

            {/* Contact Section */}
            <section className="bg-gray-50 rounded-lg p-6">
              <h2 className="text-xl font-semibold text-gray-900 mb-4">
                Contact Information
              </h2>
              <div className="text-gray-700 space-y-2">
                <p>
                  If you have questions about these Terms and Conditions, please
                  contact us:
                </p>
                <div className="space-y-1">
                  <p>
                    <strong>Email:</strong> legal@safeguardmedia.com
                  </p>
                  <p>
                    <strong>Address:</strong> [Your Company Address]
                  </p>
                  <p>
                    <strong>Phone:</strong> [Your Phone Number]
                  </p>
                </div>
              </div>
            </section>
          </div>
        </div>

        {/* Footer Actions */}
        <div className="mt-8 flex flex-col sm:flex-row gap-4 justify-center">
          <button
            type="button"
            className="px-6 py-3 bg-[#0F2FA3] hover:bg-blue-700 text-white font-medium rounded-lg"
            onClick={() => navigate("/signup")}
          >
            Accept & Sign Up
          </button>
          <button
            type="button"
            className="px-6 py-3 border border-gray-300 hover:bg-gray-50 text-gray-700 font-medium rounded-lg"
            onClick={() => navigate(-1)}
          >
            Go Back
          </button>
        </div>
      </div>
    </div>
  );
}

export default TermsAndConditions;
