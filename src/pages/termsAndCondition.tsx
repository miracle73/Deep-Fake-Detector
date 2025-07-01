import {
  // ArrowLeft,
  ChevronDown,
  ChevronUp,
} from "lucide-react";
import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "../components/ui/button";

function TermsAndConditions() {
  const navigate = useNavigate();
  const [expandedSections, setExpandedSections] = useState<
    Record<number, boolean>
  >({});

  interface ExpandedSections {
    [key: number]: boolean;
  }

  const toggleSection = (sectionId: number) => {
    setExpandedSections((prev: ExpandedSections) => ({
      ...prev,
      [sectionId]: !prev[sectionId],
    }));
  };

  const sections = [
    {
      id: 1,
      title: "Acceptance of Terms",
      content: (
        <div className="text-gray-700 space-y-3">
          <p>
            By creating an account, accessing, or using Safeguard Media's
            deepfake detection services ("Service"), you acknowledge that you
            have read, understood, and agree to be bound by these Terms and
            Conditions ("Terms"). If you do not agree to these Terms, you may
            not use our Service.
          </p>
          <p>
            These Terms constitute a legally binding agreement between you
            ("User," "you," or "your") and Safeguard Media ("Company," "we,"
            "us," or "our").
          </p>
        </div>
      ),
    },
    {
      id: 2,
      title: "Description of Service",
      content: (
        <div className="text-gray-700 space-y-3">
          <p>
            Safeguard Media provides AI-powered deepfake detection and media
            authentication services. Our platform analyzes uploaded images,
            videos, and audio files to identify potential synthetic or
            manipulated content.
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
      ),
    },
    {
      id: 3,
      title: "User Accounts and Responsibilities",
      content: (
        <div className="text-gray-700 space-y-3">
          <p>
            <strong>Account Creation:</strong> You must provide accurate,
            current, and complete information when creating your account. You
            are responsible for maintaining the confidentiality of your account
            credentials.
          </p>
          <p>
            <strong>Account Security:</strong> You are solely responsible for
            all activities that occur under your account. Notify us immediately
            of any unauthorized use of your account.
          </p>
          <p>
            <strong>Eligibility:</strong> You must be at least 18 years old or
            the age of majority in your jurisdiction to use our Service.
          </p>
        </div>
      ),
    },
    {
      id: 4,
      title: "Acceptable Use Policy",
      content: (
        <div className="text-gray-700 space-y-3">
          <p>You agree NOT to use the Service to:</p>
          <ul className="list-disc list-inside ml-4 space-y-1">
            <li>
              Upload, analyze, or distribute illegal, harmful, or inappropriate
              content
            </li>
            <li>
              Violate any applicable laws, regulations, or third-party rights
            </li>
            <li>
              Attempt to reverse engineer, hack, or compromise our systems
            </li>
            <li>
              Use the Service for harassment, defamation, or malicious purposes
            </li>
            <li>Share your account credentials with unauthorized parties</li>
            <li>Exceed usage limits or attempt to bypass security measures</li>
          </ul>
          <p>
            We reserve the right to suspend or terminate accounts that violate
            this policy.
          </p>
        </div>
      ),
    },
    {
      id: 5,
      title: "Privacy and Data Handling",
      content: (
        <div className="text-gray-700 space-y-3">
          <p>
            <strong>Data Processing:</strong> By using our Service, you consent
            to our collection, processing, and analysis of uploaded media files
            for the purpose of deepfake detection.
          </p>
          <p>
            <strong>Data Retention:</strong> We may retain uploaded files and
            analysis results for a limited time to improve our services and
            provide support.
          </p>
          <p>
            <strong>Privacy Policy:</strong> Our Privacy Policy, incorporated by
            reference, governs our data collection and processing practices.
          </p>
        </div>
      ),
    },
    {
      id: 6,
      title: "Intellectual Property Rights",
      content: (
        <div className="text-gray-700 space-y-3">
          <p>
            <strong>Our IP:</strong> The Service, including all technology,
            algorithms, software, and documentation, is owned by Safeguard Media
            and protected by intellectual property laws.
          </p>
          <p>
            <strong>Your Content:</strong> You retain ownership of media files
            you upload. By using our Service, you grant us a limited license to
            process your content for detection purposes.
          </p>
          <p>
            <strong>Restrictions:</strong> You may not copy, modify, distribute,
            or create derivative works of our proprietary technology.
          </p>
        </div>
      ),
    },
    {
      id: 7,
      title: "Payment Terms",
      content: (
        <div className="text-gray-700 space-y-3">
          <p>
            <strong>Subscription Plans:</strong> Paid plans are billed in
            advance on a monthly or annual basis. All fees are non-refundable
            except as required by law.
          </p>
          <p>
            <strong>Payment Processing:</strong> Payments are processed through
            secure third-party payment processors. You authorize us to charge
            your payment method.
          </p>
          <p>
            <strong>Price Changes:</strong> We may modify pricing with 30 days'
            notice. Changes apply to subsequent billing cycles.
          </p>
        </div>
      ),
    },
    {
      id: 8,
      title: "Service Availability and Limitations",
      content: (
        <div className="text-gray-700 space-y-3">
          <p>
            <strong>Availability:</strong> We strive to maintain high service
            availability but cannot guarantee uninterrupted access. Maintenance
            and updates may cause temporary downtime.
          </p>
          <p>
            <strong>Detection Accuracy:</strong> While we use advanced AI
            technology, our detection results are not 100% accurate. Results
            should be used as guidance, not definitive proof.
          </p>
          <p>
            <strong>Usage Limits:</strong> Plans include specific usage limits.
            Exceeding limits may result in additional charges or service
            restrictions.
          </p>
        </div>
      ),
    },
    {
      id: 9,
      title: "Limitation of Liability",
      content: (
        <div className="text-gray-700 space-y-3">
          <p>
            TO THE MAXIMUM EXTENT PERMITTED BY LAW, SAFEGUARD MEDIA SHALL NOT BE
            LIABLE FOR ANY INDIRECT, INCIDENTAL, SPECIAL, CONSEQUENTIAL, OR
            PUNITIVE DAMAGES, INCLUDING BUT NOT LIMITED TO LOSS OF PROFITS,
            DATA, OR BUSINESS INTERRUPTION.
          </p>
          <p>
            OUR TOTAL LIABILITY TO YOU FOR ANY CLAIMS ARISING FROM OR RELATED TO
            THE SERVICE SHALL NOT EXCEED THE AMOUNT YOU PAID US IN THE TWELVE
            MONTHS PRECEDING THE CLAIM.
          </p>
        </div>
      ),
    },
    {
      id: 10,
      title: "Termination",
      content: (
        <div className="text-gray-700 space-y-3">
          <p>
            <strong>By You:</strong> You may terminate your account at any time
            through your account settings or by contacting support.
          </p>
          <p>
            <strong>By Us:</strong> We may suspend or terminate your account for
            violation of these Terms, non-payment, or other legitimate business
            reasons.
          </p>
          <p>
            <strong>Effect of Termination:</strong> Upon termination, your
            access to the Service will cease, and we may delete your account
            data in accordance with our data retention policies.
          </p>
        </div>
      ),
    },
    {
      id: 11,
      title: "Changes to Terms",
      content: (
        <div className="text-gray-700 space-y-3">
          <p>
            We may update these Terms from time to time. We will notify users of
            material changes via email or through the Service. Your continued
            use after changes constitutes acceptance of the updated Terms.
          </p>
        </div>
      ),
    },
    {
      id: 12,
      title: "Governing Law and Disputes",
      content: (
        <div className="text-gray-700 space-y-3">
          <p>
            These Terms are governed by the laws of [Your Jurisdiction]. Any
            disputes arising from these Terms or the Service shall be resolved
            through binding arbitration or in the courts of [Your Jurisdiction].
          </p>
        </div>
      ),
    },
  ];
  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center">
              <span className="text-xl font-bold text-gray-900">Safeguard</span>
              <span className="text-xl font-normal text-gray-600 ml-1">
                Media
              </span>
            </div>
            <nav className="hidden md:flex items-center space-x-8">
              <a href="#" className="text-gray-600 hover:text-gray-900">
                Products
              </a>
              <a
                href="#"
                className="text-gray-600 hover:text-gray-900"
                onClick={() => navigate("/plans")}
              >
                Pricing
              </a>
              <a href="#" className="text-gray-600 hover:text-gray-900">
                News
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
        <div className=" rounded-lg shadow-sm ">
          {/* Header Section */}
          <div className=" p-8">
            <div className="flex items-center justify-center space-x-3 mb-4">
              <div className="flex items-center flex-col justify-center">
                <h1 className="text-3xl font-bold text-center text-gray-900">
                  Terms and Conditions
                </h1>
                <div className="flex items-center text-center text-sm text-gray-500 mt-1">
                  Last updated: June 23, 2025
                </div>
              </div>
            </div>
            <p className="text-gray-600 text-center">
              Please read these Terms and Conditions carefully before using the
              Safeguard Media platform. By accessing or using our service, you
              agree to be bound by these terms.
            </p>
          </div>

          {/* Terms Content - Collapsible Sections */}
          <div className="p-8 space-y-4">
            {sections.map((section) => (
              <div
                key={section.id}
                className="bg-white rounded-lg shadow-md border border-gray-200"
              >
                <div
                  className="flex items-center justify-between p-4 cursor-pointer hover:bg-gray-50 transition-colors"
                  onClick={() => toggleSection(section.id)}
                >
                  <h2 className="text-lg font-semibold text-gray-900">
                    {section.id}. {section.title}
                  </h2>
                  {expandedSections[section.id] ? (
                    <ChevronUp className="w-5 h-5 text-gray-500" />
                  ) : (
                    <ChevronDown className="w-5 h-5 text-gray-500" />
                  )}
                </div>
                {expandedSections[section.id] && (
                  <div className="px-4 pb-4 border-t border-gray-100">
                    <div className="pt-4">{section.content}</div>
                  </div>
                )}
              </div>
            ))}

            {/* Contact Section */}
            <div className="bg-gray-50 rounded-lg shadow-md border border-gray-200 p-6 mt-6">
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
                    <strong>Email:</strong>info@safeguardmedia.io
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
                  <a
                    href="#"
                    className="text-gray-300 hover:text-white text-sm"
                  >
                    About Us
                  </a>
                </li>
                <li>
                  <a
                    href="#"
                    className="text-gray-300 hover:text-white text-sm"
                  >
                    Careers
                  </a>
                </li>
                <li>
                  <a
                    href="#"
                    className="text-gray-300 hover:text-white text-sm"
                  >
                    Media
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
                    href="#"
                    className="text-gray-300 hover:text-white text-sm"
                  >
                    Terms of Use
                  </a>
                </li>
                <li>
                  <a
                    href="#"
                    className="text-gray-300 hover:text-white text-sm"
                  >
                    Privacy Policies
                  </a>
                </li>
                <li>
                  <a
                    href="#"
                    className="text-gray-300 hover:text-white text-sm"
                  >
                    Responsible Disclosure Policy
                  </a>
                </li>
                <li>
                  <a
                    href="#"
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
}

export default TermsAndConditions;
