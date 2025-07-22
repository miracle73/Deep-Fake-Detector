import { useNavigate } from "react-router-dom";
import { Button } from "../components/ui/button";
import SafeguardMediaLogo from "../assets/images/SafeguardMedia8.svg";

function PrivacyPolicy() {
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
                  Privacy Policy
                </h1>
              </div>
            </div>
            <p className="text-gray-600 text-center">
              SafeguardMedia is committed to protecting your privacy. This
              Privacy Policy describes how we collect, use, and protect your
              information.
            </p>
          </div>

          {/* Privacy Policy Content */}
          <div className="p-8 prose max-w-none">
            <p className="text-gray-700 mb-8">
              SafeguardMedia ("we," "our," or "us") is committed to protecting
              your privacy. This Privacy Policy describes the types of
              information we collect, how we use it, and your choices regarding
              your data when you access our website, tools, or services
              (collectively, the "Service").
            </p>
            <p className="text-gray-700 mb-8">
              By using our Service, you consent to the practices described in
              this Privacy Policy.
            </p>

            <section className="mb-8">
              <h2 className="text-2xl font-bold text-gray-900 mb-4">
                1. Information We Collect
              </h2>
              <p className="text-gray-700 mb-4">
                We collect limited data necessary to operate and improve our
                media verification services. This may include:
              </p>

              <div className="mb-4">
                <h3 className="text-lg font-semibold text-gray-800 mb-2">
                  a. Information You Provide Voluntarily
                </h3>
                <ul className="list-disc list-inside text-gray-700 space-y-2 ml-4">
                  <li>
                    Email address (if you create an account, join a waitlist, or
                    contact us)
                  </li>
                  <li>Phone number</li>
                  <li>Media files you upload for analysis</li>
                  <li>Feedback or support requests</li>
                </ul>
              </div>

              <div className="mb-4">
                <h3 className="text-lg font-semibold text-gray-800 mb-2">
                  b. Information We Collect Automatically
                </h3>
                <ul className="list-disc list-inside text-gray-700 space-y-2 ml-4">
                  <li>IP address and browser type (for basic analytics)</li>
                  <li>
                    Usage data (e.g., number of uploads, timestamp of use)
                  </li>
                  <li>
                    Diagnostic logs (for security, performance, and abuse
                    monitoring)
                  </li>
                </ul>
              </div>

              <p className="text-gray-700">
                We do <strong>not</strong> use tracking cookies or sell your
                personal data.
              </p>
            </section>

            <section className="mb-8">
              <h2 className="text-2xl font-bold text-gray-900 mb-4">
                2. How We Use Your Information
              </h2>
              <p className="text-gray-700 mb-4">We use your information to:</p>
              <ul className="list-disc list-inside text-gray-700 space-y-2 ml-4 mb-4">
                <li>Operate, maintain, and improve the Service</li>
                <li>Deliver detection results</li>
                <li>Respond to inquiries or feedback</li>
                <li>
                  Conduct internal research and model training (media is
                  anonymized when used for this purpose)
                </li>
                <li>Ensure platform integrity and prevent abuse</li>
              </ul>
              <p className="text-gray-700">
                We do <strong>not</strong> use your data for advertising or
                resale.
              </p>
            </section>

            <section className="mb-8">
              <h2 className="text-2xl font-bold text-gray-900 mb-4">
                3. Data Retention
              </h2>
              <p className="text-gray-700 mb-4">
                We retain your media uploads temporarily for the purpose of
                processing and model improvement. Files may be stored securely
                for internal quality checks or deleted based on user request or
                after a defined time window.
              </p>
              <p className="text-gray-700">
                User account data (e.g., email address) is retained until the
                user deletes their account or requests removal.
              </p>
            </section>

            <section className="mb-8">
              <h2 className="text-2xl font-bold text-gray-900 mb-4">
                4. Data Security
              </h2>
              <p className="text-gray-700">
                We use secure encryption, cloud-based storage, and access
                controls to protect your information. However, no method of
                transmission over the internet is 100% secure, and we cannot
                guarantee absolute security.
              </p>
            </section>

            <section className="mb-8">
              <h2 className="text-2xl font-bold text-gray-900 mb-4">
                5. Data Sharing
              </h2>
              <p className="text-gray-700 mb-4">
                We do <strong>not</strong> sell, rent, or share your personal
                data with third parties except:
              </p>
              <ul className="list-disc list-inside text-gray-700 space-y-2 ml-4">
                <li>
                  When required by law or to comply with legal obligations
                </li>
                <li>
                  To investigate or prevent fraud, abuse, or security threats
                </li>
                <li>With your consent (e.g., API integrations you enable)</li>
              </ul>
            </section>

            <section className="mb-8">
              <h2 className="text-2xl font-bold text-gray-900 mb-4">
                6. Third-Party Services
              </h2>
              <p className="text-gray-700 mb-4">
                Our platform may integrate third-party services (e.g.,
                fact-check APIs or maps). These providers have their own privacy
                policies, and we recommend reviewing them.
              </p>
              <p className="text-gray-700">
                We do not control how third-party services process your data.
              </p>
            </section>

            <section className="mb-8">
              <h2 className="text-2xl font-bold text-gray-900 mb-4">
                7. Your Rights
              </h2>
              <p className="text-gray-700 mb-4">You may:</p>
              <ul className="list-disc list-inside text-gray-700 space-y-2 ml-4 mb-4">
                <li>Request access to your personal data</li>
                <li>Request deletion of your account or uploads</li>
                <li>Opt-out of email communication</li>
                <li>Withdraw consent (where applicable)</li>
              </ul>
            </section>

            <section className="mb-8">
              <h2 className="text-2xl font-bold text-gray-900 mb-4">
                8. Children's Privacy
              </h2>
              <p className="text-gray-700">
                SafeguardMedia is not intended for children under 13. We do not
                knowingly collect data from children. If you believe we've
                collected such information, please contact us and we will delete
                it.
              </p>
            </section>

            <section className="mb-8">
              <h2 className="text-2xl font-bold text-gray-900 mb-4">
                9. International Users
              </h2>
              <p className="text-gray-700">
                If you are located outside the United States, note that your
                data may be processed and stored in the U.S. and governed by
                U.S. law. By using our Service, you consent to this transfer and
                processing.
              </p>
            </section>

            <section className="mb-8">
              <h2 className="text-2xl font-bold text-gray-900 mb-4">
                10. Changes to This Policy
              </h2>
              <p className="text-gray-700">
                We may update this Privacy Policy from time to time. If material
                changes are made, we will notify users via the website or email.
              </p>
            </section>
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

            {/* Legal Column */}
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

export default PrivacyPolicy;
