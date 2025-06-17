export const generatePaymentReceipt = ({
  amount,
  date,
  invoiceUrl,
}: {
  amount: string;
  date: string;
  invoiceUrl: string;
}) => {
  return `<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Payment Receipt</title>
    <style>
      body {
        font-family: 'Helvetica Neue', sans-serif;
        background: #f9f9f9;
        padding: 2rem;
        color: #333;
      }
      .container {
        max-width: 600px;
        margin: auto;
        background: white;
        padding: 2rem;
        border-radius: 8px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
      }
      .header {
        text-align: center;
        margin-bottom: 2rem;
      }
      .button {
        display: inline-block;
        padding: 10px 20px;
        margin-top: 1.5rem;
        background-color: #635bff;
        color: white;
        text-decoration: none;
        border-radius: 5px;
      }
      .footer {
        font-size: 0.8rem;
        color: #888;
        text-align: center;
        margin-top: 2rem;
      }
    </style>
  </head>
  <body>
    <div class="container">
      <div class="header">
        <h2>Thank you for your payment!</h2>
      </div>
      <p>Hi {{userName}},</p>
      <p>Your payment of <strong>${{
        amount,
      }}</strong> was successful on ${date}.</p>
      <p>You can view or download your receipt below:</p>
      <a class="button" href="${invoiceUrl}" target="_blank">View Receipt</a>
      <div class="footer">
        &copy; ${new Date().getFullYear()} Your Company. All rights reserved.
      </div>
    </div>
  </body>
</html>
`;
};

export const generateUpcomingInvoice = ({
  firstName,
  amount,
  plan,
  renewDate,
}: {
  firstName: string;
  amount: string;
  plan: string;
  renewDate: string;
}) => {
  return `<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Upcoming Payment</title>
    <style>
      body {
        font-family: 'Helvetica Neue', sans-serif;
        background: #f9f9f9;
        padding: 2rem;
        color: #333;
      }
      .container {
        max-width: 600px;
        margin: auto;
        background: white;
        padding: 2rem;
        border-radius: 8px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
      }
      .header {
        text-align: center;
        margin-bottom: 2rem;
      }
      .footer {
        font-size: 0.8rem;
        color: #888;
        text-align: center;
        margin-top: 2rem;
      }
    </style>
  </head>
  <body>
    <div class="container">
      <div class="header">
        <h2>Heads up: Upcoming Payment</h2>
      </div>
      <p>Hi ${firstName},</p>
      <p>This is a reminder that your <strong>${plan}</strong> subscription will renew on <strong>${renewDate}</strong>.</p>
      <p>The amount due is <strong>${amount}</strong>.</p>
      <p>No action is needed if you wish to continue with your plan. To update your billing information or cancel, please visit your account settings.</p>
      <div class="footer">
        &copy; ${new Date().getFullYear()} Your Company. All rights reserved.
      </div>
    </div>
  </body>
</html>
`;
};

export const generateFailedPaymentEmail = ({
  amount,
  date,
  billingUrl,
}: {
  amount: string;
  date: string;
  billingUrl: string;
}) => {
  return `
    <html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Payment Failed</title>
    <style>
      body {
        font-family: 'Helvetica Neue', sans-serif;
        background: #f9f9f9;
        padding: 2rem;
        color: #333;
      }
      .container {
        max-width: 600px;
        margin: auto;
        background: white;
        padding: 2rem;
        border-radius: 8px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
      }
      .header {
        text-align: center;
        margin-bottom: 2rem;
      }
      .button {
        display: inline-block;
        padding: 10px 20px;
        margin-top: 1.5rem;
        background-color: #ff3b30;
        color: white;
        text-decoration: none;
        border-radius: 5px;
      }
      .footer {
        font-size: 0.8rem;
        color: #888;
        text-align: center;
        margin-top: 2rem;
      }
    </style>
  </head>
  <body>
    <div class="container">
      <div class="header">
        <h2>Payment Failed</h2>
      </div>
      <p>Hi {{userName}},</p>
      <p>We tried to process your payment of <strong>${amount}</strong> on ${date}, but it failed.</p>
      <p>Please update your billing information to avoid interruption of service.</p>
      <a class="button" href="${billingUrl}" target="_blank">Update Payment Info</a>
      <div class="footer">
        &copy; ${new Date().getFullYear()} Your Company. All rights reserved.
      </div>
    </div>
  </body>
</html>
`;
};

export const upcomingInvoiceTemplate = ({
  plan,
  amount,
  chargeDate,
}: {
  plan: string;
  amount: number;
  chargeDate: string;
}) => `
  <div style="font-family: Arial, sans-serif; max-width: 600px; margin: auto; padding: 20px;">
    <h2 style="color: #333;">Upcoming Subscription Payment</h2>
    <p>Hi there,</p>
    <p>This is a reminder that your <strong>${plan}</strong> subscription will renew soon.</p>
    <p><strong>Amount:</strong> $${amount}</p>
    <p><strong>Charge Date:</strong> ${chargeDate}</p>
    <p>If you need to update your payment method or cancel, please visit your dashboard.</p>
    <br />
    <p>Thanks,</p>
    <p><strong>Your App Team</strong></p>
  </div>
`;

interface WelcomeEmail {
  name: string;
  appUrl?: string;
  supportUrl?: string;
  termsUrl?: string;
  privacyUrl?: string;
}

export const generateWelcomeEmail = ({
  name,
  appUrl = 'https://google.com',
  supportUrl = 'mailto:info@safeguardmedia.io',
  termsUrl = 'https://google.com',
  privacyUrl = 'https://google.com',
}: WelcomeEmail) => {
  return `<!DOCTYPE html>
<html lang="en" style="margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Welcome to SafeGuard Media</title>
  </head>
  <body style="margin: 0; padding: 0; background-color: #f5f7fa; color: #333;">
    <table cellpadding="0" cellspacing="0" width="100%" style="max-width: 600px; margin: 40px auto; background-color: #ffffff; border-radius: 12px; overflow: hidden; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);">
      <tr>
        <td style="padding: 32px 32px 16px; text-align: center; background-color: #1a73e8; color: #fff;">
          <h1 style="margin: 0; font-size: 24px;">Welcome to SafeGuard Media 👋</h1>
        </td>
      </tr>

      <tr>
        <td style="padding: 24px 32px;">
          <p style="font-size: 16px; margin: 0 0 16px;">Hey ${name},</p>

          <p style="font-size: 16px; margin: 0 0 16px;">
            Thanks for signing up! We're pumped to have you onboard. SafeGuard Media helps you stay ahead by analyzing videos and spotting fakes in seconds.
          </p>

          <p style="font-size: 16px; margin: 0 0 16px;">
            Whether you're using it for personal safety, journalism, or content integrity — you’re in good company. 🛡️
          </p>

          <div style="margin: 24px 0; text-align: center;">
            <a href="${appUrl}" style="background-color: #1a73e8; color: #fff; text-decoration: none; padding: 12px 24px; border-radius: 6px; font-size: 16px; display: inline-block;">Get Started</a>
          </div>

          <p style="font-size: 14px; color: #555; margin: 0 0 8px;">
            Need help getting started? Our team is just an email away. Let’s do great things together!
          </p>

          <p style="font-size: 14px; color: #555; margin: 0;">— The SafeGuard Media Team</p>
        </td>
      </tr>

      <tr>
        <td style="padding: 16px 32px; text-align: center; font-size: 12px; color: #999; background-color: #f0f2f5;">
          <p style="margin: 0;">© ${new Date().getFullYear()} SafeGuard Media. All rights reserved.</p>
          <p style="margin: 4px 0;">
            <a href="${privacyUrl}" style="color: #999; text-decoration: none;">Privacy</a> ·
            <a href="${termsUrl}" style="color: #999; text-decoration: none;">Terms</a> ·
            <a href="${supportUrl}" style="color: #999; text-decoration: none;">Support</a>
          </p>
        </td>
      </tr>
    </table>
  </body>
</html>
`;
};
