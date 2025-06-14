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
