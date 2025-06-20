Great question — managing **usage quotas** is crucial for products like your Deepfake Detector, especially when you have free plans and paid subscriptions. Let’s break this down:

---

### ✅ Your Schema Default

```js
usageQuota: {
  type: UsageQuotaSchema,
  required: true,
  default: {
    monthlyAnalysis: 0,
    remainingAnalysis: 0,
    lastReset: Date.now(),
  }
}
```

This is technically fine, but **in production**, it's better to populate it with the **defaults of your free plan** — not just zeros — so a new user gets value instantly.

For example:

```js
default: {
  monthlyAnalysis: 20,
  remainingAnalysis: 20,
  lastReset: Date.now(),
}
```

---

### 💡 How Quotas Are Managed in the Real World

Here’s the real-world flow:

---

#### 1. **Initial Assignment (on Signup)**

- New users are assigned the free plan’s quota at account creation.
- Store both:

  - `monthlyAnalysis` (e.g., limit)
  - `remainingAnalysis` (decrement as user consumes)

---

#### 2. **Quota Enforcement**

- Each time the user runs an analysis:

  - Check if `remainingAnalysis > 0`
  - If yes, allow and decrement.
  - If no, deny or redirect to upgrade.

---

#### 3. **Resetting Free Plan Quotas Monthly**

- Typically done via a **cron job** or **queue worker** that runs daily:

  - For users on a free plan, check `lastReset`
  - If a month has passed, reset:

    ```js
    remainingAnalysis = monthlyAnalysis;
    lastReset = new Date();
    ```

---

#### 4. **Paid Plans via Webhook (e.g., Stripe)**

- When payment is successful or a subscription is created:

  - Webhook triggers a quota assignment:

    ```js
    monthlyAnalysis = plan.limit;
    remainingAnalysis = plan.limit;
    lastReset = new Date();
    ```

- If it’s a **renewal**, same thing happens:

  - You reset `remainingAnalysis` to match `monthlyAnalysis`.

- If they **upgrade** mid-period, options:

  - Either **prorate** and increase quota
  - Or **wait until next billing cycle** to apply full change

---

#### 5. **Edge Cases to Consider**

- **Multiple payments** in one period (accidental or edge cases)

  - Protect from adding quota multiple times.

- **Canceled or downgraded plans**

  - You can let them keep quota until period ends or reduce instantly.

- **Overuse attempts**

  - Some apps allow a small buffer (e.g., 2–3 over the limit).

- **Team-based quotas**

  - If users are part of an org, quota could be shared.

---

### Summary: What You Should Do

- ✅ Populate defaults based on **free plan**, not zeros.
- ✅ Enforce quota checks on each analysis request.
- ✅ Reset **free plans monthly** with a cron job.
- ✅ Update quota on **subscription events via webhooks**.
- ✅ Record the `lastReset` time for accurate tracking.

Let me know if you want to plan out the actual webhook flows or cron job logic.
