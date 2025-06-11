| Event                           | Purpose                               |
| ------------------------------- | ------------------------------------- |
| `customer.subscription.created` | Save new subscription info to DB      |
| `customer.subscription.updated` | Update status, plan, period end, etc. |
| `customer.subscription.deleted` | Mark as canceled + revoke access      |
| `invoice.payment_succeeded`     | (optional) log or notify              |
| `invoice.payment_failed`        | (optional) warn users or lock usage   |
