import type { IUser } from '../types/user.js';

type PlanName = 'safeguard_free' | 'safeguard_pro' | 'safeguard_max';

export function getPlanConfig(planName: string) {
  const plans: Record<PlanName, { monthlyQuota: number; carryOver: boolean }> =
    {
      safeguard_free: { monthlyQuota: 3, carryOver: false },
      safeguard_pro: { monthlyQuota: 30, carryOver: true },
      safeguard_max: {
        monthlyQuota: Number.POSITIVE_INFINITY,
        carryOver: true,
      },
    };
  const key = planName.toLowerCase() as PlanName;
  console.log('key here===', key, plans[key]);
  return plans[key] || plans.safeguard_free;
}

export function isHigherPlan(currentPlan: string, newPlan: string) {
  console.log('Comparing plans:', currentPlan, newPlan);
  const planOrder = ['safeguard_free', 'safeguard_pro', 'safeguard_max'];
  return (
    planOrder.indexOf(newPlan.toLowerCase()) >=
    planOrder.indexOf(currentPlan.toLowerCase())
  );
}

export const checkSubscriptionActive = (user: IUser) => {
  return (
    user.isActive && user.currentPeriodEnd && user.currentPeriodEnd > new Date()
  );
};
