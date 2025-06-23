type PlanName = 'safeguard free' | 'safeguard pro' | 'safeguard max';

export function getPlanConfig(planName: string) {
  const plans: Record<PlanName, { monthlyQuota: number; carryOver: boolean }> =
    {
      'safeguard free': { monthlyQuota: 3, carryOver: false },
      'safeguard pro': { monthlyQuota: 100, carryOver: true },
      'safeguard max': {
        monthlyQuota: Number.POSITIVE_INFINITY,
        carryOver: true,
      },
    };
  const key = planName.toLowerCase() as PlanName;
  return plans[key] || plans['safeguard free'];
}

export function isHigherPlan(currentPlan: string, newPlan: string) {
  const planOrder = ['free', 'pro', 'max'];
  return (
    planOrder.indexOf(newPlan.toLowerCase()) >
    planOrder.indexOf(currentPlan.toLowerCase())
  );
}
