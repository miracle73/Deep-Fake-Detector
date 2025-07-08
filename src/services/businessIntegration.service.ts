import BusinessIntegration from '../models/BusinessIntegration.js';

import type { BusinessIntegrationType } from '../lib/schemas/businessIntegration.schema.js';

export const createBusinessIntegrationRequest = async (
  data: BusinessIntegrationType
) => {
  const existing = await BusinessIntegration.findOne({ email: data.email });
  if (existing) {
    return existing;
  }

  const record = await BusinessIntegration.create(data);
  return record;
};
