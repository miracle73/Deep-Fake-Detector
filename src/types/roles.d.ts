export enum UserRole {
  INDIVIDUAL = 'individual', // Free/Pro/Max tier users
  ENTERPRISE_ADMIN = 'enterprise_admin', // Company admins
  ENTERPRISE_MEMBER = 'enterprise_member', // Team members
  MODERATOR = 'moderator', // (Future) For content moderation
  SUPER_ADMIN = 'super_admin', // internal team
}

export type AccountType = 'individual' | 'enterprise';
