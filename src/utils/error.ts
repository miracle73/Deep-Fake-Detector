export class AppError extends Error {
  constructor(
    public statusCode: number,
    public message: string,
    public details?: any
  ) {
    super(message);
    Object.setPrototypeOf(this, AppError.prototype);
    Error.captureStackTrace(this, this.constructor);
  }

  toJSON() {
    const isProduction = process.env.NODE_ENV === 'production';
    return {
      success: false,
      code: this.statusCode,
      message: this.message,
      ...(this.details && { details: this.details }),
      ...(!isProduction && { stack: this.stack }),
    };
  }
}

export class ValidationError extends AppError {
  constructor(message: string, details: any = null) {
    super(400, message, details);
  }

  toJSON() {
    const base = super.toJSON();
    return {
      ...base,
      message: base.message || 'Validation failed',
    };
  }
}

export class AuthenticationError extends AppError {
  constructor(message: string = 'Authentication failed', details: any = null) {
    super(401, message, details);
  }

  toJSON() {
    const base = super.toJSON();
    return {
      ...base,
      message: base.message || 'Authentication failed',
    };
  }
}

export class AuthorizationError extends AppError {
  constructor(
    message: string = 'Not authorized to access this resource',
    details: any = null
  ) {
    super(403, message, details);
  }

  toJSON() {
    const base = super.toJSON();
    return {
      ...base,
      message: base.message || 'Not authorized to access this resource',
    };
  }
}

export class NotFoundError extends AppError {
  constructor(message: string = 'Resource not found', details: any = null) {
    super(404, message, details);
  }

  toJSON() {
    const base = super.toJSON();
    return {
      ...base,
      message: base.message || 'Resource not found',
    };
  }
}

export class ConflictError extends AppError {
  constructor(message: string, details: any = null) {
    super(409, message, details);
  }

  toJSON() {
    const base = super.toJSON();
    return {
      ...base,
      message: base.message || 'Conflict occurred',
    };
  }
}
