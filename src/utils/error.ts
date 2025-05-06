class ApiError extends Error {
  constructor(
    public code: number,
    public success: boolean,
    public message: string,
    public details?: any
  ) {
    super(message);
  }
}

export default ApiError;
