import { config } from 'dotenv';

config();

export const swaggerOptions = {
  definition: {
    openapi: '3.0.0',
    info: {
      title: 'Safeguard API Documentation',
      version: '1.0.0',
      description: 'API documentation for the Safeguard',
    },
    servers: [
      {
        url: `http://localhost:${process.env.PORT || 8080}`,
        description: 'Development server',
      },
    ],
  },
  apis: ['./src/routes/*.ts'],
};
