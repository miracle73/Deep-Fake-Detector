import logger from '../../utils/logger';
import { api, cleanup, initializeDB } from '../setup';

beforeAll(async () => {
  await initializeDB();
});

afterAll(async () => {
  await cleanup();
});

describe('Auth Tests', () => {
  describe('Login', () => {
    describe('given email or password is not provided', () => {
      it('should return a 400 error', async () => {
        const loginData = {
          email: '',
          password: '',
        };

        const response = await api
          .post('/api/v1/auth/login')
          .expect(400)
          .send(loginData);

        // logger.info(
        //   `Login test response: ${JSON.stringify(response.body, null, 2)}`
        // );
      });
    });

    describe('given the user does not exist', () => {
      it('should return a 404 error', async () => {
        const loginData = {
          email: 'hello@johndoe.com',
          password: 'password',
        };

        const result = await api
          .post('/api/v1/auth/login')
          .expect(404)
          .send(loginData);

        // logger.info(
        //   `Login test result: ${JSON.stringify(result.body, null, 2)}`
        // );
      });
    });

    describe('given password is not correct', () => {
      it('should return a 401 error', async () => {
        const loginData = {
          email: 'hello@janedoe.com',
          password: 'wrongpassword',
        };

        const result = await api
          .post('/api/v1/auth/login')
          .expect(401)
          .send(loginData);

        // logger.info(
        //   `Login test result: ${JSON.stringify(result.body, null, 2)}`
        // );
      });
    });

    describe('given the user is logged in successfully', () => {
      it('should return a 200 status code', async () => {
        const loginData = {
          email: 'hello@janedoe.com',
          password: 'password',
        };

        const response = await api
          .post('/api/v1/auth/login')
          .expect(200)
          .send(loginData);

        // logger.info(`
        //   Login test result: ${JSON.stringify(response.body, null, 2)}`);
      });
    });
  });

  describe('Register', () => {
    describe('given terms are not accepted', () => {
      it('should return a 400 error', async () => {
        const registerData = {
          email: 'hello@johndoe.com',
          password: 'password',
          agreedToTerms: false,
          userType: 'individual',
          firstName: 'John',
          lastName: 'Doe',
          termsAgreedAt: new Date(),
        };

        const response = await api
          .post('/api/v1/auth/register')
          .expect(400)
          .send(registerData);

        // logger.info(
        //   `Register test response: ${JSON.stringify(response.body, null, 2)}`
        // );
      });
    });

    describe('given the user already exists', () => {
      it('should return a 400 error', async () => {
        const registerData = {
          email: 'hello@janedoe.com',
          password: 'password',
          agreedToTerms: true,
          userType: 'individual',
          firstName: 'Jane',
          lastName: 'Doe',
          termsAgreedAt: new Date(),
        };

        const result = await api
          .post('/api/v1/auth/register')
          .expect(400)
          .send(registerData);

        // logger.info(
        //   `Register test result: ${JSON.stringify(result.body, null, 2)}`
        // );
      });
    });

    describe('given the user is registered successfully', () => {
      it('should return a 201 status code', async () => {
        const registerData = {
          email: 'finzyphinzy@gmail.com',
          password: 'password',
          agreedToTerms: true,
          userType: 'individual',
          firstName: 'John',
          lastName: 'Doe',
          termsAgreedAt: new Date(),
        };

        const result = await api
          .post('/api/v1/auth/register')
          .expect(201)
          .send(registerData);

        // logger.info(
        //   `Register test result: ${JSON.stringify(result.body, null, 2)}`
        // );
      });
    });
  });

  describe('Forgot Password', () => {
    describe('given the user does not exist', () => {
      it('should return a 404 error', async () => {
        const forgotPasswordData = {
          email: 'hello@jacksmith.com',
        };

        const result = await api
          .post('/api/v1/auth/forgot-password')
          .expect(404)
          .send(forgotPasswordData);

        // logger.info(
        //   `Forgot Password test result: ${JSON.stringify(result.body, null, 2)}`
        // );
      });
    });

    describe('given the user exists', () => {
      it('should return a 200 status code', async () => {
        const forgotPasswordData = {
          email: 'hello@janedoe.com',
        };

        const response = await api
          .post('/api/v1/auth/forgot-password')
          .expect(200)
          .send(forgotPasswordData);

        // logger.info(
        //   `Forgot Password test response: ${JSON.stringify(
        //     response.body,
        //     null,
        //     2
        //   )}`
        // );
      });
    });
  });
});
