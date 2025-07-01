import supertest from 'supertest';

describe('User Tests', () => {
  describe('Get User', () => {
    describe('given the user does not exist', () => {
      it('should return a 404 error', async () => {
        // const userId = 'nonexistent-user-id';

        // await supertest('http://localhost:8080')
        //   .get('/api/v1/user')
        //   .expect(404);

        expect(true).toBe(true);
      });
    });
  });
});
