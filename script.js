import http from 'k6/http';
import { sleep, check } from 'k6';

export const options = {
  vus: 10,
  duration: '30s',
  thresholds: {
    http_req_duration: ['p(95)<500'], // fail test if 95% of reqs > 500ms
  },
  stages: [
    { duration: '10s', target: 20 },
    { duration: '10s', target: 50 },
    { duration: '10s', target: 100 },
  ],
};

export default function () {
  // let res = http.get('http://localhost:8080');

  // check(res, {
  //   '200 OK': (r) => r.status === 200,
  //   '429 Too Many': (r) => r.status === 429,
  //   '500+ Errors': (r) => r.status >= 500,
  // });

  // sleep(1);

  const res = http.post(
    'http://localhost:8080/api/v1/auth/login',
    JSON.stringify({ email: 'finzyphinzy@gmail.com', password: 'password' }),
    {
      headers: { 'Content-Type': 'application/json' },
    }
  );
  check(res, {
    'status is 200': (r) => r.status === 200,
  });
}
