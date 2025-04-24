import express from 'express';
import dotenv from 'dotenv';
import { detectHandler } from './routes/detect.js';
// Load correct env file
dotenv.config({ path: `.env.${process.env.ENV || 'development'}` });
const app = express();
const port = process.env.PORT || 8080;
app.use(express.json());
app.post('/detect', detectHandler);
app.listen(port, () => {
    console.log(`Server running on port ${port}`);
});
