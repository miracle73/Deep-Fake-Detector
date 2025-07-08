import express from 'express';
import { demoRequestSchema } from 'lib/schemas/demo.schema.js';
import { validateInput } from 'middlewares/validate.js';

import * as DemoRequestController from '../controllers/demoRequest.controller.js';

const demoRequestRouter = express.Router();

demoRequestRouter.post(
  '/',
  validateInput(demoRequestSchema),
  DemoRequestController.Submit
);

export default demoRequestRouter;
