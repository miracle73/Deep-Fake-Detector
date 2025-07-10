import {
  getNotifications,
  markAsRead,
} from 'controllers/notification.controller';
import express from 'express';

import { protect } from 'middlewares/auth';

const router = express.Router();

router.use(protect);
router.get('/', getNotifications);
router.patch('/:id/read', markAsRead);

export default router;
