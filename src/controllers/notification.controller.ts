import type { Request, Response, NextFunction } from 'express';
import { Notification } from '../models/Notification.js';

export const getNotifications = async (req: Request, res: Response) => {
  const userId = req.user?.id;
  const notifications = await Notification.find({ userId }).sort({
    createdAt: -1,
  });
  res
    .status(20)
    .json({ success: true, message: 'Notification fetched', notifications });
};

export const markAsRead = async (req: Request, res: Response) => {
  const { id } = req.params;
  const userId = req.user?.id;

  const notif = await Notification.findOne({ _id: id, userId });
  if (!notif) {
    res.status(404).json({ message: 'Not found' });
  } else {
    notif.read = true;
    await notif.save();

    res
      .status(200)
      .json({ success: true, message: 'Notification marked as read' });
  }
};
