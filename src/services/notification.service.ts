import { Notification } from '../models/Notification.js';
// import { io } from '../sockets/socket';

export const notifyUser = async (
  userId: string,
  data: {
    type: string;
    title: string;
    message: string;
    link?: string;
  }
) => {
  const notification = await Notification.create({ userId, ...data });

  // io.to(userId).emit('notification:new', notification);

  return notification;
};
