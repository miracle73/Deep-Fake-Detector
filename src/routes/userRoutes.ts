import { Router } from 'express';
const userRouter = Router();

import * as UserController from '../controllers/user.controller.js';
import { protect } from '../middlewares/auth.js';
import { validateInput } from '../middlewares/validate.js';
import { updateUserSchema } from '../lib/schemas/user.schema.js';

userRouter.get('/', protect, UserController.getCurrentUser);

userRouter.get('/analysis-history', protect, UserController.getAnalysisHistory);

userRouter.patch(
  '/update',
  protect,
  validateInput(updateUserSchema),
  UserController.UpdateUser
);

userRouter.delete('/delete', protect, UserController.DeleteUser);

// router.post('/team/invite',
//     requireRole(UserRole.ENTERPRISE_ADMIN),
//     inviteTeamMember
//   );

//   router.get('/team',
//     requireRole(UserRole.ENTERPRISE_ADMIN, UserRole.ENTERPRISE_MEMBER),
//     getTeam
//

export default userRouter;
