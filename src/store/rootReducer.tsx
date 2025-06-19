import { combineReducers } from "@reduxjs/toolkit";
import userReducer from "./slices/userSlices";
import authReducer from "./slices/authSlices";

const rootReducer = combineReducers({
  user: userReducer,
  auth: authReducer,

  // Add other reducers here
});

export default rootReducer;
