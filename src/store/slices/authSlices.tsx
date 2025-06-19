import { createSlice } from "@reduxjs/toolkit";
import type { PayloadAction } from "@reduxjs/toolkit";

interface AuthState {
  isAuthenticated: boolean;
  token: string | null;
  expires_in: string;
  lastActivity: number;
}

const initialState: AuthState = {
  isAuthenticated: false,
  token: null,
  expires_in: "",
  lastActivity: Date.now(),
};

const authSlice = createSlice({
  name: "auth",
  initialState,
  reducers: {
    loginUser: (state, action: PayloadAction<string>) => {
      state.isAuthenticated = true;
      state.token = action.payload;
      state.lastActivity = Date.now();
    },
    logoutUser: (state) => {
      state.isAuthenticated = false;
      state.token = null;
      state.lastActivity = 0;
    },
    updateLastActivity: (state) => {
      state.lastActivity = Date.now();
    },
  },
});

export const { loginUser, logoutUser, updateLastActivity } = authSlice.actions;
export default authSlice.reducer;
