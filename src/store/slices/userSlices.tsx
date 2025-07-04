import { createSlice } from "@reduxjs/toolkit";
import type { PayloadAction } from "@reduxjs/toolkit";

export interface User {
  _id: string;
  email: string;
  userType: string;
  plan: string;
  isGoogleUser: boolean;
  firstName?: string;
  lastName?: string;
}

interface UserState {
  idToken: string;
  user: User;
  email: string;
  pin: string;
}

const createInitialUser = (): User => ({
  _id: "",
  email: "",
  userType: "",
  plan: "",
  isGoogleUser: false,
  firstName: "",
  lastName: "",
});

const initialState: UserState = {
  idToken: "",
  user: createInitialUser(),
  email: "",
  pin: "",
};

const userSlice = createSlice({
  name: "user",
  initialState,
  reducers: {
    setUserInfo: (state, action: PayloadAction<User>) => {
      state.user = { ...createInitialUser(), ...action.payload };
    },
    setEmail: (state, action: PayloadAction<string>) => {
      state.email = action.payload;
    },

    clearUserInfo: (state) => {
      return {
        ...initialState,
        state,
      };
    },
  },
});

export const { setUserInfo, clearUserInfo, setEmail } = userSlice.actions;
export default userSlice.reducer;
