import { createSlice } from "@reduxjs/toolkit";
import type { PayloadAction } from "@reduxjs/toolkit";

export interface AccountDetails {
  bankName: string;
  accountName: string;
  accountType: string;
  accountNumber: string;
  accountBalance: string;
  status: string;
  _id: string;
}

export interface User {
  _id: string;
  email: string;
  role: string;
  accountBalance: number;
  hasSetTransactionPin: boolean;
  isVerified: boolean;
  status: string;
  isGoogleUser: boolean;
  firstName: string;
  lastName: string;
  accountNumber: string;
  phoneNumber: string;
  accountDetails: AccountDetails;
  imageUrl: string;
  thumbnailUrl: string;
  originalImageUrl: string;
  isKYCVerified: boolean;
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
  role: "",
  accountBalance: 0,
  hasSetTransactionPin: false,
  isVerified: false,
  status: "",
  isGoogleUser: false,
  firstName: "",
  lastName: "",
  phoneNumber: "",
  accountNumber: "",
  accountDetails: {
    bankName: "",
    accountName: "",
    accountNumber: "",
    accountType: "",
    accountBalance: "",
    status: "",
    _id: "",
  },
  imageUrl: "",
  thumbnailUrl: "",
  originalImageUrl: "",
  isKYCVerified: false,
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
    setPin: (state, action: PayloadAction<string>) => {
      state.pin = action.payload;
    },
    setStatus: (state, action: PayloadAction<string>) => {
      state.user.accountDetails.status = action.payload;
    },
    setImageUrl: (state, action: PayloadAction<string>) => {
      state.user.imageUrl = action.payload;
    },
    setThumbnailUrl: (state, action: PayloadAction<string>) => {
      state.user.thumbnailUrl = action.payload;
    },

    clearUserInfo: (state) => {
      const currentPin = state.pin;
      return {
        ...initialState,
        pin: currentPin,
      };
    },
  },
});

export const {
  setUserInfo,
  clearUserInfo,
  setEmail,
  setStatus,
  setPin,
  setImageUrl,
  setThumbnailUrl,
} = userSlice.actions;
export default userSlice.reducer;
