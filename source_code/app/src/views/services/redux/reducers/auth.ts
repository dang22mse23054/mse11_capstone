import { ActionTypes } from '../actions';
import { IAuthActionObj } from '../actions/action-object';

const isServerSide = typeof window === 'undefined';

const initialState = {
	userInfo: null,
	apiToken: 'unknown',
	accessToken: isServerSide ? null : localStorage.getItem('accessToken'),
	refreshToken: isServerSide ? null : localStorage.getItem('refreshToken'),
	isFirstLoading: true
};

const AuthReducer = (state = initialState, action: IAuthActionObj) => {
	switch (action.type) {
		case ActionTypes.Auth.SET_AUTH_DATA:
			return setAuthData(state, action.authData);
		case ActionTypes.Auth.LOG_OUT:
			return logout(state);
		default:
			return state;
	}
};
export default AuthReducer;

const setAuthData = (state, authData) => {
	return {
		...state,
		userInfo: {
			...authData.userInfo
		},
		apiToken: authData.apiToken || initialState.apiToken,
		accessToken: authData.accessToken || state.accessToken,
		refreshToken: authData.refreshToken || state.refreshToken,
		isFirstLoading: false
	};
};

const logout = (state) => {
	return {
		...initialState,
		accessToken: null,
		refreshToken: null,
	};
};
