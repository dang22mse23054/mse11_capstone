import { IAuthActionObj } from '../action-object';
import ActionTypes from './types';

export class AuthAction {
	public static setAuthData(authData: any): IAuthActionObj {

		if (authData.apiToken) { localStorage.setItem('apiToken', authData.apiToken); }
		if (authData.refreshToken) { localStorage.setItem('refreshToken', authData.refreshToken); }
		if (authData.accessToken) { localStorage.setItem('accessToken', authData.accessToken); }

		return {
			type: ActionTypes.SET_AUTH_DATA,
			authData,
			isFirstLoading: false
		};
	}

	public static logout(): IAuthActionObj {
		localStorage.clear();

		return {
			type: ActionTypes.LOG_OUT
		};
	}
}

// --------- Redux-thunk --------- //
// Async actions
export const Creator = {

};
