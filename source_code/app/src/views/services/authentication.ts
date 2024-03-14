import axios from 'axios';
import store from 'servDir/redux/store';
import { Actions } from 'servDir/redux/actions';
import moment from 'moment';

const Authentication = {
	checkSession: (redirectOnSessionExpired = true) => {
		const response = { error: null as any, authData: null };
		//load data from API
		return axios.get('/auth/check', {
			headers: {
				...Authentication.getAuthReqHeader()
			}
		}).then(async (respObj) => {
			const authData = respObj.data;

			// NOTE: OIDC does not create lastUrl on the first login
			let lastUrl = localStorage.getItem('lastUrl');

			store.dispatch(Actions.AuthAction.setAuthData(authData.data));
				response.authData = authData;

				// get the last url to redirect
				if (lastUrl && lastUrl != '/') {
					localStorage.removeItem('lastUrl');
					window.location = lastUrl;
					response.isStopCallback = true;
				}

				return response;

		}).catch(function (error) {
			if (error.response && error.response.status == 401) {
				let errMsg = 'Unauthenticated';
				const authData = error.response.data;

				// save current page to redirect after login successfully
				const location = window.location;
				localStorage.setItem('lastUrl', `${location.pathname}${location.search}`);

				authData.data = (authData.data || {});
				authData.data.redirectUrl = (authData.data.redirectUrl || '/auth/casso');

				if (!store.getState().authReducer.isFirstLoading) {
					alert('Your session has been expired');
					errMsg = 'Session expired';
				}

				response.error = new Error(errMsg);

				if (redirectOnSessionExpired) {
					window.location = authData.data.redirectUrl;
				}
			}

			return Promise.reject(error);
		});
	},

	verifyAuthCode: (code, redirectOnSessionExpired = true) => {
		const response = { error: null as any, authData: null };
		//load data from API
		return axios.post(`/oauth2/idpresponse?code=${code}`)
			.then(async (respObj) => {
				const authData = respObj.data;

				store.dispatch(Actions.AuthAction.setAuthData(authData));
				response.authData = authData;
				return response;
			})
			.catch(function (error) {
				console.error(error);

				if (error.response && error.response.status == 401) {
					let errMsg = 'Unauthenticated';
					const authData = error.response.data;
					console.error(authData);
					if (authData.data) {
						authData.data.redirectUrl = `${authData.data.redirectUrl || '/'}`;

						if (!store.getState().authReducer.isFirstLoading) {
							alert('Your session has been expired');
							errMsg = 'Session expired';
						}

						response.error = new Error(errMsg);

						if (redirectOnSessionExpired) {
							window.location = authData.data.redirectUrl;
						}
					}
				}

				response.error = error;
				return response;
			});
	},

	loginLocalUser: (uid) => {
		const response = { error: null as any, authData: null };
		//load data from API
		return axios.post('/auth/userid', { userId: uid })
			.then((respObj) => {
				const authData = respObj.data;

				if (authData.hasError) {
					window.location = `/login?code=${authData.code}`;
					return false;
				}

				if (authData) {
					localStorage.setItem('apiToken', authData.apiToken);
					localStorage.setItem('refreshToken', authData.refreshToken);
					localStorage.setItem('accessToken', authData.accessToken);
				}

				window.location = '/';
			})
			.catch(function (error) {
				console.error(error);

				if (error.response && error.response.status == 401) {
					// const errMsg = 'Unauthenticated';
					const authData = error.response.data;
					console.error(authData);
					if (authData.data) {
						authData.data.redirectUrl = `${authData.data.redirectUrl || '/login'}`;
						window.location = authData.data.redirectUrl;
					}
				}

				response.error = error;
				return response;
			});
	},

	reAuthorize: (callback, { retryTime = 0, redirectOnSessionExpired = true, waitTimeBeforeRetry = 2000 /* milliseconds */ }) => {
		Authentication.checkSession(redirectOnSessionExpired)
			.catch(() => {
				if (retryTime-- > 0) {
					return setTimeout(() => callback(retryTime), waitTimeBeforeRetry);
				}
				alert('Your API token has been expired');
			});
	},

	getApiToken: () => {
		return localStorage.getItem('apiToken') || store.getState().authReducer.apiToken;
	},

	getRefreshToken: () => {
		return localStorage.getItem('refreshToken') || store.getState().authReducer.refreshToken;
	},

	getAccessToken: () => {
		return localStorage.getItem('accessToken') || store.getState().authReducer.accessToken;
	},

	getUserInfo: () => {
		return store.getState().authReducer.userInfo;
	},

	getAuthReqHeader: () => {
		const result = {} as any;
		const apiToken = Authentication.getApiToken();
		if (apiToken != 'unknown') {
			result.Authorization = `Bearer ${apiToken}`;
		}
		return result;
	}
};

export default Authentication;