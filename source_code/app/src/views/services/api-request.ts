import axios from 'axios';
import moment from 'moment-timezone';
import { AuthService } from '../services';

const DEFAULT_RETRY_TIME = 3;

axios.interceptors.response.use(
	(response) => response,
	(error) => {
		// we can't seem to catch the 302 status code as an error,
		// however, since it redirects to another domain (https://casso.services.isca.jp/as/authorization.oauth2?) it causes
		// a CORS error which makes error.response be undefined here.  This assumes that any time
		// error.response is undefined that we need to redirect to the login page
		if (typeof error.response === 'undefined') {
			alert('Your session has been expired');
			window.location = '/';
		}
		return Promise.reject(error);
	}
);

const ApiRequest = {

	sendGET: (url, params = {}, callback = (res): any => true, errorCallback = (err): any => false, opt = { retryTime: DEFAULT_RETRY_TIME }) => {
		return axios.get(url, {
			params: params,
			headers: {
				...AuthService.getAuthReqHeader()
			}
		}).then(callback).catch((error) => {
			console.error(error);
			if (error.messasge) {
				// console.log(error.messasge);
			} else if (error.response) {
				switch (error.response.status) {
					case 401:
						AuthService.reAuthorize((retryTime) => ApiRequest.sendGET(url, params, callback, errorCallback, { retryTime }), {
							retryTime: opt.retryTime
						});
						break;
					case 403:
						window.location.href = '/error/403';
						break;
					case 423:
						window.location.href = '/error/423';
						break;
					default:
						errorCallback(error);
						break;
				}
			}
		});
	},

	sendPOST: (url, params = {}, callback = (res): any => true, errorCallback = (error): any => false, opt = { retryTime: DEFAULT_RETRY_TIME }) => {
		return axios.post(url, params, {
			headers: {
				...AuthService.getAuthReqHeader()
			}
		}).then(callback).catch((error) => {
			console.error(error);
			if (error.messasge) {
				// console.log(error.messasge);
			} else if (error.response) {
				switch (error.response.status) {
					case 401:
						AuthService.reAuthorize((retryTime) => ApiRequest.sendPOST(url, params, callback, errorCallback, { retryTime }), {
							retryTime: opt.retryTime
						});
						break;
					case 403:
						window.location.href = '/error/403';
						break;
					case 423:
						window.location.href = '/error/423';
						break;
					default:
						errorCallback(error);
						break;
				}
			}
		});
	},

	sendDELETE: (url, params = {}, callback = (res): any => true, errorCallback = (error): any => false, opt = { retryTime: DEFAULT_RETRY_TIME }) => {
		axios.delete(url, {
			headers: {
				...AuthService.getAuthReqHeader()
			},
			data: params
		}).then(callback).catch((error) => {
			console.error(error);
			if (error.messasge) {
				// console.log(error.messasge);
			} else if (error.response) {
				switch (error.response.status) {
					case 401:
						AuthService.reAuthorize((retryTime) => ApiRequest.sendDELETE(url, params, callback, errorCallback, { retryTime }), {
							retryTime: opt.retryTime
						});
						break;
					case 403:
						window.location.href = '/error/403';
						break;
					case 423:
						window.location.href = '/error/423';
						break;
					default:
						errorCallback(error);
						break;
				}
			}
		});
	},

	uploadFile: (url, file, extParams = {}, callback = (res): any => true, errorCallback = (error): any => false, opt = { retryTime: DEFAULT_RETRY_TIME }) => {
		// Prepare form data
		const formData = new FormData();
		formData.append('uploadedFile', file);
		formData.append('uploadedTime',  moment().toISOString());

		if (extParams) {
			Object.keys(extParams).forEach((key) => {
				formData.append(key, extParams[key]);
			});
		}

		return axios.post(url, formData, {
			headers: {
				...AuthService.getAuthReqHeader(),
				'Content-Type': 'multipart/form-data'
			}
		}).then(callback).catch((error) => {
			console.error(error);
			if (error.messasge) {
				// console.log(error.messasge);
			} else if (error.response) {
				switch (error.response.status) {
					case 401:
						AuthService.reAuthorize((retryTime) => ApiRequest.sendPOST(url, params, callback, errorCallback, { retryTime }), {
							retryTime: opt.retryTime
						});
						break;
					case 403:
						window.location.href = '/error/403';
						break;
					case 423:
						window.location.href = '/error/423';
						break;
					default:
						errorCallback(error);
						break;
				}
			}
		});
	},

	// donwloadFile: (url, params = {}, errorCallback = (error): any => false, opt = { retryTime: DEFAULT_RETRY_TIME }) => {
	// 	return axios.get(url, {
	// 		params,
	// 		responseType: 'blob',
	// 		headers: {
	// 			...AuthService.getAuthReqHeader()
	// 		}
	// 	}).then((response) => {
	// 		let blob = response.data
	// 		let reader = new FileReader()
	// 		reader.readAsDataURL(blob)
	// 		reader.onload = (e) => {
	// 			let a = document.createElement('a')
	// 			a.download = `fileName.csv`
	// 			a.href = e.target.result
	// 			document.body.appendChild(a)
	// 			a.click()
	// 			document.body.removeChild(a)
	// 		}


	// 		// const url = window.URL.createObjectURL(new Blob([response.data]));
	// 		// const link = document.createElement('a');
	// 		// link.href = url;
	// 		// link.setAttribute('download', 'file.pdf'); //or any other extension
	// 		// document.body.appendChild(link);
	// 		// link.click();
	// 	}).catch((error) => {
	// 		console.error(error);
	// 		if (error.messasge) {
	// 			// console.log(error.messasge);
	// 		} else if (error.response) {
	// 			switch (error.response.status) {
	// 				case 401:
	// 					AuthService.reAuthorize((retryTime) => ApiRequest.sendGET(url, params, callback, errorCallback, { retryTime }), {
	// 						retryTime: opt.retryTime
	// 					});
	// 					break;
	// 				case 403:
	// 					window.location.href = '/error/403';
	// 					break;
	// 				case 423:
	// 					window.location.href = '/error/423';
	// 					break;
	// 				default:
	// 					errorCallback(error);
	// 					break;
	// 			}
	// 		}
	// 	});
	// }
};

export default ApiRequest;