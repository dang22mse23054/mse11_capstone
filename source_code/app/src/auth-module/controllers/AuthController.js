const BaseResponse = require('../../api/dto/BaseResponse');
const AuthService = require('../../auth-module/services/AuthService');
const { LoginTypes } = require('commonDir/constants');
const NEXT_PUBLIC_NODE_ENV = process.env.NEXT_PUBLIC_NODE_ENV || 'production';
const { userService } = require('apiDir/services');

module.exports = class AuthController {

	async checkSession(req, res, next) {
		let respObj = new BaseResponse();
		respObj.setData({ redirectUrl: NEXT_PUBLIC_NODE_ENV !== 'production' ? '/login' : '/auth/casso' });

		let authService = new AuthService();

		// Production is using OIDC => get token from req.headers['x-amzn-oidc-data'] (no Bearer)
		// Others: get from req.headers.authorization (with Bearer)
		let albToken = req.headers['x-amzn-oidc-data'];
		let apiToken = req.headers.authorization;
		let decodedAlbToken = authService.parseToken(albToken);

		//---- For debugging ----//
		// console.log('-----checkSession-----')
		// console.log('req.headers.authorization')
		// console.log(req.headers.authorization)
		// console.log('x-amzn-oidc-data')
		// console.log(req.headers['x-amzn-oidc-data'])
		// console.log('x-forwarded-for')
		// console.log(req.headers['x-forwarded-for']);
		// console.log('-----checkSession-----end')

		if (!apiToken) {
			if (NEXT_PUBLIC_NODE_ENV === 'production') {
				respObj.setData(await authService.regenerateApiToken(decodedAlbToken, req));
				return res.json(respObj);
			}
			console.log('Token NOT FOUND');
			respObj.setError(401, 'Token NOT FOUND');
			return res.status(401).json(respObj);
		}

		// verify and decode apiToken (Token format: Bearer <token_here>)
		if (req.headers.authorization) {
			apiToken = apiToken.split('Bearer')[1].trim();
		}

		let decodedToken = authService.verifyToken(apiToken);

		if (!decodedToken) {
			if (NEXT_PUBLIC_NODE_ENV === 'production') {
				respObj.setData(await authService.regenerateApiToken(decodedAlbToken, req));
				return res.json(respObj);
			}
			console.log('Invalid token');
			respObj.setError(401, 'Invalid token');
			return res.status(401).json(respObj);
		}

		// --- Validation --- //
		let currentTime = new Date();
		let expiredTime = new Date(decodedToken.exp * 1000);

		let isExpired = currentTime >= expiredTime;
		if (isExpired) {
			if (NEXT_PUBLIC_NODE_ENV === 'production') {
				respObj.setData(await authService.regenerateApiToken(decodedAlbToken, req));
				return res.json(respObj);
			}
			console.log('Token expired');
			respObj.setError(401, 'Token expired');
			return res.status(401).json(respObj);
		}

		// update last access
		const uid = decodedToken.sub;
		userService.updateLastAccess(uid);

		respObj.setData({
			userInfo: {
				uid,
				name: decodedToken.name,
				roleId: decodedToken.roleId,
			},
			apiToken
		});

		return res.json(respObj);
	}

	async logout(req, res) {
		let authService = new AuthService();

		let loginType = req.session.loginType;

		switch (loginType) {
			case LoginTypes.LOCAL:
				await authService.logoutUId(req);
				break;
			case LoginTypes.CASSO:
				await authService.logoutCAsso(req);
				break;
			default:
				break;
		}

		res.redirect('/');
	}
};