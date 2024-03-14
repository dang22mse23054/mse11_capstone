// load environment variables
require('dotenv').config();

const { Agent } = require('https');
const axios = require('axios');
const jwt = require('jsonwebtoken');
const jwtDecode = require('jwt-decode');
const LogService = require('commonDir/logger');
const log = LogService.getInstance();
const { userService } = require('apiDir/services');

module.exports = class AuthService {

	signWebApiToken(profile, remoteIp, tokenExpHours = process.env.API_SESSION_TIMEOUT /* millisenconds */) {
		let expTime = new Date();
		expTime.setTime(expTime.getTime() + Number(tokenExpHours));

		return jwt.sign(
			{
				iss: process.env.JWT_ISSUER,
				sub: profile.uid,
				name: profile.name,
				aud: remoteIp,
				iat: parseInt(new Date().getTime() / 1000),
				exp: parseInt(expTime.getTime() / 1000)
			},
			process.env.JWT_SECRET
		);
	}

	verifyToken(token) {
		try {
			return jwt.verify(token, process.env.JWT_SECRET);
		} catch (error) {
			log.error(error, true);
			return null;
		}
	}

	async regenerateApiToken (decodedAlbToken, req) {
		// Gen apiToken after login
		let clientIp = req.headers['x-forwarded-for'] || req.connection.remoteAddress;

		const uid = decodedAlbToken[process.env.AUTH_EMPLOYEE_ID];
		const user = await userService.getUserByUid(uid);

		let profile = {
			uid,
			name: user.fullname,
			roleId: user.roleId,
		};
		log.debug(JSON.stringify(profile));
		log.debug(`User ${profile.name} (${profile.uid}) LOGIN from address ${clientIp} at ${new Date()}`);

		// update last access
		userService.updateLastAccess(uid);

		return {
			userInfo: profile,
			apiToken: this.signWebApiToken(profile, clientIp),
			accessToken: req.headers['x-amzn-oidc-accesstoken']
		};
	}

	parseToken(token) {
		try {
			if (token) {
				return jwtDecode(token);
			}
		} catch (error) {
			log.error(error);
		}
		return null;
	}

	logoutCAsso(req) {
		if (req.user) {
			// Disable SSL certificate error of target server
			const agent = new Agent({
				rejectUnauthorized: false
			});

			const postParams = {
				token: req.user.refreshToken,
				token_type_hint: 'refresh_token'
			};

			const postConfigs = {
				method: 'post',
				url: `${process.env.REVOKE_TOKEN_URL}`,
				httpsAgent: agent,
				headers: {
					'Content-Type': 'application/x-www-form-urlencoded'
				},
				params: postParams,
				auth: {
					username: process.env.CLIENT_ID,
					password: process.env.CLIENT_SECRET
				}
			};

			// Request to revoke CASSO token
			return axios(postConfigs)
				.then(function (respObj) {
					req.logout();
					req.session.destroy();
					return respObj;
				}).catch(function (error) {
					log.error(error.response.data);
					return false;
				});
		}

		return true;
	}

	logoutUId(req) {
		req.logout();
		req.session.destroy();
	}
};