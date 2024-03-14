const BaseResponse = require('../../api/dto/BaseResponse');
const AuthService = require('../../auth-module/services/AuthService');
const { reqRestrictService } = require('apiDir/services');
const { userService } = require('apiDir/services');
const { Common } = require('commonDir/constants');
const LogService = require('commonDir/logger');
const log = LogService.getInstance();
const Sentry = require('commonDir/utils/Sentry');

class Authorization {
	constructor(passport) {
		this.passport = passport;
	}

	verifyCAssoCode = (req, res, next) => {
		this.passport.authenticate('CAssoStrategy', function (err, profile, info) {
			if (err) {
				console.error(err);
				err.message = 'CASSO Unauthorized!';
				err.status = 401;
				err.data = { redirectUrl: '/login' };
				return next(err);
			}
			if (!profile) {
				let errCode = info.code > 10000 ? `?code=${info.code}` : '';
				return res.redirect(`/auth${errCode}`);
			}

			let clientIp = req.headers['x-forwarded-for'] || req.connection.remoteAddress;
			let authService = new AuthService();

			// For DEBUG
			log.debug(JSON.stringify(profile));
			log.debug(`User ${profile.name} (${profile.uid}) LOGIN from address ${clientIp} at ${new Date()}`);

			res.setHeader('Content-Type', 'application/json');
			let respData = {
				userInfo: {
					uid: profile.uid,
					email: profile.email,
					name: profile.name,
					company: profile.company,
					roleId: profile.roleId
				},
				apiToken: authService.signWebApiToken(profile, clientIp),
				accessToken: profile.accessToken,
				refreshToken: profile.refreshToken
			};
			res.send(JSON.stringify(respData));
		})(req, res, next);
	}

	verifyEmployeeId = (req, res, next) => {
		this.passport.authenticate('LocalStrategy', function (err, user, info) {
			let respObj = new BaseResponse();
			if (err) {
				err.message = 'User Unauthorized!';
				err.status = 401;
				return next(err);
			}
			res.setHeader('Content-Type', 'application/json');

			if (!user) {
				respObj.setError(info.code, info.message);
				return res.status(200).json(respObj);
			}

			let authService = new AuthService();
			let clientIp = req.headers['x-forwarded-for'] || req.connection.remoteAddress;
			let respData = {
				userInfo: {
					uid: user.uid,
					name: user.fullname,
					roleId: user.roleId
				},
				apiToken: authService.signWebApiToken(
					{
						...user,
						name: user.fullname,
						roleId: user.roleId
					},
					clientIp),
				accessToken: 'NONE',
				refreshToken: 'NONE'
			};
			res.send(JSON.stringify(respData));

		})(req, res, next);
	}

	verifyApiAuth = (req, res, next, byPassOnError = false) => {
		return this.passport.authenticate('JwtWebApiStrategy', { session: false }, this.handleStrategyResponse(req, res, next, byPassOnError))(req, res, next);
	}

	/**
	 * get Graphql Tool User (GTU) for debug Graphiql 
	 * @returns 
	 */
	getGTU = async () => await userService.getUserByUid('dt0294');

	handleStrategyResponse(req, res, next, byPassOnError = false) {
		return async (err, user, info) => {
			let respObj = new BaseResponse();
			// get user for checking on API-token usage
			req.user = user;
			// add request log after first, then check the request limit
			const operationName = req.body.operationName;
			if (Common.BannedAPI.includes(operationName)) {
				await reqRestrictService.addReqInfo(user.uid, operationName, new Date());
				const { count, isRestricted } = await reqRestrictService.checkRestricted(user.uid);
				if (isRestricted) {
					respObj.setError(429, 'Too Many Requests');
					if (count == Common.RequestLimit + 1) {
						// Send Notify to Slack
						Sentry.captureMessage(`User ${user.uid} has been reached the limit of ${Common.RequestLimit} requests/min`);
					}
					return res.status(429).json(respObj);
				}
			}

			if (!byPassOnError) {
				if (err) {
					respObj.setError(500, err.message);
					return res.status(500).json(respObj);
				}

				if (info) {
					let status = info.code ? info.code : 401;
					respObj.setError(status, info.message);
					return res.status(status).json(respObj);
				}
			}

			next();
		};
	}

}

module.exports = Authorization;