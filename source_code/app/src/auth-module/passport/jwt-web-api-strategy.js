// load environment variables
require('dotenv').config();

const JwtStrategy = require('passport-jwt').Strategy;
const ExtractJwt = require('passport-jwt').ExtractJwt;
const { userService } = require('../../api/services');
const LogService = require('commonDir/logger');
const log = LogService.getInstance();

const JwtWebApiStrategy = {
	apply: (passport) => {

		let opts = {
			secretOrKey: process.env.JWT_SECRET,
			jwtFromRequest: ExtractJwt.fromAuthHeaderAsBearerToken(),
			// issuer: process.env.JWT_ISSUER,
			// audience: process.env.JWT_AUDIENCE,
			passReqToCallback: true
		};

		let jwtStrategy = new JwtStrategy(opts, JwtWebApiStrategy.verify);

		passport.use('JwtWebApiStrategy', jwtStrategy);
	},

	verify: async (req, decodedToken, done) => {
		log.debug('-------- BEGIN ---------- JWT Token Info ------------');

		try {
			// ===== Validate the token's signature ===== //
			// verify token info 
			let isNotValidIssuer = decodedToken.iss != process.env.JWT_ISSUER;
			let remoteIp = req.headers['x-forwarded-for'] || req.connection.remoteAddress;
			log.debug(`req.headers['x-forwarded-for'] = ${req.headers['x-forwarded-for']}`);
			log.debug(`req.connection.remoteAddress = ${req.connection.remoteAddress}`);
			log.debug(`remoteIp = ${remoteIp}`);
			// Reference https://stackoverflow.com/questions/10849687/express-js-how-to-get-remote-client-address

			if (isNotValidIssuer) {
				return done(null, false, { code: 401, message: 'Invalid token' });
			}

			// verify token expired time
			let currentTime = new Date();
			let expiredTime = new Date(decodedToken.exp * 1000);

			log.debug(`currentTime = ${currentTime}`);
			log.debug(`expiredTime = ${expiredTime}`);
			log.debug(`decoded exp = ${decodedToken.exp}`);

			let isExpired = currentTime >= expiredTime;
			// isExpired = true;

			if (isExpired) {
				log.debug('RESULT: IS Expired');
				return done(null, false, { code: 401, message: 'Token has been expired' });
			}

			// verify info callback function
			let uid = decodedToken.sub;
			log.debug(`uid = ${uid}`);

			let user = await userService.getUserByUid(uid);

			if (user == null) {
				return done(null, false, { code: 403, message: 'Permission denied' });
			}

			return done(null, user);

		} catch (err) {
			log.error(err);
			return done(err);

		} finally {
			log.debug('--------- END ----------- JWT Token Info ------------');
		}
	}
};

module.exports = JwtWebApiStrategy;