// load environment variables
require('dotenv').config();

const { ErrorCodes } = require('commonDir/constants');
const LocalStrategy = require('passport-local').Strategy;
const { userService } = require('../../api/services');

const InternalStrategy  =  {
	apply: (passwort) => {
		const localStrategy  = new LocalStrategy(
			{
				usernameField: 'userId',
				passwordField: 'userId', //for only using userId authorization
				passReqToCallback: true
				
			}, InternalStrategy.authEmployeeId);

		passwort.use('LocalStrategy', localStrategy);

	},
	authEmployeeId: async (req, userId, password, done) => {

		try {
			let user = await userService.getUserByUid(userId);
			if  (user === null ) {
				return done(null, false, ErrorCodes.NOT_EXISTED_USER);
			}
			if (user.deletedAt !== null) {
				return done(null, false, ErrorCodes.DELETED_USER);
			}
			
			return done(null, user);
			
		} catch (err) {
			return done(err);
		}
	}
};

module.exports =  InternalStrategy;

