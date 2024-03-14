const passport = require('passport');
const LocalStrategy = require('./local-strategy');
const JwtWebApiStrategy = require('./jwt-web-api-strategy');

// ======== Apply Authentication Strategy ======== //
LocalStrategy.apply(passport);
JwtWebApiStrategy.apply(passport);

// ======== Serialize ======== //
passport.serializeUser((userInfo, done) => {
	// TODO Get necessay info to save to session in DB
	done(null, userInfo);
});

// ======== Deserialize ======== //
passport.deserializeUser((userInfo, done) => {
	// TODO validate here
	done(null, userInfo);
});

module.exports = passport;