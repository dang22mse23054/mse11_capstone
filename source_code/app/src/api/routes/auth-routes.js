require('dotenv').config();

const { BaseRoutes } = require('./base-routes');
const AuthController = require('authDir/controllers/AuthController');
const AuthMiddleware = require('authDir/middlewares/Authorization');
const { RoutePaths } = require('commonDir/constants');

class AuthRoutes extends BaseRoutes {
	constructor(app, router, passport) {
		super(app, router);
		this.passport = passport;
		this.authController = new AuthController();
		this.authMiddleware = new AuthMiddleware(passport);
		this.initRouter();
	}

	initRouter() {
		// ========= BEGIN Routing ========= //
		this.router.get(`/${RoutePaths.PREFIX.AUTH}`, (req, res, next) => {
			return res.redirect('/auth/casso');
		});

		this.router.get(`/${RoutePaths.PREFIX.AUTH}/casso`, this.passport.authenticate('CAssoStrategy'));

		if (process.env.NEXT_PUBLIC_NODE_ENV === 'development') {
			this.router.post('/auth/userid', this.authMiddleware.verifyEmployeeId);
		}


		// Goto auth page based on strategies (Local, OAuth2... )
		// (Current only apply for OAuth2-CASSO, TODO in the future: Local, Google...)
		this.router.get(`/${RoutePaths.PREFIX.AUTH}/check`, this.authController.checkSession);

		// verification login code (OAuth2)
		this.router.post(process.env.CALLBACK_PATH, this.authMiddleware.verifyCAssoCode);

		// logout
		this.router.get(`/${RoutePaths.LOGOUT}`, this.authController.logout);
		// ========= END Routing ========= //
	}
}

module.exports = AuthRoutes;