const {BaseRoutes} = require('./base-routes');

class ErrorRoutes extends BaseRoutes{
	constructor(app, router){
		super(app, router);
		this.initRouter();
	}

	initRouter() {
		// ========= BEGIN Routing ========= //

		this.router.get('/403', (req, res, next) => {
			let error = {
				code: 'FORBIDDEN',
				status: 403,
				message: 'Permission Denied'
			};
			next(error);
		});

		this.router.get('/404', (req, res, next) => {
			let error = {
				code: 'NOTFOUND',
				status: 404,
				message: 'Page Not Found'
			};
			next(error);
		});

		this.router.get('/401', (req, res, next) => {
			let error = {
				code: 'UNAUTHORIZED',
				status: 401,
				message: 'Unauthorized'
			};
			next(error);
		});

		this.router.get('/423', (req, res, next) => {
			let error = {
				code: 'BLOCKED',
				status: 423,
				message: 'You Are Blocked'
			};
			next(error);
		});

		// ========= END Routing ========= //
	}
}

module.exports = ErrorRoutes;