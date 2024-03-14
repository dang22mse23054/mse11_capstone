const { BaseRoutes } = require('./base-routes');
const { RoutePaths } = require('commonDir/constants');

class ClientRoutes extends BaseRoutes {
	constructor(app, router) {
		super(app, router);
		this.initRouter();
	}

	initObj(req) {
		let data = { userInfo: req.user ? req.user : null };
		if (data.userInfo) {
			data.userInfo.login_type = req.session ? req.session.loginType : null;
		}
		return data;
	}

	initRouter() {
		// ========= BEGIN Routing ========= //

		// get original URL from shorten URL
		this.router.get(`/${RoutePaths.LOGIN}`, (req, res) => {
			if (process.env.NEXT_PUBLIC_NODE_ENV === 'development') {
				return this.app.render(req, res, '/login', { code: req.query.code });
			} else {
				return res.redirect('/auth/casso');
			}
		});

		this.router.get('/', (req, res) => {
			// this method receive 4 params: request, response, url, query_params
			this.app.render(req, res, `/${RoutePaths.INDEX}`, this.initObj(req));
		});
		// ========= END Routing ========= //
	}
}

module.exports = ClientRoutes;