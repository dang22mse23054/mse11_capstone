const { BaseRoutes } = require('./base-routes');

class S3Routes extends BaseRoutes {
	constructor(app, router) {
		super(app, router);
		this.initRouter();
	}

	initRouter() {
		const { fileController } = require('../controllers/rest');

		// ========= BEGIN Routing ========= //
		this.router.get('/:objKey([/_.0-9A-Za-z]{20,200})', fileController.streamFile);

		// ========= END Routing ========= //
	}
}

module.exports = S3Routes;