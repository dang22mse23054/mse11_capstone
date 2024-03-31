const { BaseRoutes } = require('./base-routes');
const BaseResponse = require('../dto/BaseResponse');

class ApiRoutes extends BaseRoutes {
	constructor(app, router) {
		super(app, router);
		this.initRouter();
	}

	initRouter() {
		const { fileController } = require('../controllers/rest');

		// ========= BEGIN Routing ========= //

		this.router.use('/files', this.subRoutes(subRouter => {
			subRouter.post('/upload', fileController.upload);
			subRouter.post('/remove', fileController.remove);
		}));

		// Default return in case of unknown URL 
		this.router.all('*', (req, res) => {
			let respObj = new BaseResponse();
			respObj.setError(404, 'Invalid request');
			return res.status(404).json(respObj);
		});

		// ========= END Routing ========= //
	}
}

module.exports = ApiRoutes;