const { BaseRoutes } = require('./base-routes');
const BaseResponse = require('../dto/BaseResponse');

class DeviceApiRoutes extends BaseRoutes {
	constructor(app, router) {
		super(app, router);
		this.initRouter();
	}

	initRouter() {
		const { adsAdviceController } = require('../controllers/rest');

		// ========= BEGIN Routing ========= //
		this.router.use('/ads', this.subRoutes(subRouter => {
			subRouter.post('/advice', adsAdviceController.advice);
			subRouter.get('/all', adsAdviceController.getAll);
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

module.exports = DeviceApiRoutes;