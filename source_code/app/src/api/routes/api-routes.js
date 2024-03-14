const { BaseRoutes } = require('./base-routes');
const { RequestValidator, ValidationType } = require('../middlewares/RequestValidator');
const BaseResponse = require('../dto/BaseResponse');
const { ApiConst, RoutePaths } = require('commonDir/constants');
const { DownloadTypes } = ApiConst;

class ApiRoutes extends BaseRoutes {
	constructor(app, router) {
		super(app, router);
		this.initRouter();
	}

	initRouter() {
		const { downloadController, fileController, statisticController } = require('../controllers/rest');

		// ========= BEGIN Routing ========= //

		this.router.use(`/${RoutePaths.PREFIX.DOWNLOAD}`, this.subRoutes(subRouter => {
			subRouter.use('/task', this.subRoutes(subRouter => {
				subRouter.get('/', RequestValidator.validate(ValidationType.TASK_DOWNLOAD.OTHER), downloadController.downloadTaskCsv());

				subRouter.get(`/${DownloadTypes.Task.BySearchResult}`,
					RequestValidator.validate(ValidationType.TASK_DOWNLOAD.SEARCH_RESULT),
					downloadController.downloadTaskCsv(DownloadTypes.Task.BySearchResult)
				);

				subRouter.post('/cancel/:csvReportId', downloadController.cancelTaskCsv);
				subRouter.post('/remove/:csvReportId', downloadController.removeTaskCsv);
			}));

			// subRouter.use('/schedule', this.subRoutes(subRouter => {
			// 	subRouter.get('/subAPI-1', ...);
			// 	subRouter.post('/subAPI-2', ...);
			// 	subRouter.delete('/subAPI-3', ...);
			// }));
		}));

		this.router.use('/files', this.subRoutes(subRouter => {
			subRouter.post('/upload', fileController.upload);
			subRouter.post('/remove', fileController.remove);
		}));

		this.router.use(`/${RoutePaths.PREFIX.STATISTIC}`, this.subRoutes(subRouter => {
			subRouter.get('/:dashboardId', statisticController.getUrl);
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