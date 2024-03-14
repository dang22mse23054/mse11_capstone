const express = require ('express');

class BaseRoutes {

	constructor(targetServer, router) {
		this.app = targetServer;
		this.router = router;
		// initData.bind(this)();
		// this.initRouter();
	}

	getInstance() {
		return this.router;
	}

	initRouter() {
		throw new Error('You have to implement the method!');
	}

	subRoutes(callback) {
		let routerFunc = express.Router;
		let subRouter = routerFunc({ mergeParams: true });
		callback(subRouter);
		return subRouter;
	}
}

module.exports = { BaseRoutes };
