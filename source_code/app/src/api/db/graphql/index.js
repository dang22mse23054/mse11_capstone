const UserLoader = require('./UserLoader');

const SystemCache = {
	user: {
		loader: new UserLoader(),
		clearAll: () => SystemCache.user.loader.clearAll(),
		clearById: (id) => SystemCache.user.loader.clear(id)
	},
};

module.exports = {
	UserLoader,
	SystemCache,

	initAllLoaders: () => ({
		// userLoader: new UserLoader(),
		userLoader: SystemCache.user.loader,
		
	})
};