const UserLoader = require('./UserLoader');
const CategoryLoader = require('./CategoryLoader');

const SystemCache = {
	user: {
		loader: new UserLoader(),
		clearAll: () => SystemCache.user.loader.clearAll(),
		clearById: (id) => SystemCache.user.loader.clear(id)
	},
	category: {
		loader: new CategoryLoader(),
		clearAll: () => SystemCache.category.loader.defaultClearAll(),
		clearById: (id) => SystemCache.category.loader.clear(id)
	},
};

module.exports = {
	UserLoader,
	SystemCache,
	CategoryLoader,

	initAllLoaders: () => ({
		// userLoader: new UserLoader(),
		userLoader: SystemCache.user.loader,
		categoryLoader: new CategoryLoader(),
	})
};