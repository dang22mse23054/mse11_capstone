const { categoryService } = require('apiDir/services');

const Query = {
	getCategories: async (obj, {}, context, info) => {
		return await categoryService.getCategories();
	},
};

module.exports = { Query };