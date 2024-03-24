const CategoryBO = require('../db/business/CategoryBO');
const Database = require('dbDir');
const LogService = require('commonDir/logger');
const log = LogService.getInstance();

module.exports = class CategoryService {
	// searchCategories = (name, parentId, nextCursor, limit = 10) => {
	// 	let categoryBO = new CategoryBO();
	// 	try {
	// 		return categoryBO.search(name, parentId, nextCursor, limit);
	// 	} catch (err) {
	// 		log.error(err);
	// 	}
	// 	return null;
	// }

	getBy = (conditions) => {
		let categoryBO = new CategoryBO();
		try {
			return categoryBO.getBy(conditions);
		} catch (err) {
			log.error(err);
		}
		return null;
	}

	getCategories = () => {
		let categoryBO = new CategoryBO();
		try {
			return categoryBO.getAll();
		} catch (err) {
			log.error(err);
		}
		return null;
	}
};