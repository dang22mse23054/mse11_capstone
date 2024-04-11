const Category = require('modelDir/Category');
const Database = require('dbDir');
const LogService = require('commonDir/logger');
const log = LogService.getInstance();

const readonlyDb = Database.getROInstance();

module.exports = class CategoryBO {
	constructor(outsideTransaction) {
		this.outTrx = outsideTransaction;
	}

	getBy = (condition) => {
		let stm = Category.query(readonlyDb).first();

		const { id = undefined, gender = null, age = null } = condition;

		if (id != null) {
			stm.where('id', id);
		}

		if (gender != null) {
			stm.where('gender', gender);
		}

		if (age != null) {
			stm.where('age', age);
		}

		return stm.then(row => row || null);
	};

	getAll = () => {
		// Simple the above query
		let stm = Category.query(readonlyDb).select();

		log.debug(stm.toKnexQuery().toSQL());
		return stm.then(rows => rows ? rows : null);
	}

};
