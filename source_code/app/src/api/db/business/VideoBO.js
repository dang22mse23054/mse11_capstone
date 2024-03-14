const Video = require('modelDir/Video');
const Database = require('dbDir');
const LogService = require('commonDir/logger');
const log = LogService.getInstance();

const readonlyDb = Database.getROInstance();

module.exports = class VideoBO {
	constructor(outsideTransaction) {
		this.outTrx = outsideTransaction;
	}

	getById = (id) => {
		let stm = Video.query(readonlyDb).first();

		if (id) {
			stm.where('id', id);
		}

		return stm.then(row => row || null);
	};

	getAll = () => {
		// Simple the above query
		let stm = Video.query(readonlyDb).select();

		log.debug(stm.toKnexQuery().toSQL());
		return stm.then(rows => rows ? rows : null);
	}

};
