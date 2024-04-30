const { raw } = require('objection');
const Statistic = require('modelDir/Statistic');
const Database = require('dbDir');
const LogService = require('commonDir/logger');
const log = LogService.getInstance();
const moment = require('moment');

const readonlyDb = Database.getROInstance();

module.exports = class StatisticBO {
	constructor(outsideTransaction) {
		this.outTrx = outsideTransaction;
	}

	get = (videoId, fromTimeStr = null, toTimeStr = null) => {
		
		if (!videoId) {
			return null
		}

		const fromTime = (fromTimeStr ? moment(fromTimeStr) : moment().startOf('day')).format('YYYYMMDDHH')
		console.log(fromTime)
		let stm = Statistic.query(readonlyDb)
			.where('videoId', videoId)
			.where('group', '>=', fromTime)
			.whereNull('deletedAt');

		if (toTimeStr) {
			const toTime = moment(toTimeStr).format('YYYYMMDDHH');
			stm = stm.where('group', '<=', toTime);
		}

		// log.debug(stm.toKnexQuery().toSQL());
		log.debug(stm.toKnexQuery().toQuery());

		return stm.then(row => row || null);
	};
};
