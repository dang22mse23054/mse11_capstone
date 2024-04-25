const { raw } = require('objection');
const Log = require('modelDir/Log');
const Database = require('dbDir');
const LogService = require('commonDir/logger');
const log = LogService.getInstance();
const Utils = require('commonDir/utils/Utils');

const readonlyDb = Database.getROInstance();

module.exports = class LogBO {
	constructor(outsideTransaction) {
		this.outTrx = outsideTransaction;
	}

	// First way: Using callback to use transaction
	insert = async (logData) => {
		// Using trx as a transaction object:
		return Database.transaction(this.outTrx, (trx) => {
			try {
				const stm = Log.query(trx)
					.insert(Utils.removeNullData(Log.filterPropsData(logData)));

				log.debug(stm.toKnexQuery().toSQL());

				// If using Trx from outside, just response data for the next process at caller
				if (this.outTrx) {
					return stm;
				}
				return trx.commit(stm);

			} catch (error) {
				if (this.outTrx) {
					throw error;
				}

				log.error(error);
				return trx.rollback(error);
			}
		});
	}

};
