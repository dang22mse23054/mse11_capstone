const { raw } = require('objection');
const Stastic = require('modelDir/Stastic');
const Database = require('dbDir');
const LogService = require('commonDir/logger');
const log = LogService.getInstance();
const Utils = require('commonDir/utils/Utils');

const readonlyDb = Database.getROInstance();

module.exports = class StatisticBO {
	constructor(outsideTransaction) {
		this.outTrx = outsideTransaction;
	}

	// First way: Using callback to use transaction
	insert = async (log) => {
		// Using trx as a transaction object:
		return Database.transaction(this.outTrx, (trx) => {
			try {
				const stm = Stastic.query(trx)
					.insert(Utils.removeNullData(Stastic.filterPropsData(stastic)));

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
