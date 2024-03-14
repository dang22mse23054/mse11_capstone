const dbConfig = require('./db-config');

let dbInstance = null;
let dbROInstance = null;
let dbOldInstance = null;
const Database = {
	getInstance: () => {
		if (!dbInstance) {
			dbInstance = require('knex')(dbConfig.master);
		}
		return dbInstance;
	},

	getROInstance: () => {
		if (!dbROInstance) {
			dbROInstance = require('knex')(dbConfig.readonly);
		}
		return dbROInstance;
	},

	getOldInstance: () => {
		if (!dbOldInstance) {
			dbOldInstance = require('knex')(dbConfig.oldKnex);
		}
		return dbOldInstance;
	},

	isTransaction: (trx) => Boolean(trx && trx.executionPromise),

	transaction: (trxOrCallback = null, callback = null) => {
		const db = Database.getInstance();
		// If 'trxOrCallback' & 'callback' is NULL => return 'Knex transaction' obj
		if (!trxOrCallback && !callback) {
			return db.transaction();
		}

		// If the 'callback' is NULL => 'trxOrCallback' must be callback function
		if (!callback) {
			if (typeof trxOrCallback !== 'function') {
				throw new Error('Invalid transaction call');
			}
			return db.transaction(trxOrCallback);
		}

		// If the 'callback' is NOT NULL => 'trxOrCallback' must be 'Knex transaction' obj
		if (Database.isTransaction(trxOrCallback)) {
			const trx = trxOrCallback;
			return callback(trx);
		}

		return db.transaction(callback);
	}
};

module.exports = Database;