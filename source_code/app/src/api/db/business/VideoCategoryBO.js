const VideoCategory = require('modelDir/VideoCategory');
const Database = require('dbDir');
const LogService = require('commonDir/logger');
const log = LogService.getInstance();
const readonlyDb = Database.getROInstance();

module.exports = class VideoCategoryBO {
	constructor(outsideTransaction) {
		this.outTrx = outsideTransaction;
	}

	// Objection not support batch insert/update/upsert 
	// => using default Knex
	upsertMulti = async (upsertData = []) => {
		if (!upsertData || upsertData.length == 0) {
			return null;
		}

		return Database.transaction(this.outTrx, (trx) => {
			try {
				const conflictArr = ['videoId', 'categoryId'];
				const updFieldsOnConflict = ['categoryId'];

				let stm = trx(VideoCategory.tableName)
					.insert(upsertData)
					.onConflict(conflictArr).merge(updFieldsOnConflict);

				// log.debug(stm.toSQL())

				if (this.outTrx) {
					return stm;
				}
				// If NOT using Trx from outside, just commit
				return trx.commit();

			} catch (error) {
				if (this.outTrx) {
					throw error;
				}

				log.error(error);
				return trx.rollback(error);
			}
		});
	}

	delete = async ({ videoId, categoryIds }) => {
		return Database.transaction(this.outTrx, (trx) => {
			try {
				let stm = VideoCategory.query(trx).delete();

				if (videoId) {
					stm.where('videoId', videoId);
				}

				if (categoryIds && categoryIds.length > 0) {
					stm.whereIn('categoryId', categoryIds);
				}

				if (this.outTrx) {
					return stm;
				}
				// If NOT using Trx from outside, just commit
				return trx.commit();

			} catch (error) {
				if (this.outTrx) {
					throw error;
				}

				log.error(error);
				return trx.rollback(error);
			}
		});
	}

	getByVideo = (videoId, selectedColumns) => {
		let stm = VideoCategory.query(readonlyDb);

		if (selectedColumns) {
			stm.select(selectedColumns);
		}

		stm.where('videoId', videoId);
		log.debug(stm.toKnexQuery().toSQL());

		return stm.then(rows => rows || null);
	}
};
