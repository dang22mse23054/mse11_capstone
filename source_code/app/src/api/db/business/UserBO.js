const User = require('modelDir/User');
const Database = require('dbDir');
const Utils = require('commonDir/utils/Utils');
const LogService = require('commonDir/logger');
const log = LogService.getInstance();

const db = Database.getInstance();
const readonlyDb = Database.getROInstance();

module.exports = class UserBO {
	constructor(outsideTransaction) {
		this.outTrx = outsideTransaction;
	}

	search = (keyword, withDeleted, nextCursor, limit = 10) => {
		let normalizedKeyword;
		let normalizedUids;
		const normalizedCondition = {
			removeSpace: true,
			toLowerCase: true,
			trim: true,
			toFullWidth: true,
			removeSpecialChar: true
		};

		if (keyword && keyword != '') {
			normalizedKeyword = Utils.strNormalize(keyword, normalizedCondition);
		}
		// convert keyword to array
		if (keyword.includes(',')) {
			normalizedUids = keyword.split(',').map(e => Utils.strNormalize(e, normalizedCondition));
		}

		let stm = User.query(readonlyDb).select()
			.where(function () {
				//case: keyword is an array uids
				if (normalizedUids?.length > 1) {
					this.whereIn('uid', normalizedUids);
				}
				// normal case: search by keyword
				else {
					this.orWhere('normalize', 'like', `%${keyword}%`);
					if (normalizedKeyword) {
						this.orWhere('normalize', 'like', `%${normalizedKeyword}%`);
					}
					this.orWhere('email', 'like', `%${keyword}%`);
					this.orWhere('uid', 'like', `%${keyword}%`);
				}
			});
		// also get deleted uses
		if (!withDeleted) {
			stm.whereNull('deletedAt');
		}

		stm.orderBy('id')
			.limit(limit)
			.cursorPage(nextCursor);

		// log.debug(stm.toSQL());

		return stm.then(rows => rows ? rows : null);
	}

	setLastAccess = (uid) => {
		// Using trx as a transaction object:
		return Database.transaction(function (trx) {
			let now = new Date();
			db.update({ lastAccess: now })
				.into(User.tableName)
				.where('uid', '=', uid)
				.transacting(trx)
				.then((data) => {
					return trx.commit(data);
				})
				.catch((err) => {
					return trx.rollback(err);
				});
		});
	}

	/**
	 * 
	 * @param {*} uid 
	 * @param {*} isDeleted If undefined then get both deleted and non-deleted users.
	 * @returns 
	 */
	getByUid = (uid, isDeleted = null) => {
		const stm = User.query(readonlyDb);
		if (Array.isArray(uid)) {
			stm.whereIn('uid', uid);

		} else {
			stm.first().where('uid', uid);
		}

		if (isDeleted != null) {
			isDeleted ? stm.whereNotNull('deletedAt') : stm.whereNull('deletedAt');
		}

		log.debug(stm.toKnexQuery().toSQL());

		return stm.then(row => row || null);
	}

	getUidsExcept = (uid) => {
		const stm = User.query(readonlyDb).select(['uid'])
			.whereNull('deletedAt');

		if (Array.isArray(uid)) {
			stm.whereNotIn('uid', uid);

		} else {
			stm.first().whereNot('uid', uid);
		}
		log.debug(stm.toKnexQuery().toSQL());

		return stm.then(row => row || null);
	}

	getByIds = async (ids, isDeleted = null) => {
		const trx = this.outTrx ? this.outTrx : await Database.transaction();
		const stm = trx(User.tableName).select();

		if (Array.isArray(ids)) {
			stm.whereIn('id', ids);

		} else {
			stm.first().where('id', ids);
		}

		if (isDeleted == true) {
			stm.whereNotNull('deletedAt');

		} else if (isDeleted == false) {
			stm.whereNull('deletedAt');
		}

		return stm.then(row => row || null);
	}

	getById = (id, isDeleted = null) => {
		return this.getByIds(id, isDeleted);
	}

	update = async ({ id, fullname, chatworkAccId, chatworkAccName, slackAccId } = {}) => {
		// Using trx as a transaction object:
		const trx = this.outTrx ? this.outTrx : await Database.transaction();
		const now = new Date();
		try {
			const stm = trx(User.tableName)
				.update(User.filterPropsData({
					// fullname,
					chatworkAccId,
					chatworkAccName,
					slackAccId,
					updatedAt: now
				}))
				.where('id', '=', id);

			log.debug(stm.toSQL());

			let data = await stm.transacting(trx);

			// If using Trx from outside, just response data for the next process at caller
			const result = {
				data,
				updatedAt: now
			};
			if (this.outTrx) {
				return result;
			}
			return trx.commit(result);

		} catch (error) {
			if (this.outTrx) {
				throw error;
			}
			log.error(error);
			return trx.rollback(error);
		}
	}

	/* prefix '_load' is using for DataLoader */
	_loadByUids = (keys, mode = 'all') => {
		let stm = User.query(readonlyDb).whereIn('uid', keys);
		switch (mode) {
			case 'active':
				stm.whereNull('deletedAt');
				break;
			case 'deleted':
				stm.whereNotNull('deletedAt');
				break;
		}
		log.debug(stm.toKnexQuery().toQuery());

		// DataLoader result's length must the same with keys 
		// (ex: keys.length is 3 => result.length must be 3)
		// DataLoader result's order must following to keys' order 
		// (ex: keys = [1,2,4] and rows = [4,1,2] => must sort result to get [1,2,4])
		return stm.then(rows => keys.map(uid => rows.find(x => x['uid'].toLowerCase() == uid)));
	}


};
