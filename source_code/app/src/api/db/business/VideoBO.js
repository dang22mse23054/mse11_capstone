const { raw } = require('objection');
const Video = require('modelDir/Video');
const VideoCategory = require('modelDir/VideoCategory');
const Category = require('modelDir/Category');
const Database = require('dbDir');
const LogService = require('commonDir/logger');
const log = LogService.getInstance();
const { PVideo, SVideo } = Video;
const { Sorting, VideoStatus } = require('commonDir/constants');
const Utils = require('commonDir/utils/Utils');

const readonlyDb = Database.getROInstance();

module.exports = class VideoBO {
	constructor(outsideTransaction) {
		this.outTrx = outsideTransaction;
	}

	getBy = ({age, gender} = {}, isEnabled = true) => {
		let stm = Video.query(readonlyDb).alias('v')
			.distinct([
				'v.id',
				'v.refFileName',
			])
			// VideoCategory
			.innerJoin(`${VideoCategory.tableName} as vc`, 'vc.videoId', 'v.id')
			// Category
			.innerJoin(`${Category.tableName} as c`, 'vc.categoryId', 'c.id')

		stm.where('isEnabled', isEnabled)
			.whereNull('v.deletedAt');

		if (age != null) {
			stm.where('c.age', age);
		}

		if (gender != null) {
			stm.where('c.gender', gender);
		}

		log.debug(stm.toKnexQuery().toSQL());

		return stm.then(row => row || null);
	};

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

	// First way: Using callback to use transaction
	insert = async (video) => {
		// Using trx as a transaction object:
		return Database.transaction(this.outTrx, (trx) => {
			try {
				const stm = Video.query(trx)
					.insert(Utils.removeNullData(Video.filterPropsData(video)));

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

	// Second way: Using transaction directly
	update = async (video) => {

		// Using trx as a transaction object:
		const trx = this.outTrx ? this.outTrx : await Database.transaction();

		try {
			let data = await trx(Video.tableName)
				.update(Video.filterPropsData({
					...video,
					updatedAt: new Date()
				}))
				.where('id', video.id)
				.transacting(trx);

			// If using Trx from outside, just response data for the next process at caller
			if (this.outTrx) {
				return data;
			}
			return trx.commit(data);

		} catch (error) {
			if (this.outTrx) {
				throw error;
			}
			log.error(error);
			return trx.rollback(error);
		}
	}

	delete = async (ids) => {
		// Using trx as a transaction object:
		const trx = this.outTrx ? this.outTrx : await Database.transaction();

		try {
			let stm = trx(Video.tableName).update({ deletedAt: new Date() });

			if (ids && ids.length > 0) {
				stm.whereIn('id', ids);
			}

			let data = await stm.transacting(trx);

			// If using Trx from outside, just response data for the next process at caller
			if (this.outTrx) {
				return data;
			}

			return trx.commit(data);

		} catch (error) {
			if (this.outTrx) {
				throw error;
			}
			log.error(error);
			return trx.rollback(error);
		}
	}

	search = async ({ keyword, startDate, endDate, categories, eagerLoad, status, isAscSIDSorting } = {}, { cursor, nextCursor, prevCursor } = {}, limit) => {
		//CONVERT_TZ(CONCAT(DATE(CONVERT_TZ(NOW(), 'SYSTEM', sched.timezone)), ' ', sd.hour, ':', sd.minute), sched.timezone, 'SYSTEM') as deadline 
		let start = new Date();
		const subQuery = Video.query(readonlyDb).alias('v').as('v');

		// choose PagingModel or ScrollingModel
		// NOTE: 
		// - PagingModel will calc total records => slow to get result on case of BigData
		// - ScrollingModel will only fetch records (not calc total records) => fast
		//  ==> should call PagingModel for the first time search by options, use ScrollingModel for the next time

		let videoModel = PVideo;
		if (cursor || nextCursor || prevCursor) {
			videoModel = cursor ? PVideo : SVideo;
		}

		let stm = videoModel.query(readonlyDb).alias('v')
			.distinct([
				'v.*',
				// raw('COUNT(DISTINCT(sp.id)) as processesCount'),
				raw('GROUP_CONCAT(DISTINCT(vc.categoryId)) as categoryIds'),
			])
			.from(subQuery)

			// Category
			.leftJoin(`${VideoCategory.tableName} as vc`, 'vc.videoId', 'v.id')

			// Group schedule to get GROUP_CONCAT Media list
			.groupBy('v.id');

		if (status != null) {
			switch (status) {
				case VideoStatus.STOPPED:
					subQuery.whereNotNull('v.deletedAt');
					break;

				default:
					subQuery.whereNull('v.deletedAt')
						.where('v.isEnabled', VideoStatus.PLAYING == status ? 1 : 0);
					break;
			}
		}

		if (startDate != null) {
			subQuery.where('v.createdAt', '>=', startDate);
		}

		if (endDate != null) {
			subQuery.where('v.createdAt', '<=', endDate);
		}

		if (keyword) {
			const parts = keyword.trim().split(' ');

			subQuery.where(function () {
				this.orWhere('v.title', 'LIKE', `%${keyword}%`)

				if (!isNaN(keyword)) {
					this.orWhere('v.id', keyword);
				}

				// if keyword has space (eg: 'abc xyz 123')
				if (parts.length > 1) {
					parts.forEach(value => {
						this.orWhere('v.title', 'LIKE', `%${value}%`)

						if (!isNaN(value)) {
							this.orWhere('v.id', value);
						}
					});
				}
			});
		}

		if (categories) {
			subQuery.whereIn('v.id', VideoCategory.query(readonlyDb).alias('vc').select('vc.videoId').whereIn('vc.categoryId', categories));
		}

		if (eagerLoad) {
			stm.withGraphFetched(eagerLoad);
		}

		stm.orderBy('v.deletedAt');
		stm.orderBy('v.isEnabled', 'desc');

		// for objection-cursor
		// Default is desc sorting
		stm.orderBy('v.id', isAscSIDSorting ? Sorting.ASC : Sorting.DESC);

		if (limit) {
			stm.limit(limit);
		}

		if (nextCursor) {
			stm.nextCursorPage(nextCursor);

		} else if (prevCursor) {
			stm.previousCursorPage(prevCursor);

		} else {
			stm.cursorPage(cursor);
		}

		// Print complete sql query
		log.debug(stm.toKnexQuery().toQuery());
		const result = await stm.then(row => row || null);

		let end = new Date();
		log.debug(`Finished SQL in ${end - start} ms`);

		return result;
	}

};
