const DataLoader = require('dataloader');
const LogService = require('commonDir/logger');
const log = LogService.getInstance();
const Database = require('dbDir');

const readonlyDb = Database.getROInstance();

module.exports = class BaseBO {

	constructor(_class, primaryKeyColumn = _class.idColumn) {
		const { one, many, _instance } = this.#initLoader(_class, primaryKeyColumn);
		this.one = one;
		this.many = many;
		this._instance = _instance;
	}

	#initLoader = (_class, primaryKeyColumn) => {
		const baseLoader = new DataLoader(async (keys) => {
			let stm = _class.query(readonlyDb).whereIn(_class.idColumn, keys);
			log.debug(stm.toKnexQuery().toSQL());
			return stm.then(rows => keys.map(key => rows.find(x => x[primaryKeyColumn] == key)));
		});

		return {
			one: async (id) => id ? baseLoader.load(id.toString()) : null,
			many: async (ids) => ids ? baseLoader.loadMany(ids.map(id => id?.toString())) : null,
			_instance: baseLoader
		};
	}

	defaultClearAll = () => this._instance.clearAll()
	clear = (key) => this._instance.clear(key)
};