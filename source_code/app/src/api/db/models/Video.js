const { ScrollingModel: Model } = require('./_base');

module.exports = class Video extends Model {

	// Required
	static get tableName() {
		return 'video';
	}

	// Optional: default is 'id'
	// Value: a column name (or a array of columns' names)
	static get idColumn() {
		return 'id';
	}

	// Only for input validation
	static get jsonSchema() {
		return {
			type: 'object',
			required: [],

			properties: {
				id: { type: 'integer' },
				title: { type: 'string', minLength: 1, maxLength: 200 },
				path: { type: 'string', minLength: 1, maxLength: 500 },

				deletedAt: { type: ['string', 'null']/*, format: 'date-time'*/ },
			}
		};
	}

};
