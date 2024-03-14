const { ScrollingModel: Model } = require('./_base');

module.exports = class User extends Model {

	// Required
	static get tableName() {
		return 'users';
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
				uid: { type: 'string', minLength: 1, maxLength: 20 },
				fullname: { type: 'string', minLength: 1, maxLength: 50 },

				updatedAt: { type: ['string']/*, format: 'date-time'*/ },
			}
		};
	}

};
