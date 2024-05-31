const { BaseModel: Model } = require('./_base');

class Log extends Model {

	// Required
	static get tableName() {
		return 'log';
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
				id: { type: 'string', minLength: 1, maxLength: 100 },
				videoId: { type: 'integer' },
				gender: { type: 'string', minLength: 1, maxLength: 200 },
				age: { type: 'string', minLength: 1, maxLength: 200 },
				happy: { type: 'string', minLength: 1, maxLength: 500 },
			}
		};
	}

	// This object defines the relations to other models.
	static get relationMappings() {
		// Importing models here is one way to avoid require loops.
		const Video = require('./Video');
		return {
			video: {
				relation: Model.BelongsToOneRelation,
				modelClass: Video,
				join: {
					from: `${Log.tableName}.videoId`,
					to: `${Video.tableName}.id`
				}
			},
		};
	}

};

module.exports = Log;
