const { BaseModel: Model } = require('./_base');

class Statistic extends Model {

	// Required
	static get tableName() {
		return 'statistic';
	}

	// Optional: default is 'id'
	// Value: a column name (or a array of columns' names)
	static get idColumn() {
		return ['videoId', 'createdAt'];
	}

	// Only for input validation
	static get jsonSchema() {
		return {
			type: 'object',
			required: [],

			properties: {
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
					from: `${Statistic.tableName}.videoId`,
					to: `${Video.tableName}.id`
				}
			},
		};
	}

};

module.exports = Statistic;
