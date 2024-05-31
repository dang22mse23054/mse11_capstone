const { BaseModel: Model, cursorForPaging, cursorForScrolling } = require('./_base');

class Video extends Model {

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
				isEnabled: { type: 'boolean' },
				refFileName: { type: 'string', minLength: 1, maxLength: 500 },
				refFilePath: { type: 'string', minLength: 1, maxLength: 500 },

				deletedAt: { type: ['string', 'null']/*, format: 'date-time'*/ },
			}
		};
	}

	// This object defines the relations to other models.
	static get relationMappings() {
		// Importing models here is one way to avoid require loops.
		const Category = require('./Category');
		const VideoCategory = require('./VideoCategory');
		return {
			categories: {
				relation: Model.ManyToManyRelation,
				modelClass: Category,
				join: {
					from: `${Video.tableName}.id`,
					through: {
						from: `${VideoCategory.tableName}.videoId`,
						to: `${VideoCategory.tableName}.categoryId`
					}
				},
				to: `${Category.tableName}.id`
			},
		};
	}

};

const PVideo = cursorForPaging(Video);
const SVideo = cursorForScrolling(Video);

module.exports = Video;
module.exports.PVideo = PVideo;
module.exports.SVideo = SVideo;
