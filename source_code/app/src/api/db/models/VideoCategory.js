const { BaseModel } = require('./_base');

module.exports = class VideoCategory extends BaseModel {

	// Required
	static get tableName() {
		return 'video_category';
	}
	
	// Optional: default is 'id'
	// Value: a column name (or a array of columns' names)
	static get idColumn() {
		return ['videoId', 'categoryId'];
	}

	// Only for input validation
	static get jsonSchema() {
		return {
			type: 'object',
			required: [],

			properties: {
				videoId: { type: 'integer' },
				categoryId: { type: 'integer' },
			}
		};
	}

};
