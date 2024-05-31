const { BaseModel } = require('./_base');

class Category extends BaseModel {

	// Required
	static get tableName() {
		return 'category';
	}

	// Optional: default is 'id'
	// Value: a column name (or a array of columns' names)
	static get idColumn() {
		return 'id';
	}

	// Only for input validation & GraphQL mapping
	static get jsonSchema() {
		return {
			type: 'object',
			required: [],
			properties: {
				id: { type: 'integer' },
				name: { type: 'string', minLength: 1, maxLength: 200 },
				gender: { type: ['integer', 'null'] },
				age: { type: ['integer', 'null'] },
			}
		};
	}
}

module.exports = Category;

