const Database = require('../../../database');
const { Model } = require('objection');
const cursorMixin = require('objection-cursor');

// Init DB & Binding Models to DB instance
const knex = Database.getInstance();
Model.knex(knex);

let CursorSetting = {
	default: {
		limit: 30,
		results: false,
		nodes: true,
	}
};
CursorSetting.forScrolling = {
	...CursorSetting.default,
	pageInfo: {
		total: false,
		remaining: false,
		remainingBefore: false, // Remaining amount of rows before current results
		remainingAfter: false, // Remaining amount of rows after current results
		hasNext: false,
		hasPrevious: false,
	}
};
CursorSetting.forPaging = {
	...CursorSetting.default,
	pageInfo: {
		total: true,
		remaining: false,
		remainingBefore: false, // Remaining amount of rows before current results
		remainingAfter: false, // Remaining amount of rows after current results
		hasNext: false,
		hasPrevious: false,
	}
};

const cursorForPaging = cursorMixin(CursorSetting.forPaging);
const cursorForScrolling = cursorMixin(CursorSetting.forScrolling);

class BaseModel extends Model {
	static tableName = 'need_to_override'

	// Only for input validation
	static jsonSchema = {
		type: 'object',
		required: [],
		properties: {}
	}

	// remove data is null/undefined or have not been in properties
	static filterPropsData = function (param, removeNull = false) {
		const properties = this.jsonSchema.properties;
		const objRt = {};
		if (properties) {
			for (const [key, value] of Object.entries(properties)) {
				if ((!removeNull && param[key] == null) || (param[key] != null && param[key] != undefined)) {
					objRt[key] = param[key];
				}
			}
		}
		return objRt;
	}

}

const PagingModel = cursorForPaging(BaseModel);
const ScrollingModel = cursorForScrolling(BaseModel);

module.exports = {
	BaseModel,
	PagingModel,
	ScrollingModel,
	cursorForPaging,
	cursorForScrolling,
	HasOneRelation: Model.HasOneRelation,
	HasManyRelation: Model.HasManyRelation,
	ManyToManyRelation: Model.ManyToManyRelation,
	BelongsToOneRelation: Model.BelongsToOneRelation,
	HasOneThroughRelation: Model.HasOneThroughRelation,
};