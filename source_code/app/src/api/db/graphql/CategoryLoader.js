const Category = require('modelDir/Category');
const CategoryBO = require('../business/CategoryBO');
const BaseLoader = require('./_base');

module.exports = class CategoryLoader extends BaseLoader {

	constructor() {
		super(Category);
		this.categoryBO = new CategoryBO();
	}

};