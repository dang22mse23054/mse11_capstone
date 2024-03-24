const { mergeTypeDefs } = require('@graphql-tools/merge');
const { typeDefs: baseTypes } = require('./base');
const { typeDefs: videoTypes } = require('./video');
const { typeDefs: cursorTypes } = require('./cursor');
const { typeDefs: categoryTypes } = require('./category');

module.exports = mergeTypeDefs([
	baseTypes,
	videoTypes,
	cursorTypes,
	categoryTypes,
]);