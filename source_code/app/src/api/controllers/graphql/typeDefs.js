const { mergeTypeDefs } = require('@graphql-tools/merge');
const { typeDefs: baseTypes } = require('./base');

module.exports = mergeTypeDefs([
	baseTypes,
]);