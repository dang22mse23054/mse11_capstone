const { mergeResolvers } = require('@graphql-tools/merge');
const {resolvers: baseResolvers} = require('./base');

module.exports = mergeResolvers([
	baseResolvers,
]);