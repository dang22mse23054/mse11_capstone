const { mergeResolvers } = require('@graphql-tools/merge');
const {resolvers: baseResolvers} = require('./base');
const {resolvers: videoResolvers} = require('./video');
const {resolvers: categoryResolvers} = require('./category');

module.exports = mergeResolvers([
	baseResolvers,
	categoryResolvers,
	videoResolvers,
]);