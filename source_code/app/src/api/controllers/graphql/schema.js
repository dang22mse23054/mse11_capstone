const { makeExecutableSchema } = require('@graphql-tools/schema');
const typeDefs = require('./typeDefs');
const resolvers = require('./resolvers');

module.exports = makeExecutableSchema({
	typeDefs,
	resolvers, // optional
	// logger, // optional
	// resolverValidationOptions: {}, // optional
	// parseOptions: {},  // optional
	// inheritResolversFromInterfaces: false  // optional
});