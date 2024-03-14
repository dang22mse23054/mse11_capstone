const { Query } = require('./query.resolver');

const resolvers = {
	Query
};

const { loadFilesSync } = require('@graphql-tools/load-files');
const typeDefs = loadFilesSync(__dirname, { extensions: ['graphql', 'gql'] });

module.exports = {
	typeDefs,
	resolvers
};
