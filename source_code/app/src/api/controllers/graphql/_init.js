const LogService = require('commonDir/logger');
const log = LogService.getInstance();
// How to create chain (middleware of GraphQL) https://the-guild.dev/graphql/tools/docs/resolvers-composition
const { composeResolvers } = require('@graphql-tools/resolvers-composition');

const initImport = (moduleName, path) => {

	log.info(`[${moduleName}] - GraphQL def files are loading...`);
	/**
	 * Registry Type Definitions
	 */
	const { loadFilesSync } = require('@graphql-tools/load-files');
	const typeDefs = loadFilesSync(path, { extensions: ['graphql', 'gql'] });

	/**
	 * Registry Resolver
	 * 
	 * NOTE:
	 * const ResolverName = require(file_path)
	 * "ResolverName" must be the same with "TypeName"
	 */

	log.info(`[${moduleName}] - Resolvers files are loading...`);

	const resolvers = {};

	try {
		const { Type, ExtTypes } = require(`${path}/types.resolver`);

		if (Type) {
			resolvers[moduleName] = Type;
		}

		if (ExtTypes) {
			for (var typeName in ExtTypes) {
				resolvers[typeName] = ExtTypes[typeName];
				log.info(`[${moduleName}] - '[ext] ${typeName}' has been Added`);
			}
		}
	} catch (err) {
		log.info(`[${moduleName}] - 'Type' resolvers are NOT found`);
	}

	try {
		const { Query, Middleware } = require(`${path}/query.resolver`);
		if (Query) {
			resolvers.Query = Middleware ? composeResolvers({ Query }, Middleware).Query : Query;
		}
	} catch (err) {
		log.info(`[${moduleName}] - 'Query' resolvers are NOT found`);
	}

	try {
		const { Mutation, Middleware } = require(`${path}/mutation.resolver`);
		if (Mutation) {
			resolvers.Mutation = Middleware ? composeResolvers({ Mutation }, Middleware).Mutation : Mutation;
		}
	} catch (err) {
		log.info(`[${moduleName}] - 'Mutation' resolvers are NOT found`);
	}
	return {
		typeDefs,
		resolvers
	};

};

module.exports = {
	initImport
};