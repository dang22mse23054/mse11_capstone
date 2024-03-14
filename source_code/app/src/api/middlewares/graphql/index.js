const LogService = require('commonDir/logger');
const log = LogService.getInstance();
const { graphqlHTTP, getGraphQLParams } = require('express-graphql');
const { parse } = require('graphql');
const { initAllLoaders } = require('apiDir/db/graphql');
const { ErrorCodes } = require('commonDir/constants');

// GraphQL API
const schema = require('graphqlDir/schema');

const Graphql = {
	instance: async (req, res) => {
		const DEFAULT_OPERATION = 'query';
		const { query, operationName } = await getGraphQLParams(req);
		const node = query ? parse(query) : { definitions: [{}] };
		const { operation = DEFAULT_OPERATION, name } = node.definitions[0];

		if (operation != DEFAULT_OPERATION && operationName != name?.value) {
			return res.status(400).json({ status: 400, message: 'Invalid operationName' });
		}

		return graphqlHTTP({
			schema: schema,
			graphiql: (process.env.NEXT_PUBLIC_NODE_ENV != 'production'),
			pretty: true,
			context: {
				req,
				userInfo: req.user,
				...initAllLoaders()
			},
			customFormatErrorFn: (err) => {
				log.error(err, true);
				let message = err.message;
				let code = err.statusCode;

				let errObj = { message, code };

				if (message.startsWith('GRAPHQL|')) {
					errObj = ErrorCodes.Map[message];
					code = errObj.code < 600 ? errObj.code : 500;
				}

				res.statusCode = code || 500;
				return (errObj);
			}
		})(req, res);
	}
};

module.exports = Graphql;