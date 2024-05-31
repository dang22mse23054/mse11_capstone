require('module-alias/register');
const dotenv = require('dotenv');
dotenv.config({ path: '.env' });

const NEXT_PUBLIC_NODE_ENV = process.env.NEXT_PUBLIC_NODE_ENV || 'production';
const PORT = Number(process.env.NODE_PORT) || 8080;

const next = require('next');
const helmet = require('helmet');
const cors = require('cors');
const favicon = require('serve-favicon');
const express = require('express');
const hostValidation = require('host-validation');
const passport = require('authDir/passport');
const LogService = require('commonDir/logger');
const log = LogService.getInstance();
const ErrorObject = require('apiDir/dto/ErrorObject');
const { RoutePaths } = require('commonDir/constants');
const AuthMiddleware = require('authDir/middlewares/Authorization');
const GraphQL = require('apiDir/middlewares/graphql');
const ClientRoutes = require('routeDir/client-routes');
const AuthRoutes = require('routeDir/auth-routes');
const ApiRoutes = require('routeDir/api-routes');
const DeviceApiRoutes = require('routeDir/device-api-routes');
const S3Routes = require('routeDir/s3-routes');

// NextJS folder
const nextApp = next({ dir: './src/views', dev: (NEXT_PUBLIC_NODE_ENV == 'development') });

const authMiddleware = new AuthMiddleware(passport);

const handler = nextApp.getRequestHandler();
global.__basedir = __dirname;
nextApp.prepare().then(() => {
	let server = express();
	const limitSize = process.env.REQUEST_SIZE_LIMIT || '1mb';

	server.use(
		cors({
			origin: process.env.NEXT_PUBLIC_SERVER_DOMAIN,
		}),
		express.json({ limit: limitSize }),
		express.urlencoded({ limit: limitSize }),
		helmet({
			contentSecurityPolicy: false
		})
	);

	// Setting favicon
	server.use(favicon(__dirname + '/src/views/public/img/favicon.png'));

	// Setting robots.txt
	// server.use(robots(__dirname + '/src/views/public/robots.txt'));

	// ========= CSS/JS Resource Configuration ========= //
	server.use('/resources', express.static(__dirname + '/node_modules/'));

	// Setting "static" folder to be static folder
	server.use('/static', express.static(__dirname + '/src/views/public'));

	// ========= HEALTH CHECK ========= //
	server.use(`/${RoutePaths.HEALTH_CHECK}`, (req, res) => {
		res.statusCode = 200;
		res.type('text');
		res.end('Health-check OK');
	});

	// ========= Authentication ========= //

	// Aplly security middlewares
	let validHosts = process.env.ACCEPTED_HOSTS.split('|');
	if (NEXT_PUBLIC_NODE_ENV !== 'production') {
		validHosts.push('127.0.0.1:443', `${process.env.NEXT_PUBLIC_SERVER_DOMAIN}:443`);
	}

	server.use(hostValidation({
		hosts: validHosts,
		fail: (req, res, next) => {
			let err = new Error();
			err.code = 'Forbiden';
			err.status = 403;
			err.message = 'Invalid host';
			next(err);
		}
	}));

	//================ For DEV/Test Only ================//
	if (NEXT_PUBLIC_NODE_ENV !== 'production') {

		// Get Client IP
		server.use((req, res, next) => {
			log.debug('------------ REQUEST IP INFO ----------');
			log.debug(`originalUrl = ${req.originalUrl}`);
			log.debug(`method = ${req.method}`);
			log.debug(`ip = ${req.ip}`);
			log.debug(`headers['x-forwarded-for'] = ${req.headers['x-forwarded-for']}`);
			log.debug(`headers['host'] = ${req.headers['host']}`);
			log.debug(`socket = ${req.socket}`);
			log.debug(`socket.remoteAddress = ${req.socket.remoteAddress}`);
			log.debug('--------- REQUEST IP INFO (END)--------');

			// var ip = req.headers['x-forwarded-for'] ||
			// 	req.connection.remoteAddress ||
			// 	req.socket.remoteAddress ||
			// 	(req.connection.socket ? req.connection.socket.remoteAddress : null);
			// req.session.clientIp = ip.split(',')[0];

			next();
		});

		// ========= For Dev Only: Authentication / Dump data ========= //
		server.get(process.env.CALLBACK_PATH, function (req, res) {
			req.url = '/';
			return handler(req, res);
		});
	}

	server.use(new AuthRoutes(nextApp, new express.Router(), passport).getInstance());

	// ========= GraphQL API  ========= //
	server.use(`/:graphtype(${RoutePaths.PREFIX.GRAPHQL}|${RoutePaths.PREFIX.GRAPHQL_TOOL})`, async (req, res, next) => {
		if (req.params.graphtype == RoutePaths.PREFIX.GRAPHQL) {
			if (!req.body.operationName) {
				return res.status(400).json({ status: 400, message: 'Required operationName' });
			}

			return authMiddleware.verifyApiAuth(req, res, next);
		} else {
			req.user = await authMiddleware.getGTU();

		}

		next();

	}, GraphQL.instance);


	// ========= rest api API  ========= //
	
	server.use(`/${RoutePaths.PREFIX.S3}`, new S3Routes(nextApp, new express.Router({ mergeParams: true })).getInstance());
	
	server.use(`/${RoutePaths.PREFIX.REST_API}`, async (req, res, next) => {
		return authMiddleware.verifyApiAuth(req, res, next);
	}, new ApiRoutes(nextApp, new express.Router({ mergeParams: true })).getInstance());
	
	server.use(`/${RoutePaths.PREFIX.DEVICE_API}`, async (req, res, next) => {
		return authMiddleware.verifyDeviceAuth(req, res, next);
	}, new DeviceApiRoutes(nextApp, new express.Router({ mergeParams: true })).getInstance());

	// matching URL between NextJs with Express
	server.use('/', new ClientRoutes(nextApp, new express.Router({ mergeParams: true })).getInstance());

	//==============================================//


	// ========= Catch Exception ========= //
	server.use('/error/:code([0-9]+)', function (req, res, next) {
		res.statusCode = req.params.code;
		res.locals.errObj = {
			errCode: req.params.code,
			statusCode: req.params.code,
		};
		nextApp.render(req, res, '/_error');
	});

	// The final URL catcher (for website)
	server.get('*', (req, res) => {
		return handler(req, res);
	});

	// Default error handler 
	server.use(function (err, req, res, next) {
		if (err instanceof ErrorObject) {
			log.error(err, err.byPassSentry);
		} else {
			log.error(err.stack ? err.stack : err);
		}

		let statusCode = err.status || 500;
		// Ajax call running
		if (/^(\/local\/|\/api\/)/i.test(req.originalUrl)) {
			res.setHeader('Content-Type', 'application/json');
			res.statusCode = err.status || 500;
			res.send(JSON.stringify({
				errCode: err.code,
				statusCode,
				errMsg: err.message,
				data: err.data
			}));
		}

		res.locals.errObj = {
			errCode: err.code,
			statusCode,
			errMsg: err.message
		};
		nextApp.render(req, res, '/_error');
	});

	// NOTE: Please add all routing above this line
	if (PORT == 443) {
		// ========= Catch Exception ========= //

		const https = require('https');
		const path = require('path');
		const fs = require('fs');

		const certOptions = {
			key: fs.readFileSync(path.resolve('./ssl/server.key')),
			cert: fs.readFileSync(path.resolve('./ssl/server.crt'))
		};
		server = https.createServer(certOptions, server);
	}

	// ====== NodeJS default error handler ====== //
	process.on('uncaughtException', err => {
		log.info(`Uncaught Exception: ${err.message}`);
		process.exit(1);
	});

	process.on('unhandledRejection', (reason, promise) => {
		log.info('Unhandled rejection at ', promise, `reason: ${reason}`);
		process.exit(1);
	});

	process.on('SIGTERM', (/* signal */) => {
		log.info(`Process ${process.pid} received a SIGTERM signal`);
		process.exit(0);
	});

	process.on('SIGINT', (/* signal */) => {
		log.info(`Process ${process.pid} has been interrupted`);
		process.exit(0);
	});

	process.on('beforeExit', code => {
		// Can make asynchronous calls
		setTimeout(() => {
			log.info(`Process will exit with code: ${code}`);
			process.exit(code);
		}, 100);
	});

	process.on('exit', code => {
		// Only synchronous calls
		log.info(`Process exited with code: ${code}`);
	});
	// =========================================== //

	server.listen(PORT, (err) => {
		if (err) { throw err; }

		let webUrl = `http${PORT == 443 ? 's' : ''}://${process.env.NEXT_PUBLIC_SERVER_DOMAIN}${PORT != 443 ? `:${PORT}` : ''}`;
		log.info(`> [${process.env.NEXT_PUBLIC_NODE_ENV}] Server has been already`);
		log.info(`> Express on ${webUrl}`);
		log.info(`> GraphQL on ${webUrl}/${RoutePaths.PREFIX.GRAPHQL}`);
		if (process.env.NEXT_PUBLIC_NODE_ENV !== 'production') {
			log.info(`> GraphTool on ${webUrl}/${RoutePaths.PREFIX.GRAPHQL_TOOL}`);
		}
		log.info(`> at ${new Date()}`);
	});

}).catch((ex) => {
	log.error(ex.stack || ex);
	setTimeout(() => process.exit(1), 3000);
});
