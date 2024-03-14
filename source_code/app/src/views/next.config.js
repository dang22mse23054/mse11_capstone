const path = require('path');
const appDir = path.resolve(__dirname, '../../');
const env = require('dotenv-defaults');
const UI_VERSION = 'v.0.1';

// Load .env content
env.config({
	// path: path.resolve(webDir, 'webpack-dev-server/.env'),
	path: path.resolve(appDir, '../.env'),
	encoding: 'utf8',
});


module.exports = {
	env: {
		UI_VERSION
	},
	typescript: {
		// !! WARN !!
		// Dangerously allow production builds to successfully complete even if
		// your project has type errors.
		// !! WARN !!
		ignoreBuildErrors: true,
	},
	rewrites: async () => {
		return [
			{
				source: '/oauth2/idpresponse',
				destination: '/',
			},
		];
	},
	webpack: (config, { buildId, dev, isServer, defaultLoaders, webpack }) => {
		// console.log('webpack-config....' + webpack.version);

		if (!dev) {
			config.devtool = 'source-map';
		}
		return config;
	},
	generateBuildId: async () => {
		// You can, for example, get the latest git commit hash here
		return `tasktracker.${UI_VERSION}`;
	},
};